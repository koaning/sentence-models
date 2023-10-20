from typing import List, Dict
from pathlib import Path

import spacy
from spacy.language import Language
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from lazylines import read_jsonl, LazyLines
from embetter.text import SentenceEncoder

from .types import Example
from .util import console


class SentenceModel:
    """
    **SentenceModel**

    This object represents a model that can apply predictions per sentence.

    **Usage:**

    ```python
    from sentence_model import SentenceModel

    smod = SentenceModel()
    ```

    You can customise some of the settings if you like, but it comes with sensible defaults.

    ```python
    from sentence_model import SentenceModel
    from embetter.text import SentenceEncoder 
    from sklearn.linear_model import LogisticRegression

    smod = SentenceModel(
        encoder=SentenceEncoder(), 
        clf_head=LogisticRegression(class_weight="balanced"),
        spacy_model="en_core_web_sm", 
        verbose=False
    )
    ```
    """
    def __init__(self, encoder=SentenceEncoder(), clf_head: ClassifierMixin=LogisticRegression(class_weight="balanced"), spacy_model: str="en_core_web_sm", verbose: bool=False):
        self.encoder = encoder
        self.clf_head = clf_head
        self.spacy_model = spacy_model if isinstance(spacy_model, Language) else spacy.load(spacy_model, disable=["ner", "lemmatizer", "tagger"])
        self.classifiers = {}
        self.verbose = verbose
        if verbose:
            console.log("SentenceModel initialized.")
    
    def _prepare_stream(self, stream):
        lines = LazyLines(stream).map(lambda d: Example(**d))
        lines_orig, lines_new = lines.tee()
        labels = {lab for ex in lines_orig for lab in ex.target.keys()}
        
        mapper = {}
        for ex in lines_new:
            if ex.text not in mapper:
                mapper[ex.text] = {}
            for lab in ex.target.keys():
                if lab in mapper[ex.text]:
                    print("WARNING! Duplicate example found: ", ex.text, ex.target)
                mapper[ex.text][lab] = ex.target[lab]
        if self.verbose:
            console.log(f"Found {len(mapper)} examples for {len(labels)} labels.")
        return labels, mapper

    def learn(self, generator) -> "SentenceModel":
        """
        Learn from a generator of examples. Can update a previously loaded model.
        
        Each example should be a dictionary with a "text" key and a "target" key.
        Internally this method checks via this Pydantic model:

        ```python
        class Example(BaseModel):
            text: str
            target: Dict[str, bool]
        ```
        
        As long as your generator emits dictionaries in this format, all will go well.

        **Usage:**

        ```python
        from sentence_model import SentenceModel

        smod = SentenceModel().learn(some_generator)
        ```
        """
        labels, mapper = self._prepare_stream(generator)
        self.classifiers = {lab: clone(self.clf_head) for lab in labels}
        for lab, clf in self.classifiers.items():
            texts = [text for text, targets in mapper.items() if lab in targets]
            labels = [mapper[text][lab] for text in texts]
            X = self.encode(texts)
            clf.fit(X, labels)
            if self.verbose:
                console.log(f"Trained classifier head for {lab=}")
        return self

    def learn_from_disk(self, path: Path) -> "SentenceModel":
        """
        Load a JSONL file from disk and learn from it.
        
        **Usage:**

        ```python
        from sentence_model import SentenceModel

        smod = SentenceModel().learn_from_disk("path/to/file.jsonl")
        ```
        """
        return self.learn(read_jsonl(Path(path)))
    
    def _to_sentences(self, text: str):
        for sent in self.spacy_model(text).sents:
            yield sent.text
    
    def encode(self, texts: List[str]):
        """
        Encode a list of texts into a matrix of shape (n_texts, n_features)
        
        **Usage::**

        ```python
        from sentence_model import SentenceModel

        smod = SentenceModel()
        smod.encode(["example text"])
        ```
        """
        return self.encoder.transform(texts)

    def __call__(self, text):
        """
        Make a prediction for a single text.
        
        **Usage:**

        ```python
        from sentence_model import SentenceModel

        smod = SentenceModel().learn_from_disk("path/to/file.jsonl")
        smod("Predict this. Per sentence!")
        ```
        """
        result = {"text": text}
        sents = list(self._to_sentences(text))
        result["sentences"] = [{"sentence": sent, "cats": {}} for sent in sents]
        X = self.encode(sents)
        for lab, clf in self.classifiers.items(): 
            probas = clf.predict_proba(X)[:, 1]
            for i, proba in enumerate(probas):
                result["sentences"][i]['cats'][lab] = float(proba)
        return result

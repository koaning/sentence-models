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
    This model will train a classifier head for each label on top of an encoder.
    """
    def __init__(self, encoder=SentenceEncoder(), clf_head: ClassifierMixin=LogisticRegression(class_weight="balanced"), spacy_model: str="en_core_web_sm", verbose=False):
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
        Learn from a generator of examples.
        
        Each example should be a dictionary with a "text" key and a "target" key.
        Internally this method checks via this Pydantic model:

        ```python
        class Example(BaseModel):
            text: str
            target: Dict[str, bool]
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
        """Load a JSONL file from disk and learn from it"""
        return self.learn(read_jsonl(path))
    
    def _to_sentences(self, text: str):
        for sent in self.spacy_model(text).sents:
            yield sent.text
    
    def encode(self, texts: List[str]):
        """Encode a list of texts into a matrix of shape (n_texts, n_features)"""
        return self.encoder.transform(texts)

    def __call__(self, text):
        """Make a prediction for a single text"""
        result = {"text": text}
        sents = list(self._to_sentences(text))
        result["sentences"] = [{"sentence": sent, "cats": {}} for sent in sents]
        X = self.encode(sents)
        for lab, clf in self.classifiers.items(): 
            probas = clf.predict_proba(X)[:, 1]
            for i, proba in enumerate(probas):
                result["sentences"][i]['cats'][lab] = float(proba)
        return result

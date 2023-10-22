import srsly 
from typing import List, Optional, Dict, Union
from pathlib import Path

import numpy as np
import spacy
from spacy.language import Language
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from lazylines import read_jsonl, LazyLines
from embetter.text import SentenceEncoder
from skops.io import dump, load

from .types import Example, SentencePrediction
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
    def __init__(self, 
                 encoder=SentenceEncoder(), 
                 clf_head: ClassifierMixin=LogisticRegression(class_weight="balanced"), 
                 spacy_model: str="en_core_web_sm", 
                 verbose: bool=False, 
                 finetuner = None
        ):
        self.encoder = encoder
        self.clf_head = clf_head
        self.spacy_model = spacy_model if isinstance(spacy_model, Language) else spacy.load(spacy_model, disable=["ner", "lemmatizer", "tagger"])
        self.classifiers = {}
        self.verbose = verbose
        if verbose:
            console.log("SentenceModel initialized.")
        self.finetuner = finetuner
    
    # TODO: add support for finetuners
    # def _generate_finetune_dataset(self, examples):
    #     if self.verbose:
    #         console.log("Generating pairs for finetuning.")
    #     all_labels = {cat for ex in examples for cat in ex['target'].keys()}

    #     # Calculating embeddings is usually expensive so only run this once
    #     arrays = {}
    #     for label in all_labels:
    #         subset = [ex for ex in examples if label in ex['target'].keys()]
    #         texts = [ex['text'] for ex in subset]
    #         arrays[label] = self.encoder.transform(texts)

    #     def concat_if_exists(main, new):
    #         """This function is only used here, so internal"""
    #         if main is None:
    #             return new
    #         return np.concatenate([main, new])
        
    #     X1 = None
    #     X2 = None
    #     lab = None
    #     for label in all_labels:
    #         subset = [ex for ex in examples if label in ex['target'].keys()]
    #         labels = [ex['target'][label] for ex in subset]
    #         pairs = generate_pairs_batch(labels)
    #         X = arrays[label]
    #         X1 = concat_if_exists(X1, np.array([X[p.e1] for p in pairs]))
    #         X2 = concat_if_exists(X2, np.array([X[p.e2] for p in pairs]))
    #         lab = concat_if_exists(lab, np.array([p.val for p in pairs], dtype=float))
    #     if self.verbose:
    #         console.log(f"Generated {len(lab)} pairs for contrastive finetuning.")
    #     return X1, X2, lab
        
    # def _learn_finetuner(self, examples):
    #     X1, X2, lab = self._generate_finetune_dataset(examples)
    #     self.finetuner.construct_models(X1, X2)
    #     self.finetuner.learn(X1, X2, lab)

    def _prepare_stream(self, stream):
        lines = LazyLines(stream).map(lambda d: Example(**d))
        lines_orig, lines_new = lines.tee()
        labels = {lab for ex in lines_orig for lab in ex.target.keys()}
        
        mapper = {}
        for ex in lines_new:
            if ex.text not in mapper:
                mapper[ex.text] = {}
            for lab in ex.target.keys():
                mapper[ex.text][lab] = ex.target[lab]
        if self.verbose:
            console.log(f"Found {len(mapper)} examples for {len(labels)} labels.")
        return labels, mapper

    def learn(self, examples: List[Dict]) -> "SentenceModel":
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
        labels, mapper = self._prepare_stream(examples)
        # if self.finetuner is not None:
        #     self._learn_finetuner([{"text": k, "target": v} for k, v in mapper.items()])
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
        return self.learn(list(read_jsonl(Path(path))))
    
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
        console.log(self.finetuner)
        X = self.encoder.transform(texts)
        if self.finetuner is not None:
            return self.finetuner.encode(X) 
        return X

    def __call__(self, text:str):
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
    
    def pipe(self, texts):
        # Currently undocumented because I want to make it faster
        for ex in texts:
            yield self(ex)
    
    def to_disk(self, folder: Union[str, Path]) -> None:
        """
        Writes a `SentenceModel` to disk.
        
        **Usage:**

        ```python
        from sentence_model import SentenceModel

        smod = SentenceModel().learn_from_disk("path/to/file.jsonl")
        smod.to_disk("path/to/model")
        ```
        """
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        for name, clf in self.classifiers.items():
            if self.verbose:
                console.log(f"Writing to disk {folder}/{name}.skops")
            dump(clf, folder / f"{name}.skops")
        if self.finetuner is not None:
            self.finetuner.to_disk(folder)
        settings = {
            "encoder_str": str(self.encoder)
        }
        srsly.write_json("settings.json", settings)

    @classmethod
    def from_disk(self, folder:Union[str, Path], encoder, spacy_model:str="en_core_web_sm", verbose:bool=False) -> "SentenceModel":
        """
        Loads a `SentenceModel` from disk.
        
        **Usage:**

        ```python
        from sentence_model import SentenceModel
        from embetter.text import SentenceEncoder

        # It's good to be explicit with the encoder. Internally this method will check if 
        # the encoder matches what was available during training. The spaCy model is less 
        # critical because it merely splits the sentences during inference.
        smod = SentenceModel.from_disk("path/to/model", encoder=SentenceEncoder(), spacy_model="en_core_web_sm")
        smod("Predict this. Per sentence!")
        ```
        """
        folder = Path(folder)
        keras_files = set(str(s) for s in folder.glob("*.keras"))
        print(keras_files)
        models = {p.parts[-1].replace(".skops", ""): load(p, trusted=True) for p in folder.glob("*.skops")}
        settings = srsly.read_json("settings.json")
        assert str(encoder) == settings["encoder_str"], f"The encoder at time of saving ({settings['encoder_str']}) differs from this one ({encoder})."
        smod = SentenceModel(
            encoder=encoder,
            clf_head=list(models.values())[0], 
            spacy_model=spacy_model, 
            verbose=verbose, 
            finetuner=None
        )
        smod.classifiers = models
        return smod

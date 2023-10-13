from typing import List 

from spacy.language import Language
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from lazylines import read_jsonl, LazyLines

from .types import Example 




class SentenceModel:
    def __init__(self, encoder, clf_head: ClassifierMixin=LogisticRegression(class_weight="balanced"), spacy_model: str="en_core_web_sm"):
        self.encoder = encoder
        self.clf_head = clf_head
        self.spacy_model = spacy_model if isinstance(spacy_model, Language) else spacy.load(spacy_model, disable=["ner", "lemmatizer", "tagger"])
        self.classifiers = {}
    
    def _prepare_stream(self, stream):
        lines = LazyLines(stream).map(lambda d: Exampe(**d))
        lines_orig, lines_new = lines.tee()
        labels = {ex for ex in lines_orig for lab in ex.keys()}
        
        mapper = {}
        for ex in lines_new:
            if ex.text not in mapper:
                mapper[ex.text] = {}
            if ex.target in mapper[ex.text]:
                print("WARNING! Duplicate example found: ", ex.text, ex.target)
            mapper[ex.text][ex.target] = ex.label
        return labels, mapper

    def learn(self, generator):
        labels, mapper = self._prepare_stream(generator)
        self.classifiers = {lab: clone(self.clf_head) for lab in labels}
        for lab, clf in self.classifiers.items():
            texts = [text for text, targets in mapper.items() if lab in targets]
            labels = [mapper[text][lab] for text in texts]
            X = self.encode(texts)
            clf.fit(X, labels)
        return self

    def learn_from_disk(self, path: Path):
        return self.learn(read_jsonl(path))
    
    def _to_sentences(self, text: str):
        for sent in self.spacy_model(text).sents:
            yield sent.text
    
    def encode(self, texts: List[str]):
        return self.encoder.transform(texts)

    def __call__(self, text):
        result = {"text": text}
        sents = list(self._to_sentences(text))
        result["sentences"] = [{"sentence": sent} for sent in sents}]
        X = self.encode(sents)
        for lab, clf self.classifiers.items(): 
            probas = clf.predict_proba(X)[:, 1]
            for i, proba in enumerate(probas):
                result["sentences"][i][lab] = float(proba)
        return result

from typing import Dict, List
from pydantic import BaseModel 


class Example(BaseModel):
    text: str
    target: Dict[str, bool]


class ExamplePair(BaseModel):
    text1: str
    text2: str
    similarity: float


class SentencePrediction(BaseModel):
    sentence: str
    cats: Dict[str, float]


class Predictions(BaseModel):
    text: str
    sentences: List[SentencePrediction]
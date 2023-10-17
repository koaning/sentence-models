from typing import Dict
from pydantic import BaseModel 


class Example(BaseModel):
    text: str
    target: Dict[str, bool]


class ExamplePair(BaseModel):
    text1: str
    text2: str
    similarity: float

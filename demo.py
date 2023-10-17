from sentence_models import SentenceModel
from sentence_models.util import console

smod = SentenceModel()

smod.learn([
    {"text": "hello there", "target": {"greeting": True}}, 
    {"text": "goodbye there", "target": {"greeting": False}},
    {"text": "llms are great", "target": {"new-dataset": False, "benchmark": False, "llm": True}},
    {"text": "this new dataset totally serves as a benchmark", "target": {"new-dataset": True, "benchmark": True, "llm": False}},
])

console.print(smod("Hi there. This is dog. How are you?"))
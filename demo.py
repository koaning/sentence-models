from pprint import pprint 

from sentence_models import SentenceModel

smod = SentenceModel()

smod.learn([
    {"text": "hello there", "target": {"greeting": True}}, 
    {"text": "goodbye there", "target": {"greeting": False}}
])

pprint(smod("Hi there. This is dog. How are you?"))
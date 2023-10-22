from sentence_models import SentenceModel
from embetter.text import SentenceEncoder
from sentence_models.util import console
import srsly 

def test_smoke():
    examples = list(srsly.read_jsonl("datasets/new-dataset.jsonl")) + list(srsly.read_jsonl("datasets/data-quality.jsonl"))
    examples = [{"text": ex["text"], "target": ex["cats"]} for ex in examples]

    smod = SentenceModel(encoder=SentenceEncoder(), verbose=True)

    smod.learn(examples)
    sentence = "This new corpus will be very useful in the research of annotator disagreement. But it won't be about llms."
    out1 = smod(sentence)
    console.log("Demo prediction:")

    smod.to_disk("demo")
    smod_reloaded = SentenceModel.from_disk("demo", encoder=SentenceEncoder())

    out2 = smod_reloaded(sentence)
    assert out1 == out2
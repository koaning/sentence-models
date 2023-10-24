from sentence_models import SentenceModel
from sklearn.feature_extraction.text import HashingVectorizer
from sentence_models.util import console
import srsly


def test_smoke(tmpdir):
    examples = list(srsly.read_jsonl("datasets/new-dataset.jsonl")) + list(srsly.read_jsonl("datasets/data-quality.jsonl"))
    examples = [{"text": ex["text"], "target": ex["cats"]} for ex in examples]

    smod = SentenceModel(encoder=HashingVectorizer(), verbose=True)

    smod.learn(examples)
    sentence = "This new corpus will be very useful in the research of annotator disagreement. But it won't be about llms."
    out1 = smod(sentence)
    console.log("Demo prediction:")

    smod.to_disk(tmpdir)
    smod_reloaded = SentenceModel.from_disk(tmpdir, encoder=HashingVectorizer())

    out2 = smod_reloaded(sentence)
    assert out1 == out2

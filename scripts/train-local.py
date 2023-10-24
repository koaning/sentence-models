import srsly 
from pathlib import Path 

from sentence_models import SentenceModel
from sklearn.feature_extraction.text import HashingVectorizer

if __name__ == "__main__":
    examples = []
    for file in Path("datasets").glob("*.jsonl"):
        new_examples = [{"text": ex["text"], "target": ex["cats"]} for ex in srsly.read_jsonl(file)]
        examples.extend(new_examples)
    smod = SentenceModel(verbose=True, encoder=HashingVectorizer())
    smod.learn(examples)
    smod.to_disk("smod-trained")

    smod_reloaded = SentenceModel.from_disk("smod-trained", encoder=HashingVectorizer())
    print(smod_reloaded("this is a test"))

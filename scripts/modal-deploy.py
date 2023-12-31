import srsly 
from pathlib import Path
from pydantic import BaseModel
import modal
from sentence_models import SentenceModel
from sklearn.feature_extraction.text import HashingVectorizer


image =modal.Image.debian_slim(python_version="3.10").run_commands(
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "pip install sentence-models>=0.1.1",
    "python -m spacy download en_core_web_sm",
)
stub = modal.Stub("sentence-model-demo-hash", image=image)



@stub.cls(cpu=1, mounts=[modal.Mount.from_local_dir("/home/vincent/Development/sentence-models/smod-trained/", remote_path="/root/smod-trained/")])
class Model:
    def __enter__(self):
        self.model = SentenceModel.from_disk("/root/smod-trained/", encoder=HashingVectorizer(), verbose=True)
        print(self.model)

    @modal.method()
    def predict(self, x):
        out = self.model(x)
        print(out)
        return out


class Item(BaseModel):
    text: str

@stub.function()
@modal.web_endpoint(method="POST")
def main(item: Item):
    return Model().predict.remote(item.text)


if __name__ == "__main__":
    examples = []
    for file in Path("../datasets").glob("*.jsonl"):
        new_examples = [{"text": ex["text"], "target": ex["cats"]} for ex in srsly.read_jsonl(file)]
        examples.extend(new_examples)
    smod = SentenceModel(verbose=True, encoder=HashingVectorizer())
    smod.learn(examples)
    smod.to_disk("smod-trained")

    smod_reloaded = SentenceModel.from_disk("smod-trained", encoder=HashingVectorizer())
    print(smod_reloaded("this is a test"))

from pydantic import BaseModel
import modal
from sentence_models import SentenceModel, SentenceEncoder
from sklearn.feature_extraction.text import HashingVectorizer

image =modal.Image.debian_slim(python_version="3.10").run_commands(
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "pip install sentence-models",
    "python -m spacy download en_core_web_sm",
)
stub = modal.Stub("sentence-model-demo", image=image)



@stub.cls(cpu=1, mounts=[modal.Mount.from_local_dir("/home/vincent/Development/sentence-models/smod-trained/", remote_path="/root/smod-trained/")])
class Model:
    def __enter__(self):
        self.model = SentenceModel.from_disk("/root/smod-trained/", encoder=HashingVectorizer(), verbose=True)

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

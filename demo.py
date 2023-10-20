import os 

os.environ["KERAS_BACKEND"] = "torch"

from sentence_models import SentenceModel
from embetter.text import SentenceEncoder
from sentence_models.finetune import ContrastiveFinetuner
from sentence_models.util import console
import srsly 

examples = list(srsly.read_jsonl("data/new-dataset.jsonl")) + list(srsly.read_jsonl("data/data-quality.jsonl"))
examples = [{"text": ex["text"], "target": ex["cats"]} for ex in examples]

smod = SentenceModel(finetuner=ContrastiveFinetuner())

smod.learn(examples)

console.print(smod("This new corpus will be very useful in the research of annotator disagreement. But it won't be about llms."))

smod.to_disk("demo")
smod_reloaded = SentenceModel.from_disk("demo", encoder=SentenceEncoder())

console.print(smod("This new corpus will be very useful in the research of annotator disagreement. But it won't be about llms."))
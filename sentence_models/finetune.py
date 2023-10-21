from typing import Optional, List
import srsly 
import numpy as np
from pathlib import Path

from keras_core.losses import MeanSquaredError
from keras_core.models import Model, Sequential, load_model
from keras_core.layers import Dense, Input, Lambda, Dot, Flatten
from keras_core import backend as K
from keras_core import ops
from keras_core.optimizers import Adam
from keras_core.callbacks import LambdaCallback

import random 
import numpy as np
from itertools import groupby, chain
from collections import namedtuple, defaultdict
from .util import console


def generate_pairs_batch(labels: List[str], n_neg:int=2):
    """
    Copied with permission from Peter Baumgartners implementation
    https://github.com/pmbaumgartner/setfit
    """
    # 7x faster than original implementation on small data,
    # 14x faster on 10000 examples
    Pair = namedtuple('Pair', ['e1', 'e2', 'val'])
    pairs = []
    lookup = defaultdict(list)
    single_example = {}
    indices = np.arange(len(labels))
    for label, grouper in groupby(
        ((s, l) for s, l in zip(indices, labels)), key=lambda x: x[1]
    ):
        lookup[label].extend(list(i[0] for i in grouper))
        single_example[label] = len(lookup[label]) == 1
    neg_lookup = {}
    for current_label in lookup:
        negative_options = list(
            chain.from_iterable(
                [indices for label, indices in lookup.items() if label != current_label]
            )
        )
        neg_lookup[current_label] = negative_options

    for current_idx, current_label in zip(indices, labels):
        positive_pair = random.choice(lookup[current_label])
        if not single_example[current_label]:
            # choosing itself as a matched pair seems wrong,
            # but we need to account for the case of 1 positive example
            # so as long as there's not a single positive example,
            # we'll reselect the other item in the pair until it's different
            while positive_pair == current_idx:
                positive_pair = random.choice(lookup[current_label])
        pairs.append(Pair(current_idx, positive_pair, 1))
        for i in range(n_neg):
            negative_pair = random.choice(neg_lookup[current_label])
            pairs.append(Pair(current_idx, negative_pair, 0))

    return pairs


def create_base_model(hidden_dim, n_layers, activation, input_shape):
    model = Sequential()
    model.add(Input(input_shape))
    for layer in range(n_layers):
        model.add(Dense(hidden_dim, activation=activation))
    return model

def cosine_similarity(vectors):
    return Dot(axes=-1, normalize=True)(vectors)


class ContrastiveFinetuner:
    def __init__(self, hidden_dim: int=300, n_layers: int=1, activation: Optional[str]=None, epochs=5, verbose=False):
        self.hidden_dim = hidden_dim 
        self.activation = activation
        self.n_layers = n_layers
        self.epochs = epochs
        self.model_base = None
        self.model_full = None
        self.verbose = verbose

    def construct_models(self, X1, X2):
        if not self.model_full:
            shape1 = (X1.shape[1], )
            shape2 = (X2.shape[1], )
            self.model_base = create_base_model(self.hidden_dim, self.n_layers, self.activation, shape1)
            input1 = Input(shape=shape1)
            input2 = Input(shape=shape2)
            vector1 = self.model_base(input1)
            vector2 = self.model_base(input2)
            cosine_sim = Lambda(cosine_similarity, output_shape=vector1.shape)([vector1, vector2])
            cosine_sim_flat = Flatten()(cosine_sim)
            self.model_full = Model(inputs=[input1, input2], outputs=cosine_sim_flat)
            self.model_full.compile(optimizer=Adam(), loss=MeanSquaredError())
    
    def to_disk(self, folder: Path):
        self.model_full.save(folder / 'model_full.keras')
        self.model_base.save(folder / 'model_base.keras')
        srsly.write_json(folder / "finetuner.json", {n: getattr(self, n) for n in ["hidden_dim", "activation", "n_layers", "epochs", "verbose"]})
    
    @classmethod
    def from_disk(cls, folder: Path):
        settings = srsly.read_json(folder / "finetuner.json")
        tuner = ContrastiveFinetuner(**settings)
        tuner.model_full = load_model(folder / 'model_full.keras')
        tuner.model_base = load_model(folder / 'model_base.keras')
        return tuner
    
    def encode(self, X):
        return self.model_base.predict(X, verbose=0)

    def learn(self, X1, X2, lab):
        callbacks = []
        if self.verbose:
            console.log("Finetuner starts training.")
            callbacks = [LambdaCallback(on_epoch_end=lambda epoch, logs: console.log(f"Iteration completed {epoch=} loss={logs['loss']}"))]
        self.model_full.fit([X1, X2], lab, epochs=self.epochs, verbose=0, callbacks=callbacks)
        if self.verbose:
            console.log("Finetuner training done.")

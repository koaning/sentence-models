import srsly 
import numpy as np
import matplotlib.pylab as plt 
from sklearn.decomposition import PCA 

from embetter.text import SentenceEncoder
from embetter.utils import cached

from keras.losses import MeanSquaredError
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Lambda, Dot, Flatten
from keras import backend as K
from keras.optimizers import Adam


def generate_pairs_batch(labels, n_neg=3):
    """
    Copied with permission from Peter Baumgartners implementation
    https://github.com/pmbaumgartner/setfit
    """
    # 7x faster than original implementation on small data,
    # 14x faster on 10000 examples
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
        pairs.append((current_idx, positive_pair, 1))
        for i in range(n_neg):
            negative_pair = random.choice(neg_lookup[current_label])
            pairs.append((current_idx, negative_pair, 0))

    return pairs


def create_base_model(hidden_dim, n_layers, activation, input_shape):
    model = Sequential()
    for layer in range(n_layers):
        model.add(Dense(hidden_dim, activation=activation, input_shape=input_shape))
    return model

def cosine_similarity(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return Dot(axes=-1, normalize=False)([x, y])

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


class ContrastiveFinetuner:
    def __init__(self, hidden_dim: int=300, n_layers: int=1, activation: Optional[str]=None):
        self.hidden_dim = hidden_dim 
        self.activation = activation
        self.n_layers = n_layers

    def construct_models(self, X1, X2):
        shape1 = (X1.shape[1], )
        shape2 = (X2.shape[1], )
        self.model_base = create_base_model(self.hidden_dim, self.n_layers, self.activation, shape1)
        input1 = Input(shape=shape1)
        input2 = Input(shape=shape2)
        vector1 = model_base(input1)
        vector2 = model_base(input2)
        cosine_sim = Lambda(cosine_similarity)([vector1, vector2])
        cosine_sim = Flatten()(cosine_sim)
        self.model_full = Model(inputs=[input1, input2], outputs=cosine_sim)
        self.model_full.compile(optimizer=Adam(), loss=MeanSquaredError())
    
    def to_disk(self, folder: Path):
        model_full.save(folder / 'model_full.keras')
        model_base.save(folder / 'model_base.keras')
        srsly.write_json(folder / "finetuner.json", {"hidden_dim": self.hidden_dim, "n_layers": self.n_layers, "activation": self.activation})
    
    @classmethod
    def from_disk(cls, folder: Path):
        settings = srsly.read_json(folder / "finetuner.json")
        tuner = ContrastiveFinetuner(**settings)
        tuner.model_full = keras.models.load_model(folder / 'model_full.keras')
        tuner.model_base = keras.models.load_model(folder / 'model_base.keras')
        return tuner
    
    def encode(self, X):
        return self.model_base.predict(X)


dataset = list(srsly.read_jsonl("new-dataset.jsonl"))
labels = [ex['cats']['new-dataset'] for ex in dataset]
texts = [ex['text'] for ex in dataset]
pairs = generate_pairs_batch(labels)
enc = cached("sbert", SentenceEncoder())
X = enc.transform(texts)

X1 = np.array([X[0] for ex in pairs])
X2 = np.array([X[1] for ex in pairs])

# Before
X_pca = PCA(2).fit_transform(X)
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=5)

tuner = ContrastiveFinetuner(n_layers=1)
tuner.construct_model(X1, X2)
model.fit([X1, X2], np.array([ex[2] for ex in pairs], dtype=float), epochs=100, verbose=2)

# After
X_pca = PCA(2).fit_transform(enc.predict(X))

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=5)
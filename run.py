import copy
import pickle

import keras
from keras.utils.vis_utils import plot_model
import numpy as np
from scipy.spatial.distance import cosine

from util import symbol_table


def cosine_similarity(term_1, term_2, embeddings, model, symbols):
    for term in [term_1, term_2]:
        if term not in symbols:
            return None
        if term not in embeddings:
            index_array = np.array([symbols.get_index(term)])
            embeddings[term] = model.predict(index_array).flatten()
    return 1 - cosine(embeddings[term_1], embeddings[term_2])


def most_similar_term(embedding, embeddings):
    max_sim = -np.inf
    most_similar = None
    for term, embed_vector in embeddings.items():
        sim = 1 - cosine(embedding, embed_vector)
        if sim > max_sim:
            max_sim = sim
            most_similar = term
    return most_similar


class Frontend:

    def __init__(self, model_dir):
        self._model = keras.models.load_model(model_dir)

    def pickle_embeddings(self, model_dir, symbols_txt, pickle_file):
        model = keras.models.load_model(model_dir)
        symbols = symbol_table.SymbolTable.read(symbols_txt)

        # Creates a model that simply outputs embeddings. Layers 0, 2, 3 are
        # the target word input, embedding, and reshape layers respectiely.
        new_model = keras.models.Sequential()
        for index in [0, 2, 3]:
            new_model.add(model.layers[index])
        
        # Stores embeddings for all words in vocabulary as arrays of shape
        # (embedding_dim, 1).
        print('Calculating all embeddings ...')
        embeddings = {}
        for word, index in symbols.dictionary.items():
            if not index % 100:
                print(word, index)
            target_vector = np.array([index])
            prediction = new_model.predict(target_vector)
            embeddings[word] = prediction.flatten()

        with open(pickle_file, 'wb') as outp:
            pickle.dump(embeddings, outp)

    def traverse(self, symbols_txt, source_word, target_word):
        symbols = symbol_table.SymbolTable.read(symbols_txt)

        # Creates a model that simply outputs embeddings. Layers 0, 2, 3 are
        # the target word input, embedding, and reshape layers respectiely.
        new_model = keras.models.Sequential()
        for index in [0, 2, 3]:
            new_model.add(self._model.layers[index])
        
        # Stores embeddings for all words in vocabulary as arrays of shape
        # (embedding_dim, 1).
        print('Calculating all embeddings ...')
        embeddings = {}
        for word, index in symbols.dictionary.items():
            if not index % 100:
                print(word, index)
            target_vector = np.array([index])
            prediction = new_model.predict(target_vector)
            embeddings[word] = prediction.flatten()

        source = copy.deepcopy(embeddings[source_word])
        delta = (embeddings[target_word] - source) / 100
        terms = []
        for i in range(100):
            print(i)
            terms.append(most_similar_term(source, embeddings))
            source += delta
        terms.append(target_word)
        print(' '.join(terms))

    def plot(self, model_dir, plot_png):
        model = keras.models.load_model(model_dir)
        model._layers = [
                layer
                for layer in model._layers
                if not isinstance(layer, dict)]
        plot_model(
                model, to_file=plot_png, show_shapes=True,
                show_layer_names=True)


if __name__ == '__main__':
    import fire
    fire.Fire(Frontend)

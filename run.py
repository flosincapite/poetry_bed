import copy
import heapq
import pickle
import random

from scipy.spatial.distance import cosine

from util import symbol_table


def n_most_similar(embedding, embeddings, n=5):
    heap = []

    for term, embed_vector in embeddings.items():
        distance = cosine(embedding, embed_vector)
        if len(heap) < n:
            push = heapq.heappush
        else:
            push = heapq.heappushpop
        push(heap, (distance, term))
    return heap


class Frontend:

    def pickle_embeddings(self, model_dir, symbols_txt, pickle_file):
        import keras
        import numpy as np

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

    def traverse(self, embeddings_pickle, source_word, target_word):
        with open(embeddings_pickle, 'rb') as inp:
            embeddings = pickle.load(inp)

        def _get_terms():
            yield source_word
            source = copy.deepcopy(embeddings[source_word])
            delta = (embeddings[target_word] - source) / 100
            for i in range(25):
                print(i)
                yield random.choice(n_most_similar(source, embeddings, 3))[-1]
                source += delta
            yield target_word

        terms = []
        seen = set()

        for new_term in _get_terms():
            if new_term not in seen:
                terms.append(new_term)
                seen.add(new_term)

        print(' '.join(terms))

    def plot(self, model_dir, plot_png):
        import keras
        from keras.utils.vis_utils import plot_model

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

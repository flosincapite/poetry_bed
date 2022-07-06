import logging
import os
import word2vec


class Trainer:

    _WORD2VEC_PHRASE_FILE = 'phrases.txt'
    _WORD2VEC_CLUSTER_FILE = 'clusters.txt'
    _WORD2VEC_BINARY_FILE = 'binary.bin'

    def word2vec(self, input_file, output_directory):
        phrase_file = os.path.join(output_directory, self._WORD2VEC_PHRASE_FILE)
        logging.info(f'Training phrases for {input_file} to {phrase_file} ...')
        word2vec.word2phrase(input_file, phrase_file, verbose=True)

        cluster_file = os.path.join(
                output_directory, self._WORD2VEC_CLUSTER_FILE)
        logging.info(
                f'Training clusters for {input_file} to {cluster_file} ...')
        word2vec.word2clusters(phrase_file, binary_file, size=100, verbose=True)

        cluster_file = os.path.join(
                output_directory, self._WORD2VEC_BINARY_FILE)
        logging.info(f'Training binary for {input_file} to {binary_file} ...')
        word2vec.word2vec(
                phrase_file, binary_file, size=100, binary=True, verbose=True)


if __name__ == '__main__':
    import fire
    fire.Fire(Trainer)

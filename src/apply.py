import word2vec


class Runner:

    def similar_terms(self, binary_file, phrase):
        model = word2vec.load(binary_file)
        indexes, metrics = model.similar(phrase)
        print(model.generate_response(indexes, metrics).tolist())


if __name__ == '__main__':
    import fire
    fire.Fire(Runner)

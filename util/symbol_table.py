class SymbolTable:

    UNK = 'UNK'

    def __init__(self, vocabulary_size):
        self._word_to_index = {}
        self._index_to_word = []
        self._vocabulary_size = vocabulary_size

    def _add_word(self, word):
        self._word_to_index[word] = len(self._word_to_index)
        self._index_to_word.append(word)

    def populate(self, reader):
        """Processes raw inputs into a dataset."""
        count = [[self.UNK, 0]]
        count.extend(collections.Counter(
                reader.read_all_words()).most_common(self._vocabulary_size - 1))

        for word, _ in count:
            self._add_word(word)

    @property
    def dictionary(self):
        return self._word_to_index

    @property
    def reversed_dictionary(self):
        return self._index_to_word

    def index_for(self, word):
        return self._word_to_index.get(word, self._word_to_index[self.UNK])

    def words_to_indices(self, words):
        return list(map(self.index_for, words))

    def write(self, file_name):
        with open(file_name, 'w') as outp:
            outp.write(f'{self._vocabulary_size}\n')
            outp.write('\n'.join(self._index_to_word))

    @classmethod
    def read(cls, file_name):
        with open(file_name, 'r') as inp:
            vocabulary_size = int(next(inp).strip())
            result = SymbolTable(vocabulary_size)
            added = 0
            for line in inp:
                result._add_word(line.strip())
                added += 1
            assert added == vocabulary_size
        return result



    
        


from nltk.tokenize import sent_tokenize, word_tokenize
import glob
import json
import logging
import os
import re


class TextNorm:
    
    def __init__(self, output_file, input_globs=None, input_glob_file=None, mode='json'):
        if input_globs is None:
            if input_glob_file is None:
                raise Error('Must supply either --input_globs or --input_glob_file')
            else:
                input_globs = []
                with open(input_glob_file, 'r') as inp:
                    for line in inp:
                        line = line.strip()
                        if line:
                            input_globs.append(line.strip())
        if isinstance(input_globs, str):
            input_globs = input_globs.split(',')

        self._input_globs = input_globs
        self._output_file = output_file
        self._mode = mode

    def _line_generator(self):
        logging.info('self._input_globs is %s', self._input_globs)
        for input_glob in self._input_globs:
            for fname in glob.glob(input_glob):
                logging.info('Getting data for file %s.', fname)
                if self._mode == 'text':
                    if not re.search(r'^\d*\.txt$', fname):
                        continue
                    with open(fname, 'r') as inp:
                        for line in inp:
                            line = line.strip()
                            if line:
                                yield line
                elif self._mode == 'json':
                    if not re.search(r'^\d*\.json$', fname):
                        continue
                    with open(fname, 'r') as inp:
                        json_object = json.loads(inp)
                        for post in json_object:
                            for line in post.get('full_text', '').strip().split('\\n'):
                                line = line.strip()
                                if line:
                                    yield line
                else:
                    raise Exception('nope')

    def _sent_generator(self):
        for line in self._line_generator():
            # TODO: Before tokenization, replace acronyms like G&T with a special
            # token for the term of interest.
            for sent in sent_tokenize(line):
                words = []
                for word in word_tokenize(sent):
                    word = word.lower()
                    if re.search(r'[a-z]', word):
                        words.append(word)
                if len(words) > 4:
                    words = ['<s>'] + words + ['<s>']
                    yield ' '.join(words)
                    """
                    if re.search(r'G\&T', line, re.IGNORECASE):
                        logging.info('For line <%s>, yielding <%s>.', line, ' '.join(words)) 
                    """

    def norm(self):
        with open(self._output_file, 'w') as outp:
            for sent in self._sent_generator():
                outp.write(sent + '\n')


if __name__ == '__main__':
    import fire
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire(TextNorm)

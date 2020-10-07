# poetry_bed

Found text poetry generation.

This is an extra paragraph.

Uses word2vec-style embeddings to traverse chains of semantic similarity between words.

## installation and setup

```
git clone https://github.com/flosincapite/poetry_bed.git
cd poetry_bed
pip install -r requirements_slim.txt
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## usage

```
python run.py traverse --embeddings_pickle trained_models/20000_vocab/embeddings.pickle \
    --source_word <word> --target_word <word>
```

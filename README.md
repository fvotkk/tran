# Word-level RNN applications

Both applications use PyTorch and define their recurrent updates directly with
embedding lookups, weight matrices, and `tanh`. Neither uses PyTorch's built-in
RNN layers. Install PyTorch with `python -m pip install torch` if needed.

## Text generation

This application builds an original story corpus from the seed story and varied
generated chapters in `text_generation/corpus.txt`. The default build creates
240 chapters, about 28,000 word and punctuation tokens.

```bash
python text_generation/build_corpus.py --chapter-count 240
python text_generation/train.py --epochs 20
python text_generation/generate.py --seed-text "mira lived in" --new-token-count 40
```

The trained model is saved in `text_generation/generation_model.pt`.

## English to Spanish translation

This application builds its separate English-Spanish parallel corpus in
`translation/parallel_corpus.tsv`. It combines the seed pairs with thousands of
generated sentence pairs covering pronouns, verbs, objects, and time phrases;
the default build produces 5,833 pairs.
Each line contains an English sentence, a tab, and its Spanish translation.

```bash
python translation/build_corpus.py
python translation/train.py --epochs 100
python translation/translate.py "good morning"
```

The trained model is saved in `translation/translation_model.pt`.

The original single-corpus training example remains available as
`python train_rnn.py sample_corpus.txt`.

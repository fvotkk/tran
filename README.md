# Word-level RNN applications

Both applications use PyTorch and define their recurrent updates directly with
embedding lookups, weight matrices, and `tanh`. Neither uses PyTorch's built-in
RNN layers. Install PyTorch with `python -m pip install torch` if needed.

## Text generation

This application trains on an original 20-chapter story in
`text_generation/corpus.txt`, about the river community of Bellweather. It has
about 8,100 word and punctuation tokens.

```bash
python text_generation/train.py --epochs 100
python text_generation/generate.py --seed-text "mira lived in" --new-token-count 40
```

The trained model is saved in `text_generation/generation_model.pt`.

## English to Spanish translation

This application trains on its separate, hand-written English-Spanish parallel
corpus in `translation/parallel_corpus.tsv`. Its 177 sentence pairs follow a
coherent story about the same river community, with additional everyday phrases.
Each line contains an English sentence, a tab, and its Spanish translation.

```bash
python translation/train.py --epochs 100
python translation/translate.py "good morning"
```

The trained model is saved in `translation/translation_model.pt`.

The original single-corpus training example remains available as
`python train_rnn.py sample_corpus.txt`.

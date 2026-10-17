# tran

Train a word-level recurrent neural network on a UTF-8 text file:

```bash
python train_rnn.py sample_corpus.txt --epochs 10 --output-path sample_rnn_model.pt
```

The model tokenizes words and punctuation, looks up a learned embedding for
each token, and predicts the next token from its tanh recurrent hidden state.
The recurrence is defined directly in `ManualRecurrentLanguageModel`; it does
not use PyTorch's built-in RNN layer. Training uses backpropagation through time
and saves the vocabulary and learned parameters in a PyTorch `.pt` file. Install
PyTorch with `python -m pip install torch` if it is not already available.

import argparse
import re
from pathlib import Path

import torch
from torch import nn


class ManualRecurrentLanguageModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension, hidden_dimension):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.input_weights = nn.Parameter(torch.empty(embedding_dimension, hidden_dimension))
        self.recurrent_weights = nn.Parameter(torch.empty(hidden_dimension, hidden_dimension))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dimension))
        self.output_weights = nn.Parameter(torch.empty(hidden_dimension, vocabulary_size))
        self.output_bias = nn.Parameter(torch.zeros(vocabulary_size))

        nn.init.xavier_uniform_(self.input_weights)
        nn.init.orthogonal_(self.recurrent_weights)
        nn.init.xavier_uniform_(self.output_weights)

    def forward(self, input_token_indices):
        batch_size = input_token_indices.shape[0]
        sequence_length = input_token_indices.shape[1]
        hidden_dimension = self.recurrent_weights.shape[0]
        hidden_state = torch.zeros(
            batch_size,
            hidden_dimension,
            device=input_token_indices.device,
        )
        sequence_logits = []

        for sequence_position in range(sequence_length):
            input_embedding = self.token_embedding(input_token_indices[:, sequence_position])
            hidden_state = torch.tanh(
                input_embedding @ self.input_weights
                + hidden_state @ self.recurrent_weights
                + self.hidden_bias
            )
            token_logits = hidden_state @ self.output_weights + self.output_bias
            sequence_logits.append(token_logits)

        return torch.stack(sequence_logits, dim=1)


def tokenize_corpus(corpus_text):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", corpus_text.lower())


def create_training_sequences(token_indices, sequence_length):
    sequence_count = (len(token_indices) - 1) // sequence_length
    if sequence_count < 1:
        raise ValueError("the corpus must contain more tokens than the sequence length")

    usable_token_count = sequence_count * sequence_length + 1
    input_token_sequences = token_indices[:usable_token_count - 1].reshape(
        sequence_count,
        sequence_length,
    )
    target_token_sequences = token_indices[1:usable_token_count].reshape(
        sequence_count,
        sequence_length,
    )
    return input_token_sequences, target_token_sequences


def train_recurrent_language_model(
    model,
    input_token_sequences,
    target_token_sequences,
    epochs,
    batch_size,
    learning_rate,
    maximum_gradient_norm,
    vocabulary_size,
    random_generator,
):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch_number in range(1, epochs + 1):
        shuffled_sequence_indices = torch.randperm(
            input_token_sequences.shape[0],
            generator=random_generator,
        )
        total_epoch_loss = 0.0
        total_epoch_tokens = 0

        for batch_start in range(0, len(shuffled_sequence_indices), batch_size):
            batch_sequence_indices = shuffled_sequence_indices[
                batch_start:batch_start + batch_size
            ]
            input_batch = input_token_sequences[batch_sequence_indices]
            target_batch = target_token_sequences[batch_sequence_indices]

            optimizer.zero_grad(set_to_none=True)
            prediction_logits = model(input_batch)
            batch_loss = loss_function(
                prediction_logits.reshape(-1, vocabulary_size),
                target_batch.reshape(-1),
            )
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), maximum_gradient_norm)
            optimizer.step()

            batch_token_count = target_batch.numel()
            total_epoch_loss += batch_loss.item() * batch_token_count
            total_epoch_tokens += batch_token_count

        average_epoch_loss = total_epoch_loss / total_epoch_tokens
        print(f"Epoch {epoch_number}/{epochs} - loss: {average_epoch_loss:.4f}")


def main():
    argument_parser = argparse.ArgumentParser(
        description="Train a word-level recurrent neural network language model in PyTorch."
    )
    argument_parser.add_argument("corpus_path", type=Path)
    argument_parser.add_argument("--output-path", type=Path, default=Path("rnn_model.pt"))
    argument_parser.add_argument("--epochs", type=int, default=10)
    argument_parser.add_argument("--sequence-length", type=int, default=32)
    argument_parser.add_argument("--batch-size", type=int, default=16)
    argument_parser.add_argument("--embedding-dimension", type=int, default=64)
    argument_parser.add_argument("--hidden-dimension", type=int, default=128)
    argument_parser.add_argument("--learning-rate", type=float, default=0.001)
    argument_parser.add_argument("--maximum-gradient-norm", type=float, default=5.0)
    argument_parser.add_argument("--seed", type=int, default=7)
    arguments = argument_parser.parse_args()

    if arguments.epochs < 1 or arguments.sequence_length < 1 or arguments.batch_size < 1:
        argument_parser.error("epochs, sequence length, and batch size must be positive")
    if arguments.embedding_dimension < 1 or arguments.hidden_dimension < 1:
        argument_parser.error("embedding and hidden dimensions must be positive")
    if arguments.learning_rate <= 0 or arguments.maximum_gradient_norm <= 0:
        argument_parser.error("learning rate and maximum gradient norm must be positive")

    corpus_text = arguments.corpus_path.read_text(encoding="utf-8")
    corpus_tokens = tokenize_corpus(corpus_text)
    if len(corpus_tokens) < 2:
        argument_parser.error("the corpus must contain at least two word or punctuation tokens")

    vocabulary_tokens = sorted(set(corpus_tokens))
    token_to_index = {
        token: vocabulary_index
        for vocabulary_index, token in enumerate(vocabulary_tokens)
    }
    token_indices = torch.tensor(
        [token_to_index[token] for token in corpus_tokens],
        dtype=torch.long,
    )
    input_token_sequences, target_token_sequences = create_training_sequences(
        token_indices,
        arguments.sequence_length,
    )
    vocabulary_size = len(vocabulary_tokens)
    torch.manual_seed(arguments.seed)
    random_generator = torch.Generator().manual_seed(arguments.seed)

    model = ManualRecurrentLanguageModel(
        vocabulary_size,
        arguments.embedding_dimension,
        arguments.hidden_dimension,
    )
    print(
        f"Training on {len(corpus_tokens)} word and punctuation tokens "
        f"({vocabulary_size} unique tokens, {input_token_sequences.shape[0]} sequences)."
    )
    train_recurrent_language_model(
        model,
        input_token_sequences,
        target_token_sequences,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.maximum_gradient_norm,
        vocabulary_size,
        random_generator,
    )

    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vocabulary_tokens": vocabulary_tokens,
            "model_state_dict": model.state_dict(),
            "embedding_dimension": arguments.embedding_dimension,
            "hidden_dimension": arguments.hidden_dimension,
        },
        arguments.output_path,
    )
    print(f"Saved model to {arguments.output_path}")


if __name__ == "__main__":
    main()

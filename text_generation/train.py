import argparse
import re
from pathlib import Path

import torch
from torch import nn

from model import ManualRecurrentLanguageModel


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


def train_language_model(
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
    application_directory = Path(__file__).parent
    argument_parser = argparse.ArgumentParser(
        description="Train the word-level text generation application."
    )
    argument_parser.add_argument(
        "--corpus-path",
        type=Path,
        default=application_directory / "corpus.txt",
    )
    argument_parser.add_argument(
        "--output-path",
        type=Path,
        default=application_directory / "generation_model.pt",
    )
    argument_parser.add_argument("--epochs", type=int, default=100)
    argument_parser.add_argument("--sequence-length", type=int, default=64)
    argument_parser.add_argument("--batch-size", type=int, default=32)
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
    corpus_tokens = tokenize_corpus(corpus_text) + ["<end>"]
    vocabulary_tokens = sorted(set(corpus_tokens) | {"<unknown>"})
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
    train_language_model(
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

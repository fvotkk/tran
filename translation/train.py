import argparse
import re
from pathlib import Path

import torch
from torch import nn

from model import ManualEncoderDecoderTranslator


PADDING_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unknown>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"


def tokenize_sentence(sentence):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", sentence.lower())


def read_parallel_corpus(corpus_path):
    sentence_pairs = []

    for line_number, corpus_line in enumerate(
        corpus_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not corpus_line.strip():
            continue
        sentence_pair = corpus_line.split("\t")
        if len(sentence_pair) != 2:
            raise ValueError(f"line {line_number} must contain one tab between sentences")
        sentence_pairs.append((sentence_pair[0].strip(), sentence_pair[1].strip()))

    if not sentence_pairs:
        raise ValueError("the parallel corpus contains no sentence pairs")

    return sentence_pairs


def pad_token_sequences(token_sequences, padding_token_index):
    maximum_sequence_length = max(len(token_sequence) for token_sequence in token_sequences)
    padded_sequences = torch.full(
        (len(token_sequences), maximum_sequence_length),
        padding_token_index,
        dtype=torch.long,
    )

    for sequence_index, token_sequence in enumerate(token_sequences):
        padded_sequences[sequence_index, :len(token_sequence)] = torch.tensor(
            token_sequence,
            dtype=torch.long,
        )

    return padded_sequences


def train_translator(
    model,
    source_sequences,
    target_input_sequences,
    target_output_sequences,
    epochs,
    batch_size,
    learning_rate,
    maximum_gradient_norm,
    source_padding_token_index,
    target_padding_token_index,
    random_generator,
):
    loss_function = nn.CrossEntropyLoss(ignore_index=target_padding_token_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch_number in range(1, epochs + 1):
        shuffled_pair_indices = torch.randperm(len(source_sequences), generator=random_generator)
        total_epoch_loss = 0.0
        total_epoch_tokens = 0

        for batch_start in range(0, len(shuffled_pair_indices), batch_size):
            batch_pair_indices = shuffled_pair_indices[batch_start:batch_start + batch_size].tolist()
            source_batch = pad_token_sequences(
                [source_sequences[pair_index] for pair_index in batch_pair_indices],
                source_padding_token_index,
            )
            target_input_batch = pad_token_sequences(
                [target_input_sequences[pair_index] for pair_index in batch_pair_indices],
                target_padding_token_index,
            )
            target_output_batch = pad_token_sequences(
                [target_output_sequences[pair_index] for pair_index in batch_pair_indices],
                target_padding_token_index,
            )

            optimizer.zero_grad(set_to_none=True)
            prediction_logits = model(source_batch, target_input_batch)
            batch_loss = loss_function(
                prediction_logits.reshape(-1, prediction_logits.shape[-1]),
                target_output_batch.reshape(-1),
            )
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), maximum_gradient_norm)
            optimizer.step()

            target_token_count = (target_output_batch != target_padding_token_index).sum().item()
            total_epoch_loss += batch_loss.item() * target_token_count
            total_epoch_tokens += target_token_count

        average_epoch_loss = total_epoch_loss / total_epoch_tokens
        print(f"Epoch {epoch_number}/{epochs} - loss: {average_epoch_loss:.4f}")


def main():
    application_directory = Path(__file__).parent
    argument_parser = argparse.ArgumentParser(
        description="Train the English-to-Spanish word-level translation application."
    )
    argument_parser.add_argument(
        "--corpus-path",
        type=Path,
        default=application_directory / "parallel_corpus.tsv",
    )
    argument_parser.add_argument(
        "--output-path",
        type=Path,
        default=application_directory / "translation_model.pt",
    )
    argument_parser.add_argument("--epochs", type=int, default=100)
    argument_parser.add_argument("--batch-size", type=int, default=64)
    argument_parser.add_argument("--embedding-dimension", type=int, default=64)
    argument_parser.add_argument("--hidden-dimension", type=int, default=128)
    argument_parser.add_argument("--learning-rate", type=float, default=0.005)
    argument_parser.add_argument("--maximum-gradient-norm", type=float, default=5.0)
    argument_parser.add_argument("--seed", type=int, default=19)
    arguments = argument_parser.parse_args()

    if arguments.epochs < 1 or arguments.batch_size < 1:
        argument_parser.error("epochs and batch size must be positive")
    if arguments.embedding_dimension < 1 or arguments.hidden_dimension < 1:
        argument_parser.error("embedding and hidden dimensions must be positive")
    if arguments.learning_rate <= 0 or arguments.maximum_gradient_norm <= 0:
        argument_parser.error("learning rate and maximum gradient norm must be positive")

    try:
        sentence_pairs = read_parallel_corpus(arguments.corpus_path)
    except ValueError as error:
        argument_parser.error(str(error))

    source_sentences = [
        tokenize_sentence(source_sentence)
        for source_sentence, target_sentence in sentence_pairs
    ]
    target_sentences = [
        tokenize_sentence(target_sentence)
        for source_sentence, target_sentence in sentence_pairs
    ]
    source_vocabulary_tokens = sorted(
        {PADDING_TOKEN, UNKNOWN_TOKEN, END_TOKEN}
        | {token for sentence_tokens in source_sentences for token in sentence_tokens}
    )
    target_vocabulary_tokens = sorted(
        {PADDING_TOKEN, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN}
        | {token for sentence_tokens in target_sentences for token in sentence_tokens}
    )
    source_token_to_index = {
        token: vocabulary_index
        for vocabulary_index, token in enumerate(source_vocabulary_tokens)
    }
    target_token_to_index = {
        token: vocabulary_index
        for vocabulary_index, token in enumerate(target_vocabulary_tokens)
    }
    source_sequences = [
        [source_token_to_index[token] for token in sentence_tokens]
        + [source_token_to_index[END_TOKEN]]
        for sentence_tokens in source_sentences
    ]
    target_sequences = [
        [target_token_to_index[START_TOKEN]]
        + [target_token_to_index[token] for token in sentence_tokens]
        + [target_token_to_index[END_TOKEN]]
        for sentence_tokens in target_sentences
    ]
    target_input_sequences = [target_sequence[:-1] for target_sequence in target_sequences]
    target_output_sequences = [target_sequence[1:] for target_sequence in target_sequences]
    source_padding_token_index = source_token_to_index[PADDING_TOKEN]
    target_padding_token_index = target_token_to_index[PADDING_TOKEN]
    torch.manual_seed(arguments.seed)
    random_generator = torch.Generator().manual_seed(arguments.seed)
    model = ManualEncoderDecoderTranslator(
        len(source_vocabulary_tokens),
        len(target_vocabulary_tokens),
        arguments.embedding_dimension,
        arguments.hidden_dimension,
        source_padding_token_index,
    )

    print(
        f"Training on {len(sentence_pairs)} English-Spanish sentence pairs "
        f"({len(source_vocabulary_tokens)} source tokens, "
        f"{len(target_vocabulary_tokens)} target tokens)."
    )
    train_translator(
        model,
        source_sequences,
        target_input_sequences,
        target_output_sequences,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.maximum_gradient_norm,
        source_padding_token_index,
        target_padding_token_index,
        random_generator,
    )

    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_vocabulary_tokens": source_vocabulary_tokens,
            "target_vocabulary_tokens": target_vocabulary_tokens,
            "model_state_dict": model.state_dict(),
            "embedding_dimension": arguments.embedding_dimension,
            "hidden_dimension": arguments.hidden_dimension,
            "source_padding_token_index": source_padding_token_index,
            "target_padding_token_index": target_padding_token_index,
        },
        arguments.output_path,
    )
    print(f"Saved model to {arguments.output_path}")


if __name__ == "__main__":
    main()

import argparse
import re
from pathlib import Path

import torch

from model import ManualEncoderDecoderTranslator


def tokenize_sentence(sentence):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", sentence.lower())


def detokenize_sentence(tokens):
    translated_sentence = " ".join(tokens)
    for punctuation in (".", ",", "!", "?", ";", ":"):
        translated_sentence = translated_sentence.replace(f" {punctuation}", punctuation)
    return translated_sentence.replace("¿ ", "¿").replace("¡ ", "¡")


def translate_sentence(model, source_sentence, checkpoint, maximum_output_tokens):
    source_vocabulary_tokens = checkpoint["source_vocabulary_tokens"]
    target_vocabulary_tokens = checkpoint["target_vocabulary_tokens"]
    source_token_to_index = {
        token: vocabulary_index
        for vocabulary_index, token in enumerate(source_vocabulary_tokens)
    }
    target_token_to_index = {
        token: vocabulary_index
        for vocabulary_index, token in enumerate(target_vocabulary_tokens)
    }
    source_token_indices = [
        source_token_to_index.get(token, source_token_to_index["<unknown>"])
        for token in tokenize_sentence(source_sentence)
    ]
    source_token_indices.append(source_token_to_index["<end>"])
    source_tensor = torch.tensor([source_token_indices], dtype=torch.long)
    hidden_state = model.encode(source_tensor)
    target_token_index = target_token_to_index["<start>"]
    translated_tokens = []

    with torch.no_grad():
        for output_position in range(maximum_output_tokens):
            target_tensor = torch.tensor([target_token_index], dtype=torch.long)
            output_logits, hidden_state = model.decode_step(target_tensor, hidden_state)
            target_token_index = output_logits.argmax(dim=1).item()
            target_token = target_vocabulary_tokens[target_token_index]
            if target_token in ("<end>", "<pad>"):
                break
            translated_tokens.append(target_token)

    return detokenize_sentence(translated_tokens)


def main():
    application_directory = Path(__file__).parent
    argument_parser = argparse.ArgumentParser(
        description="Translate a short English sentence into Spanish."
    )
    argument_parser.add_argument("sentence")
    argument_parser.add_argument(
        "--model-path",
        type=Path,
        default=application_directory / "translation_model.pt",
    )
    argument_parser.add_argument("--maximum-output-tokens", type=int, default=32)
    arguments = argument_parser.parse_args()

    if arguments.maximum_output_tokens < 1:
        argument_parser.error("maximum output tokens must be positive")

    checkpoint = torch.load(arguments.model_path, map_location="cpu", weights_only=True)
    model = ManualEncoderDecoderTranslator(
        len(checkpoint["source_vocabulary_tokens"]),
        len(checkpoint["target_vocabulary_tokens"]),
        checkpoint["embedding_dimension"],
        checkpoint["hidden_dimension"],
        checkpoint["source_padding_token_index"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    translated_sentence = translate_sentence(
        model,
        arguments.sentence,
        checkpoint,
        arguments.maximum_output_tokens,
    )
    print(translated_sentence)


if __name__ == "__main__":
    main()

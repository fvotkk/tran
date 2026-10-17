import argparse
import re
from pathlib import Path

import torch

from model import ManualRecurrentLanguageModel


def tokenize_text(text):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower())


def generate_text(model, seed_tokens, vocabulary_tokens, new_token_count, temperature):
    token_to_index = {
        token: vocabulary_index
        for vocabulary_index, token in enumerate(vocabulary_tokens)
    }
    unknown_token_index = token_to_index["<unknown>"]
    end_token_index = token_to_index["<end>"]
    generated_tokens = list(seed_tokens)
    context_token_indices = [
        token_to_index.get(token, unknown_token_index)
        for token in seed_tokens
    ]
    if not context_token_indices:
        context_token_indices = [unknown_token_index]

    model.eval()
    with torch.no_grad():
        for generation_step in range(new_token_count):
            input_token_indices = torch.tensor([context_token_indices], dtype=torch.long)
            prediction_logits = model(input_token_indices)[0, -1] / temperature
            next_token_probabilities = torch.softmax(prediction_logits, dim=0)
            next_token_index = torch.multinomial(next_token_probabilities, 1).item()
            if next_token_index == end_token_index:
                break

            next_token = vocabulary_tokens[next_token_index]
            generated_tokens.append(next_token)
            context_token_indices.append(next_token_index)

    return " ".join(generated_tokens)


def main():
    application_directory = Path(__file__).parent
    argument_parser = argparse.ArgumentParser(
        description="Generate text with the manually defined word-level RNN."
    )
    argument_parser.add_argument(
        "--model-path",
        type=Path,
        default=application_directory / "generation_model.pt",
    )
    argument_parser.add_argument("--seed-text", default="mira lived in")
    argument_parser.add_argument("--new-token-count", type=int, default=40)
    argument_parser.add_argument("--temperature", type=float, default=0.8)
    argument_parser.add_argument("--seed", type=int, default=13)
    arguments = argument_parser.parse_args()

    if arguments.new_token_count < 1 or arguments.temperature <= 0:
        argument_parser.error("token count and temperature must be positive")

    torch.manual_seed(arguments.seed)
    checkpoint = torch.load(arguments.model_path, map_location="cpu", weights_only=True)
    vocabulary_tokens = checkpoint["vocabulary_tokens"]
    model = ManualRecurrentLanguageModel(
        len(vocabulary_tokens),
        checkpoint["embedding_dimension"],
        checkpoint["hidden_dimension"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    generated_text = generate_text(
        model,
        tokenize_text(arguments.seed_text),
        vocabulary_tokens,
        arguments.new_token_count,
        arguments.temperature,
    )
    print(generated_text)


if __name__ == "__main__":
    main()

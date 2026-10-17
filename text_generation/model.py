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
            sequence_logits.append(hidden_state @ self.output_weights + self.output_bias)

        return torch.stack(sequence_logits, dim=1)

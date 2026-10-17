import torch
from torch import nn


class ManualEncoderDecoderTranslator(nn.Module):
    def __init__(
        self,
        source_vocabulary_size,
        target_vocabulary_size,
        embedding_dimension,
        hidden_dimension,
        padding_token_index,
    ):
        super().__init__()
        self.source_embedding = nn.Embedding(source_vocabulary_size, embedding_dimension)
        self.target_embedding = nn.Embedding(target_vocabulary_size, embedding_dimension)
        self.encoder_input_weights = nn.Parameter(torch.empty(embedding_dimension, hidden_dimension))
        self.encoder_recurrent_weights = nn.Parameter(torch.empty(hidden_dimension, hidden_dimension))
        self.encoder_hidden_bias = nn.Parameter(torch.zeros(hidden_dimension))
        self.decoder_input_weights = nn.Parameter(torch.empty(embedding_dimension, hidden_dimension))
        self.decoder_recurrent_weights = nn.Parameter(torch.empty(hidden_dimension, hidden_dimension))
        self.decoder_hidden_bias = nn.Parameter(torch.zeros(hidden_dimension))
        self.output_weights = nn.Parameter(torch.empty(hidden_dimension, target_vocabulary_size))
        self.output_bias = nn.Parameter(torch.zeros(target_vocabulary_size))
        self.padding_token_index = padding_token_index

        nn.init.xavier_uniform_(self.encoder_input_weights)
        nn.init.orthogonal_(self.encoder_recurrent_weights)
        nn.init.xavier_uniform_(self.decoder_input_weights)
        nn.init.orthogonal_(self.decoder_recurrent_weights)
        nn.init.xavier_uniform_(self.output_weights)

    def encode(self, source_token_indices):
        batch_size = source_token_indices.shape[0]
        sequence_length = source_token_indices.shape[1]
        hidden_dimension = self.encoder_recurrent_weights.shape[0]
        hidden_state = torch.zeros(batch_size, hidden_dimension, device=source_token_indices.device)

        for sequence_position in range(sequence_length):
            input_token_indices = source_token_indices[:, sequence_position]
            input_embedding = self.source_embedding(input_token_indices)
            candidate_hidden_state = torch.tanh(
                input_embedding @ self.encoder_input_weights
                + hidden_state @ self.encoder_recurrent_weights
                + self.encoder_hidden_bias
            )
            token_is_not_padding = input_token_indices != self.padding_token_index
            hidden_state = torch.where(
                token_is_not_padding.unsqueeze(1),
                candidate_hidden_state,
                hidden_state,
            )

        return hidden_state

    def decode_step(self, target_token_indices, previous_hidden_state):
        input_embedding = self.target_embedding(target_token_indices)
        hidden_state = torch.tanh(
            input_embedding @ self.decoder_input_weights
            + previous_hidden_state @ self.decoder_recurrent_weights
            + self.decoder_hidden_bias
        )
        output_logits = hidden_state @ self.output_weights + self.output_bias
        return output_logits, hidden_state

    def forward(self, source_token_indices, target_input_indices):
        hidden_state = self.encode(source_token_indices)
        sequence_length = target_input_indices.shape[1]
        sequence_logits = []

        for sequence_position in range(sequence_length):
            target_token_indices = target_input_indices[:, sequence_position]
            output_logits, hidden_state = self.decode_step(target_token_indices, hidden_state)
            sequence_logits.append(output_logits)

        return torch.stack(sequence_logits, dim=1)

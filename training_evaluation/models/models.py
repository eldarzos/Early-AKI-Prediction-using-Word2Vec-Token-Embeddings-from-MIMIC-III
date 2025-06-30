"""Module containing the main model class."""

import torch.nn as nn
import torch.nn.functional as F

from .embedders import Embedder
from .decoders import get_decoder
from .lstms import LSTM


MODELS = ['AKI']


def init_model(model_type='AKI', n_tokens=1000, latent_dim=128, hidden_dim=128,
               p_dropout=0.0, dt=1.0, weighted=True, dynamic=False):
    """
    Return an instance of AKI prediction LSTM model.

    Parameters
    ----------
    model_type : str
        Type of model class to use. Only 'AKI' supported.

    n_tokens : int
        Number of tokens in embedding.

    latent_dim : int
        Dimensionality of latent embedding.

    hidden_dim : int
        Dimensionality of RNN hidden state.

    p_dropout : float, optional
        Dropout probability of embedding.

    dt : float, optional
        Time step between model predictions.

    weighted : bool, optional
        Whether to weight embeddings before aggregation.

    dynamic : bool, optional
        Whether to predict dynamically (throughout sequence) or not.
    """

    print(f'Initializing AKI prediction model...')
    if model_type != 'AKI':
        raise ValueError(f'Only AKI model supported. Got: {model_type}')

    decoder = get_decoder(model_type)
    model = Model(decoder, n_tokens, latent_dim, hidden_dim,
                  p_dropout=p_dropout, dt=dt, weighted=weighted,
                  dynamic=dynamic)
    model.model_type = model_type  # store to help reloading

    return model


class Model(nn.Module):
    def __init__(self, decoder, n_tokens, latent_dim, hidden_dim,
                 p_dropout=0.0, dt=1.0, weighted=True, dynamic=True):
        """Model class.

        Parameters
        ----------
        decoder : flexehr.models.decoders.decoder
            Decoder class.

        n_tokens : int
            Number of tokens in embedding.

        latent_dim : int
            Dimensionality of latent embedding.

        hidden_dim : int
            Dimensionality of RNN hidden state.

        p_dropout : float, optional
            Dropout probability of embedding.

        dt : float, optional
            Time step between model predictions.

        weighted : bool, optional
            Whether to weight embeddings before aggregation.

        dynamic : bool, optional
            Whether to predict dynamically (throughout sequence) or not.
        """
        super(Model, self).__init__()

        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.p_dropout = p_dropout
        self.dt = dt
        self.weighted = weighted
        self.dynamic = dynamic

        self.embedder = Embedder(self.n_tokens, self.latent_dim,
                                 dt=self.dt, weighted=self.weighted)
        self.dropout = nn.Dropout(self.p_dropout)
        self.lstm = LSTM(self.latent_dim, self.hidden_dim)
        self.decoder = decoder(self.hidden_dim)

    def forward(self, input):
        """
        Forward pass of model.

        Parameters
        ----------
        input: torch.Tensor
            Batch of data. Shape (batch_size, 10000, 2)
        """
        emb = self.dropout(F.relu(self.embedder(input)))
        all_hidden, (final_hidden, _) = self.lstm(emb)
        if self.dynamic:
            output = self.decoder(self.dropout(all_hidden))
        else:
            output = self.decoder(self.dropout(final_hidden))

        return output 
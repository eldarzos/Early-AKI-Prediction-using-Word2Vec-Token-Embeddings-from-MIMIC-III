"""
Module containing the decoders.
"""
from torch import nn


DECODER_DICT = {
    'AKI': 'Binary'
}


def get_decoder(model_type):
    """Get the appropriate decoder for a model type."""
    decoder_name = DECODER_DICT.get(model_type)
    if decoder_name is None:
        raise ValueError(f"Unknown or unsupported model_type for decoder: {model_type}")

    import sys
    current_module = sys.modules[__name__]
    try:
        return getattr(current_module, f'Decoder{decoder_name}')
    except AttributeError:
        raise ValueError(f"Decoder class 'Decoder{decoder_name}' not found in decoders.py")


class DecoderBinary(nn.Module):
    def __init__(self, hidden_dim):
        """Hidden state decoder for binary classification tasks. Outputs LOGITS."""
        super(DecoderBinary, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        # Output raw logits, remove sigmoid
        y = self.fc(h).squeeze(-1)
        return y


# Additional decoders removed - only Binary decoder needed for AKI prediction 
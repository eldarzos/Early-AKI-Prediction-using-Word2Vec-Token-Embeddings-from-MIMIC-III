"""Model Input/Output module."""

import json
import numpy as np
import os
import re
import torch

from .helpers import get_device


def save_model(model, directory, metadata=None, filename='model.pt'):
    """
    Save a model and corresponding metadata.

    Parameters
    ----------
    model : nn.Module
        Model.

    directory : str
        Path to save directory.

    metadata : dict, optional
        Metadata to save.

    filename: str, optional
    """
    device = next(model.parameters()).device
    model.cpu()

    if metadata is None:
        metadata = dict(
            n_tokens=model.n_tokens,
            latent_dim=model.latent_dim,
            hidden_dim=model.hidden_dim,
            model_type=getattr(model, 'model_type', 'AKI'),
            dt=model.dt,
            weighted=model.weighted,
            dynamic=getattr(model, 'dynamic', True),
            p_dropout=model.p_dropout
        )

    save_metadata(metadata, directory)

    path_to_model = os.path.join(directory, filename)
    torch.save(model.state_dict(), path_to_model)

    model.to(device)  # restore device


def load_metadata(directory, filename='meta.json'):
    """Load the metadata of a training directory.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata) as metadata_file:
        metadata = json.load(metadata_file)

    return metadata


def save_metadata(metadata, directory, filename='meta.json', **kwargs):
    """Save the metadata of a training directory.

    Parameters
    ----------
    metadata:
        Object to save

    directory: string
        Path to folder where to save model. For example './experiments/mnist'.

    kwargs:
        Additional arguments to `json.dump`
    """
    os.makedirs(directory, exist_ok=True)
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


def load_model(directory, is_gpu=True, filename='model.pt'):
    """Load a trained model.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    from models.models import init_model
    
    device = get_device(is_gpu=is_gpu)
    metadata = load_metadata(directory)

    model = init_model(
        metadata.get('model_type', 'Mortality'), 
        metadata['n_tokens'],
        metadata['latent_dim'], 
        metadata['hidden_dim'],
        p_dropout=metadata.get('p_dropout', 0.0),
        dt=metadata.get('dt', 1.0), 
        weighted=metadata.get('weighted', True),
        dynamic=metadata.get('dynamic', True)
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(directory, filename), map_location=device), strict=False)
    model.eval()

    return model


def numpy_serialize(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def save_np_arrays(arrays, directory, filename):
    """Save dictionary of arrays in json file."""
    save_metadata(arrays, directory,
                  filename=filename, default=numpy_serialize)


def load_np_arrays(directory, filename):
    """Load dictionary of arrays from json file."""
    arrays = load_metadata(directory, filename=filename)

    return {k: np.array(v) for k, v in arrays.items()} 
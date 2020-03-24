import os
import itertools

import numpy as np

NUMPY_EXTENSION = '.npz'


def load_ngrams_probs(file_path, filter_nan=True):
    """Load the N-grams probabilities to standard numpy format

    :param str file_path:
    """
    assert os.path.isfile(file_path)
    data = np.load(file_path)
    body, words, types = data.f.body, data.f.words, data.f.types

    ngrams_probs = {tp: dict(zip(words, body[:, i])) for i, tp in enumerate(types)}
    if filter_nan:
        for tp in types:
            ngrams_probs[tp] = {wd: val for wd, val in ngrams_probs[tp].items()
                                if np.isfinite(val)}

    return ngrams_probs


def export_ngrams_probs(file_path, ngrams_probs):
    """Export the N-grams probabilities to standard numpy format

    :param str file_path:
    :param dict ngrams_probs:
    :return:

    >>> ngrams_probs = {
    ...     "suffix": {"my": 0.2, "hello": 0.5, "world": 0.9},
    ...     "prefix": {"my": 0.7, "hello": 0.45, "world": 0.8},
    ...     "infix": {"my": 1.0, "nice": 0.65, "world": 0.35}
    ... }
    >>> fpath = './my_ngrams.npz'
    >>> _= export_ngrams_probs(fpath, ngrams_probs)
    >>> os.path.isfile(fpath)
    True
    >>> ngrams_probs2 = load_ngrams_probs(fpath)
    >>> import pprint
    >>> pprint.pprint(ngrams_probs2)  # doctest: +NORMALIZE_WHITESPACE
    {'infix': {'my': 1.0, 'nice': 0.65, 'world': 0.35},
     'prefix': {'hello': 0.45, 'my': 0.7, 'world': 0.8},
     'suffix': {'hello': 0.5, 'my': 0.2, 'world': 0.9}}
    >>> os.remove(fpath)
    """
    types = sorted(list(ngrams_probs.keys()))
    words = sorted(set(itertools.chain(*[ngrams_probs[k].keys() for k in types])))
    body = np.empty((len(words), len(types)), dtype=np.float32)

    for i, tp in enumerate(types):
        for j, wd in enumerate(words):
            body[j, i] = ngrams_probs[tp].get(wd, np.NaN)

    file_path = os.path.splitext(file_path)[0] + NUMPY_EXTENSION
    np.savez_compressed(file_path, body=body, types=types, words=words)
    return file_path

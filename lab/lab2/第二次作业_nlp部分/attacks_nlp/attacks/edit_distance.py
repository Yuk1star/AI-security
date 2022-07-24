# coding:utf-8

import numpy as np


def levenshtein_distance(src, tgt):
    """

    Parameters
    ----------
    src : str
    tgt : str

    Returns
    -------
    float :

    """
    if not src or not tgt:
        return None
    if src == tgt:
        return 0
    if len(src) == 0:
        return len(tgt)
    if len(tgt) == 0:
        return len(src)

    src_len = len(src)
    tgt_len = len(tgt)

    matrix = np.zeros([src_len + 1, tgt_len + 1], dtype=np.float32)
    for i in range(src_len + 1):
        matrix[i, 0] = i
    for i in range(tgt_len + 1):
        matrix[0, i] = i

    for i in range(src_len):
        for j in range(tgt_len):
            matrix[i + 1, j + 1] = min(
                # delete
                matrix[i, j + 1] + 1,
                # insert
                matrix[i + 1, j] + 1,
                # replace
                matrix[i, j] + float(src[i] != tgt[j])
            )
    return matrix[src_len, tgt_len].item()


def levenshtein_sim(a, b):
    """

    Parameters
    ----------
    a : str
    b : str

    Returns
    -------
    float :

    """
    if a is None or b is None:
        return None
    if a == b:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / max(len(a), len(b))

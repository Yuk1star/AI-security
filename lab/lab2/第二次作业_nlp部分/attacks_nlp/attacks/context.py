# coding:utf-8

import numpy as np

from attacks import constants


class Context(dict):
    def __init__(self, history=None, max_tries=5, rg=None):
        super(Context, self).__init__()
        self.history = history or []
        self.max_tries = max_tries
        self.rg = rg or np.random.default_rng(constants.SEED)

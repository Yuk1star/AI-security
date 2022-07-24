# coding:utf-8

from __future__ import unicode_literals

import string
from collections import defaultdict
import random


class RegistryHolder(type):

    REGISTRY = {}

    def __new__(mcs, name, bases, attrs):
        new_cls = type.__new__(mcs, name, bases, attrs)
        if name != 'DedupRule':
            mcs.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(mcs):
        return dict(mcs.REGISTRY)


class DedupRule(object, metaclass=RegistryHolder):
    def __init__(self):
        self.rule_name = self.__class__.__name__

    def single_transform(self, src, ctx):
        raise NotImplementedError

    def transform(self, src, ctx):
        history = set(ctx.history)
        for i in range(ctx.max_tries):
            result = self.single_transform(src, ctx)
            if result and result not in history:
                return result
        return 'empty'


def swap(src, left_index, right_index):
    if src is not None and left_index is not None and right_index is not None\
            and 0 <= left_index < right_index < len(src):
        return src[:left_index] + src[right_index] + src[left_index + 1:right_index] + src[left_index] + src[right_index + 1:]
    return None


class AdjacentTranspose(DedupRule):
    def single_transform(self, src, ctx):
        if not src:
            return None
        pivot_index = ctx.rg.integers(0, len(src))
        return self._adjacent_transpose(src, pivot_index)

    def _adjacent_transpose(self, src, pivot_index):
        right_index = self._get_neighbor(src, pivot_index)
        mid = src[pivot_index]
        if right_index:
            right = src[right_index]
            if right and mid != right:
                return swap(src, pivot_index, right_index)
        else:
            left_index = self._get_neighbor(src, pivot_index, right=False)
            if left_index:
                left = src[left_index]
                if left and mid != left:
                    return swap(src, left_index, pivot_index)
        return None

    def _get_neighbor(self, src, pivot_index, right=True):
        len_ = len(src)
        if right and pivot_index + 1 < len_:
            return pivot_index + 1
        if not right and pivot_index - 1 >= 0:
            return pivot_index - 1
        return None


class RandomInsert(DedupRule):
    
    def single_transform(self, src, ctx):
        if not src:
            return None
        src_len = len(src)
        index = ctx.rg.integers(0, src_len)
        # table_len = len(string.ascii_lowercase)
        # return src[:index] + string.ascii_lowercase[ctx.rg.integers(0, table_len)] + src[index:]
        return src[:index] + self.unicode() + src[index:]

    def unicode(self):
        val = random.randint(0x4e00, 0x9fbf)
        return chr(val)

class RandomDelete(DedupRule):

    def single_transform(self, src, ctx):
        if not src:
            return None
        src_len = len(src)
        index = ctx.rg.integers(0, src_len)
        return src[:index] + src[index + 1:]

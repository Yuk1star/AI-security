# coding:utf-8

import itertools
from collections import defaultdict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from dataclasses import dataclass

from attacks import constants


@dataclass
class MatchResult:
    text: str
    index_pair_list: Optional[Sequence[Tuple[int, int]]]

    def __hash__(self):
        return hash((self.text, self.index_pair_list))


def to_match_result(a, comb):
    res = []
    for i, _ in comb:
        res.append(a[i])
    return MatchResult(constants.EMPTY.join(res), comb)


def get_all_lcs_recur(a, b, lens, **kwargs):
    """

    Parameters
    ----------
    a : str
        a
    b : str
        b
    lens : np.array of np.int8
        动态规划所用的矩阵, shape = [len(a), len(b)]

    Returns
    -------
    list of MatchResult

    """

    def _get_all_lcs(a, b, lens, i, j, lcs, index_pair_list, lcs_list, lcs_set):
        if i == 0 or j == 0:
            if len(index_pair_list) > 0:
                matched = MatchResult(lcs, tuple(index_pair_list))
                if matched not in lcs_set:
                    lcs_list.append(matched)
                    lcs_set.add(matched)
            return
        # ↑
        if lens[i][j] == lens[i-1][j]:
            _get_all_lcs(a, b, lens, i - 1, j, lcs, index_pair_list, lcs_list, lcs_set)
        # ←
        if lens[i][j] == lens[i][j-1]:
            _get_all_lcs(a, b, lens, i, j - 1, lcs, index_pair_list, lcs_list, lcs_set)
        # ↖
        if a[i-1] == b[j-1]:
            _get_all_lcs(a, b, lens, i - 1, j - 1, a[i - 1] + lcs, [(i - 1, j - 1)] + index_pair_list, lcs_list, lcs_set)

    if lens[-1][-1] == 0:
        return []
    lcs_list = []
    index_pair_list = []
    i = len(a)
    j = len(b)
    _get_all_lcs(a, b, lens, i, j, '', index_pair_list, lcs_list, set())
    return lcs_list


def get_all_lcs_pos_iter_then_filter(a, lcs_len, index_pair_list, **kwargs):
    def check_comb(comb):
        row_idx = set()
        col_idx = set()
        for i, j in comb:
            row_idx.add(i)
            col_idx.add(j)
        if len(row_idx) == len(col_idx) == lcs_len > 0:
            for i in range(len(comb) - 1):
                prev_pair = comb[i]
                next_pair = comb[i+1]
                if not (prev_pair[0] < next_pair[0] and prev_pair[1] < next_pair[1]):
                    return False
            return True
        return False

    lcs_list = []
    for comb in itertools.combinations(index_pair_list, lcs_len):
        if check_comb(comb):
            lcs_list.append(to_match_result(a, comb))
    return lcs_list


def build_cand(index_pair_list):
    """

    Parameters
    ----------
    index_pair_list

    Returns
    -------
    defaultdict[int, Set]

    """

    cand_row = defaultdict(set)
    cand_col = defaultdict(set)

    row_set = set()
    col_set = set()

    for r, c in index_pair_list:
        row_set.add(r)
        col_set.add(c)

    for pair in index_pair_list:
        for i in range(pair[0]):
            if i in row_set:
                cand_row[i].add(pair)
        for i in range(pair[1]):
            if i in col_set:
                cand_col[i].add(pair)
    return cand_row, cand_col


def update_cand(indices, k, cand_row, cand_col, cands_list):
    """

    Parameters
    ----------
    indices : List[int]
        下标列表
    k : int
        为第几个下标生成候选
    cand_row : List[Set]
        各行对应的候选集合
    cand_col : List[Set]
        列对应的候选集合
    cands_list : List[List]
        第几位对应的候选集合

    Returns
    -------

    """
    if k == 0:
        return cands_list[0]
    idx = indices[0]
    r, c = cands_list[0][idx]
    cands = cand_row[r].intersection(cand_col[c])
    for i in range(1, k):
        idx = indices[i]
        r, c = cands_list[i][idx]
        cands.intersection_update(cand_row[r])
        cands.intersection_update(cand_col[c])
    return sorted(cands)


def idx_to_item(indices, items_list):
    return tuple(items[i] for i, items in zip(indices, items_list))


def initialize_loop(cands_list, cand_row, cand_col):
    # 循环层数
    loop_cnt = len(cands_list)
    indices = [-1] * loop_cnt

    ends = []
    for cands in cands_list:
        ends.append(len(cands))

    idx = 0
    while True:
        cands = update_cand(indices, idx, cand_row, cand_col, cands_list)
        # 回退
        if len(cands) == 0:
            idx -= 1
            if idx < 0:
                return None
            indices[idx] += 1
            while indices[idx] == ends[idx]:
                idx -= 1
                if idx < 0:
                    return None
                indices[idx] += 1

        else:
            cands_list[idx] = cands
            ends[idx] = len(cands)
            indices[idx] = 0
        idx += 1
        if idx == loop_cnt:
            return indices


def iter_nest_loop(cands_list: List[List], cand_row, cand_col):
    # 循环层数
    loop_cnt = len(cands_list)
    # indices[i] 第 i 层循环下标
    indices = initialize_loop(cands_list, cand_row, cand_col)

    ends = []
    for cands in cands_list:
        ends.append(len(cands))

    yield idx_to_item(indices, cands_list)

    cur_loop = loop_cnt - 1
    while True:
        indices[-1] += 1
        idx = loop_cnt - 1
        # 进位操作
        while idx >= cur_loop and indices[idx] == ends[idx]:
            if idx == cur_loop:
                if cur_loop == 0:
                    return
                cur_loop -= 1
            indices[idx] = 0
            idx -= 1
            # 确定一个合适的 indices[idx]
            # 是否继续寻找
            cont = True
            while indices[idx] < ends[idx] and cont:
                indices[idx] += 1
                if indices[idx] == ends[idx]:
                    break
                for i in range(idx + 1, loop_cnt):
                    cand = update_cand(indices, i, cand_row, cand_col, cands_list)
                    if len(cand) > 0:
                        cands_list[i] = cand
                        ends[i] = len(cand)
                        cont = False
                    else:
                        cont = True
        yield idx_to_item(indices, cands_list)


def get_all_lcs_pos_iter_and_filter(a, lcs_len, index_pair_list, **kwargs):
    if lcs_len == 0:
        return []
    cand_row, cand_col = build_cand(index_pair_list)
    cands_list = [[]] * lcs_len
    cands_list[0] = index_pair_list
    lcs_list = []
    for comb in iter_nest_loop(cands_list, cand_row, cand_col):
        lcs_list.append(to_match_result(a, comb))
    return lcs_list


class LCSAlgo(object):
    def __init__(self, name, get_all_lcs_fn):
        self.name = name
        self.get_all_lcs_fn = get_all_lcs_fn

    def get_all_lcs(self, a, b):
        if not (a and b):
            return []
        if a == b:
            return [MatchResult(a, tuple(zip(range(len(a)), range(len(a)))))]
        lens = np.zeros([len(a) + 1, len(b) + 1], dtype=np.int8)
        index_pair_list = []
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    lens[i+1][j+1] = lens[i][j] + 1
                    index_pair_list.append((i, j))
                else:
                    lens[i+1][j+1] = max(lens[i+1][j], lens[i][j+1])
        lcs_len = lens[-1][-1]
        return self.get_all_lcs_fn(a=a, b=b, lens=lens, lcs_len=lcs_len, index_pair_list=index_pair_list)

    def __hash__(self):
        return hash(self.name)


def lcs_sim(a, b, alpha=0.5, lcs_algo=None):
    """基于 Longest Common Subsequence 的相似度

    Parameters
    ----------
    a : str
        a
    b : str
        b
    alpha : float
        相似度由 2 部分组成: 1) lcs 和 `b` 即关键词的相似度; 2) lcs 和 `a` 中实际片段的重叠程度. `alpha` 控制 1) 的比重

    Returns
    -------
    list of (float, MatchResult) :
        每个元素是一个 (相似度, LCS 匹配详情) 的 tuple

    """
    if a is None or b is None:
        return None
    if lcs_algo is None:
        lcs_algo = LCSAlgo('pos_iter_and_filter', get_all_lcs_pos_iter_and_filter)
    lcs_list = lcs_algo.get_all_lcs(a, b)
    result = []
    for lcs in lcs_list:
        if len(lcs.index_pair_list) > 1:
            len_lcs_span_in_a = lcs.index_pair_list[-1][0] - lcs.index_pair_list[0][0] + 1
        else:
            len_lcs_span_in_a = 1
        len_lcs = len(lcs.text)
        sim = alpha * len_lcs / len(b) + (1 - alpha) * len_lcs / len_lcs_span_in_a
        result.append((sim, lcs))
    return result

# coding:utf-8

import numpy as np


def merge_ranges(a, b):
    """不同分段方式融合, 如果冲突了以 b 为准

    Parameters
    ----------
    a : list of (int, int)
    b : list of (int, int)

    Returns
    -------
    list of (int, int)

    """
    if not b:
        return a
    if not a:
        return b
    a_index = 0
    a_len = len(a)
    b_index = 0
    b_len = len(b)

    res = []
    while a_index < a_len and b_index < b_len:
        a_start, a_end = a[a_index]
        b_start, b_end = b[b_index]

        if len(res) > 0:
            end_added = res[-1][1]
        else:
            end_added = 0

        s = max(a_start, end_added)
        if s < a_end <= b_start:
            res.append((s, a_end))
            a_index += 1
        elif a_start <= b_start <= a_end and a_start <= b_end:
            if s < b_start:
                res.append((s, b_start))
            res.append((b_start, b_end))
            b_index += 1
            while a_index < a_len and a[a_index][1] <= b_end:
                a_index += 1
        elif b_start <= a_start and b_end <= a_start and b_start <= a_end and b_end <= a_end:
            res.append((b_start, b_end))
            b_index += 1
        elif b_start <= a_start <= b_end <= a_end and b_start <= a_end:
            res.append((b_start, b_end))
            b_index += 1
        elif b_start <= a_start <= b_end and b_start <= a_end <= b_end:
            while a_index < a_len and a[a_index][1] <= b_end:
                a_index += 1
    while a_index < a_len:
        a_start, a_end = a[a_index]
        if len(res) > 0:
            end_added = res[-1][1]
        else:
            end_added = 0
        if end_added <= a_start:
            res.append((a_start, a_end))
        elif a_start <= end_added < a_end:
            res.append((end_added, a_end))
        a_index += 1
    while b_index < b_len:
        res.append(b[b_index])
        b_index += 1
    return res


def index_to_range(indices):
    """

    Parameters
    ----------
    indices : list of int

    Returns
    -------
    list of (int, int) :
        每个二元组代表一个左闭右开的区间

    """
    res = []
    if indices:
        start = None
        prev = None
        for i in indices:
            if start is None:
                start = i
            elif prev is not None and prev + 1 != i:
                res.append((start, prev + 1))
                start = i
            prev = i
        res.append((start, prev + 1))
    return res


def v_index_to_range(indices):
    """

    Parameters
    ----------
    indices : np.array of list of int

    Returns
    -------
    np.array of list of (int, int) :

    """
    if indices is None:
        return None
    res = np.empty_like(indices, dtype=object)
    for i, ind in enumerate(indices):
        res[i] = index_to_range(ind)
    return res


def filter_index(src, indices_to_rm):
    """

    Parameters
    ----------
    src : list of int
    indices_to_rm : list of int

    Returns
    -------
    list of int :

    """
    to_rm = set(indices_to_rm)
    return [s for s in src if s not in to_rm]


def v_filter_index(src, indices_to_rm):
    """

    Parameters
    ----------
    src : np.array of list of int
    indices_to_rm : np.array of list of int

    Returns
    -------
    np.array of list of int

    """
    res = np.empty_like(src, dtype=object)
    for i, s_ in enumerate(src):
        res[i] = filter_index(s_, indices_to_rm[i])
    return res


def range_to_seg(src, ranges):
    """

    Parameters
    ----------
    src : str
    ranges : list of (int, int)

    Returns
    -------
    list of str :

    """
    segs = []
    if not (src is not None and ranges):
        return segs
    for s, e in ranges:
        segs.append(src[s:e])
    return segs


def get_src_to_seg_indices(len_, ranges):
    src_seg_indices = []
    if not ranges:
        return src_seg_indices
    for _ in range(len_):
        src_seg_indices.append([])

    for i, (s, e) in enumerate(ranges):
        for j in range(s, e):
            src_seg_indices[j].append(i)
    return src_seg_indices


def get_seg_indices(src, pat):
    """

    Parameters
    ----------
    src : list of (int, int)
    pat : list of (int, int)

    Returns
    -------
    list of int

    """
    seg_indices = []
    if not (src and pat):
        return seg_indices
    for _ in pat:
        seg_indices.append(None)
    pat_to_i = {p: i for i, p in enumerate(pat)}
    for i, s in enumerate(src):
        if s in pat_to_i:
            seg_indices[pat_to_i[s]] = i
    return seg_indices


def text_to_seg_range_by_whitespace(text):
    """

    Parameters
    ----------
    text : str

    Returns
    -------
    list of (int, int)

    """
    seg_ranges = []
    if not text:
        return seg_ranges
    range_start = 0
    for i, c in enumerate(text):
        if i > 0 and text[i - 1] != c and (text[i - 1] == ' ' or c == ' '):
            seg_ranges.append((range_start, i))
            range_start = i
    seg_ranges.append((range_start, len(text)))
    return seg_ranges


def text_to_segs_by_whitespace(text, preset_seg_ranges=None):
    """

    Parameters
    ----------
    text : str
    preset_seg_ranges : list of (int, int)

    Returns
    -------
    list of str :
        segs
    list of list of int :
        text_seg_indices
    list of int :
        preset_seg_indices. 预设的片段所处 `segs` 的下标列表

    """
    seg_ranges = text_to_seg_range_by_whitespace(text)
    seg_ranges = merge_ranges(seg_ranges, preset_seg_ranges)
    segs = range_to_seg(text, seg_ranges)
    text_seg_indices = get_src_to_seg_indices(len(text) if text else 0, seg_ranges)
    preset_seg_indices = get_seg_indices(seg_ranges, preset_seg_ranges)
    return segs, text_seg_indices, preset_seg_indices


def v_text_to_segs_by_whitespace(text, preset_seg_ranges=None):
    """

    Parameters
    ----------
    text : np.array of str
        文本. shape = [batch_size, ]
    preset_seg_ranges : np.array of list of (int, int)
        预设的片段范围. 二元组代表一个片段范围, 左闭右开

    Returns
    -------
    np.array of list of str :
        segs. 文本片段表示, shape 同 `text`
    np.array of list of list of int :
        text_seg_indices. `text_seg_indices[i][j]` 代表中第 i 个 `text` 中第 j 个元素所对应在 segs 里的下标列表
    np.array of list of int :
        preset_seg_indices

    """
    segs = np.empty_like(text, dtype=object)
    text_seg_indices = np.empty_like(text, dtype=object)
    preset_seg_indices = np.empty_like(text, dtype=object)
    for i, t in enumerate(text):
        p = preset_seg_ranges[i] if preset_seg_ranges is not None else None
        segs_, seg_indices, preset_seg_indices_ = text_to_segs_by_whitespace(t, preset_seg_ranges=p)
        segs[i] = segs_
        text_seg_indices[i] = seg_indices
        preset_seg_indices[i] = preset_seg_indices_
    return segs, text_seg_indices, preset_seg_indices


def v_text_to_segs_by_char(text, preset_seg_ranges=None):
    """

    Parameters
    ----------
    text : np.array of str
        文本. shape = [batch_size, ]
    preset_seg_ranges : np.array of list of (int, int)
        预设的片段范围. 二元组代表一个片段范围, 左闭右开

    Returns
    -------
    np.array of list of str :
        segs. 文本片段表示, shape 同 `text`
    np.array of list of list of int :
        text_seg_indices. `text_seg_indices[i][j]` 代表中第 i 个 `text` 中第 j 个元素所对应在 segs 里的下标列表
    np.array of list of int :
        preset_seg_indices

    """
    segs = np.empty_like(text, dtype=object)
    text_seg_indices = np.empty_like(text, dtype=object)
    preset_seg_indices = np.empty_like(text, dtype=object)
    for i, t in enumerate(text):
        p = preset_seg_ranges[i] if preset_seg_ranges is not None else None
        segs_, seg_indices, preset_seg_indices_ = text_to_segs_by_char(t, preset_seg_ranges=p)
        segs[i] = segs_
        text_seg_indices[i] = seg_indices
        preset_seg_indices[i] = preset_seg_indices_
    return segs, text_seg_indices, preset_seg_indices

def segs_to_text(segs):
    """

    Parameters
    ----------
    segs : list of str

    Returns
    -------
    str :

    """
    return ''.join(segs)


def replace(segs, index, tgt):
    """

    Parameters
    ----------
    segs : list of str
    index : int
    tgt : str

    Returns
    -------
    list of str :

    """
    result = list(segs)
    if tgt == 'empty':
        tgt = ''
    result[index] = tgt
    return result



def replace_for_zh(segs, index, tgt):
    """

    Parameters
    ----------
    segs : list of str
    index : int (for chinese index:index+2)
    tgt : str

    Returns
    -------
    list of str :

    """
    result = list(segs)
    if tgt == 'empty':
        tgt = ''
    if index < len(result):
        result[index:index+2] = tgt
    else:
        result[index] = tgt

    return result


def text_to_seg_range_by_char(text):
    """

    Parameters
    ----------
    text : str

    Returns
    -------
    list of (int, int)

    """
    seg_ranges = []
    if not text:
        return seg_ranges
    for i in range(len(text)):
        seg_ranges.append((i, i+1))
    return seg_ranges


def text_to_segs_by_char(text, preset_seg_ranges=None):
    """

    Parameters
    ----------
    text : str
    preset_seg_ranges : list of (int, int)

    Returns
    -------
    list of str :
        segs
    list of list of int :
        text_seg_indices
    list of int :
        preset_seg_indices. 预设的片段所处 `segs` 的下标列表

    """
    seg_ranges = text_to_seg_range_by_char(text)
    seg_ranges_merge = merge_ranges(seg_ranges, preset_seg_ranges)
    segs = range_to_seg(text, seg_ranges_merge)
    text_seg_indices = get_src_to_seg_indices(len(text) if text else 0, seg_ranges_merge)
    preset_seg_indices = get_seg_indices(seg_ranges_merge, preset_seg_ranges)
    return segs, text_seg_indices, preset_seg_indices


def range_to_index(ranges):
    if not ranges:
        return []
    indices = set()
    for s, e in ranges:
        indices = indices.union(range(s, e))
    return sorted(indices)


def parse_ranges(text, pair_sep=',', sep=':'):
    ranges = []
    if text is None:
        return ranges
    for pair in text.split(pair_sep):
        parts = pair.split(sep)
        if len(parts) < 2:
            continue
        ranges.append((int(parts[0]), int(parts[1])))
    return ranges

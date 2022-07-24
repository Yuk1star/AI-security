# coding:utf-8

import numpy as np

import attacks.seg_index_utils
from attacks import common_attack
from attacks import context
from attacks import edit_distance
from attacks import seg_index_utils


def remove_item(seq, i):
    """

    Parameters
    ----------
    seq : list
    i : int

    Returns
    -------
    list :
        去掉 `seq[i]` 后的 `seq`

    """
    return [seq[j] for j in range(len(seq)) if j != i]


def get_sorted_seg_indices(importance):
    """

    Parameters
    ----------
    importance : np.array of list of float

    Returns
    -------
    np.array of list of int :

    """
    result = np.empty_like(importance)
    for i, scores in enumerate(importance):
        scores_ = np.array(scores, dtype=np.float32)
        scores_ = np.nan_to_num(scores_, nan=-np.inf)
        result[i] = np.argsort(scores_)[::-1].tolist()

    return result


class TextBugger(common_attack.Attacker):

    def __init__(self, estimator, transformation_manager):
        """

        Parameters
        ----------
        estimator : Estimator
            待攻击模型
        transformation_manager : TransformationManager
        """
        super(TextBugger, self).__init__()
        self.estimator = estimator
        self.transformation_manager = transformation_manager

    def attack(self, src, rg=None, skip_ranges=None, y=None, white_box=True, sim_thres=0.8, **kwargs):
        """

        Parameters
        ----------
        y : np.array of int
            标签. shape 同 `src`
        white_box : bool
        sim_thres : float

        """
        # segs, text_seg_indices, skip_seg_indices = attacks.seg_index_utils.v_text_to_segs_by_whitespace(src, preset_seg_ranges=skip_ranges)
        segs, text_seg_indices, skip_seg_indices = attacks.seg_index_utils.v_text_to_segs_by_char(src,
                                                                                                        preset_seg_ranges=skip_ranges)
        if y is None:
            y = np.ones_like(src, dtype=int)
        importance = self._get_importance(src, segs, text_seg_indices, white_box, y)
        sorted_seg_indices = seg_index_utils.v_filter_index(get_sorted_seg_indices(importance), skip_seg_indices)
        return self._attack(src, segs, sorted_seg_indices, y, sim_thres)

    def _get_importance(self, text, segs, text_seg_indices, white_box, y):
        """

        Parameters
        ----------
        text : np.array of str
        segs : np.array of list of str
        text_seg_indices : np.array of list of list of int
        white_box : bool
        y : np.array of int

        Returns
        -------
        np.array of list of float :
            importance. `importance[i][j]` 是 `segs[i][j]` 的重要性

        """
        if white_box:
            return self.estimator.gradient_raw(text, segs, text_seg_indices, y)
        score = self.estimator.predict_proba_raw(text, y)
        importance = np.empty_like(segs)
        for i, segs_ in enumerate(segs):
            importance_i = []
            for j, seg in enumerate(segs_):
                score_remove = self.estimator.predict_proba_raw(
                    np.array([attacks.seg_index_utils.segs_to_text(remove_item(segs_, j))], dtype=object), y)[0]
                importance_i.append(score.tolist()[i][0] - score_remove.tolist()[0])

            importance_i = np.array(importance_i, dtype=np.float32)
            importance[i] = importance_i
        return importance

    def _attack(self, src, segs, sorted_seg_indices, y, sim_thres):
        """

        Parameters
        ----------
        src : np.array of str
        segs : np.array of list of str
        sorted_seg_indices : np.array of list of int
        y : np.array of int
        sim_thres : float

        """
        tgt = np.empty_like(src, dtype=object)
        ext_info = np.empty_like(src, dtype=object)
        for i, s in enumerate(src):
            times, tgt[i], ext_info[i] = self._attack_single(s, segs[i], sorted_seg_indices[i], y[i], sim_thres)
        return times, tgt, ext_info

    def _attack_single(self, src, segs, sorted_seg_indices, y, sim_thres):
        """

        Parameters
        ----------
        src : str
        segs : list of str
        sorted_seg_indices : list of int
        y : int
        sim_thres : float

        Returns
        -------
        str :
            tgt
        AttackExtInfo :
            ext_info

        """
        trans_segs = segs
        trans_types = []
        times = 0
        for i in sorted_seg_indices:
            bug, trans_type = self._select_bug(src, trans_segs, i, y)
            trans_segs = attacks.seg_index_utils.replace(trans_segs, i, bug)
            tgt = attacks.seg_index_utils.segs_to_text(trans_segs)
            times += 1
            if edit_distance.levenshtein_sim(src, tgt) < sim_thres:
                return times, None, None
            else:
                trans_types.append(trans_type)
                pred = self.estimator.predict_raw(np.array([tgt], dtype=object))[0]

                if pred != y:
                    return times, tgt, common_attack.AttackExtInfo('success', trans_types)
        return times, None, None

    def _generate_bugs(self, segs, index):
        """

        Parameters
        ----------
        segs : list of str
        index : int

        Returns
        -------
        list of str :
            bugs
        list of str :
            trans_types

        """
        seg = segs[index]
        ctx = context.Context()
        return self.transformation_manager.transform(seg, tags=None, ctx=ctx)

    def _generate_bugs_for_zh(self, segs, index):
        """

        Parameters
        ----------
        segs : list of str
        index : int

        Returns
        -------
        list of str :
            bugs
        list of str :
            trans_types

        """
        seg = segs[index]
        if index < len(segs):
            seg = ''.join(segs[index : index+2])
        ctx = context.Context()
        return self.transformation_manager.transform(seg, tags=None, ctx=ctx)

    def _select_bug(self, text, segs, index, y):
        """

        Parameters
        ----------
        text : str
        segs : list of str
        index : int
        y : int

        Returns
        -------
        str :
            bug
        str :
            trans_type

        """
        bugs, trans_types = self._generate_bugs_for_zh(segs, index)
        cand = np.empty_like(bugs, dtype=object)
        for i, bug in enumerate(bugs):
            cand[i] = attacks.seg_index_utils.segs_to_text(attacks.seg_index_utils.replace_for_zh(segs, index, bug))
        y_ = np.array([y], dtype=np.int32)
        score_orig = self.estimator.predict_proba_raw(np.array([text], dtype=object), y_)
        score_bug = self.estimator.predict_proba_raw(cand, np.broadcast_to(y, np.shape(cand)))
        score = score_orig - score_bug
        best_i = np.argmax(score)
        return bugs[best_i], trans_types[best_i]

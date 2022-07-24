# coding:utf-8

import abc
import argparse
import logging
import sys
from operator import attrgetter
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from dataclasses import dataclass
from dataclasses import field

from attacks import constants
from attacks import context
from attacks import edit_distance
# from attacks import io
from attacks import match
from attacks import seg_index_utils
from attacks import transformation_rules
from attacks.context import Context

logger = logging.getLogger(__name__)


@dataclass
class Name:
    text: str
    tags: Sequence[str] = field(default_factory=list)


@dataclass
class LocateResult:
    range: Tuple[int, int]
    name: Optional[Name]


class AttackSourceLocater(object):
    """待变种片段定位"""
    def __init__(self, namelist):
        """

        Parameters
        ----------
        namelist : list of Name
            匹配名单
        """
        self.namelist = namelist

    def locate(self, text):
        """

        Parameters
        ----------
        text : str
            text

        Returns
        -------
        LocateResult :
            定位结果, 可能为 None


        """
        a = text
        best_sim = 0
        res = None
        for name in self.namelist:
            b = name.text
            for sim, lcs in match.lcs_sim(a, b):
                if sim > best_sim:
                    best_sim = sim
                    res = LocateResult(
                        range=(lcs.index_pair_list[0][0], lcs.index_pair_list[-1][0]),
                        name=name)
        return res


class AttackAllSourceLocater(object):
    """待变种片段定位"""

    def locate(self, text):
        """

        Parameters
        ----------
        text : str
            text

        Returns
        -------
        LocateResult :
            定位结果, 可能为 None


        """
        if not text:
            return None
        return LocateResult(range=(0, len(text)), name=None)


@dataclass
class AttackExtInfo:
    locateResult: Optional[LocateResult]
    trans_types: Sequence[str]


class Attacker(abc.ABC):
    @abc.abstractmethod
    def attack(self, src, rg=None, skip_ranges=None, **kwargs):
        """

        Parameters
        ----------
        src : np.array of str
            待变种样本. shape = [batch_size, ]
        rg : RandomGenerator
            rg
        skip_ranges : np.array of list of (int, int)
            跳过的区间列表. shape 同 `src`. 二元组代表一个左闭右开的区间. 默认为 None, 即 `src` 所有位置都可以攻击
        kwargs :
            其他参数

        Returns
        -------
        np.array of str :
            tgt, 变种样本. shape 同 `src`
        np.array of AttackExtInfo :
            shape 同 `src`
        """
        pass


class RuleBasedAttacker(Attacker):
    def __init__(self, locater, transformation_manager):
        self.locater = locater
        self.transformation_manager = transformation_manager

    def attack(self, src, rg=None, skip_ranges=None, depth=1, try_n=1, sim_thres=0.8, **kwargs):
        """

        Parameters
        ----------
        depth : int
            变换次数
        try_n : int
            完成成功变换最多尝试次数
        sim_thres : float
            相似度阈值. 保证生成的变种和原始相似度不低于 `sim_thres`
        """
        rg = rg or np.random.default_rng(constants.SEED)
        tgt = np.empty_like(src, dtype=object)
        ext_info = np.empty_like(src, dtype=object)
        for i, s in enumerate(src):
            skip_ranges_ = skip_ranges[i] if skip_ranges is not None else None
            tgt[i], ext_info[i] = self._attack_single(s, rg, depth, skip_ranges=skip_ranges_, try_n=try_n,
                                                      sim_thres=sim_thres)
        return tgt, ext_info

    def _attack_single(self, src, rg, depth, skip_ranges=None, try_n=1, sim_thres=0.8):
        if not src:
            return None, None
        if skip_ranges is None:
            skip_ranges = []
        segs, text_seg_indices, preset_seg_indices = seg_index_utils.text_to_segs_by_char(
            src, preset_seg_ranges=skip_ranges)
        skip_seg_indices_set = set(preset_seg_indices)
        src_no_skip = seg_index_utils.segs_to_text(
            [seg for i, seg in enumerate(segs) if i not in skip_seg_indices_set])

        locate_res_no_skip = self.locater.locate(src_no_skip.lower())
        if locate_res_no_skip is None:
            return None, None
        attack_range = locate_res_no_skip.range
        tags = None if locate_res_no_skip.name is None else locate_res_no_skip.name.tags
        # 待攻击区间换算回原始区间
        offset = np.zeros([len(segs)+1], dtype=int)
        if len(preset_seg_indices) > 0:
            offset[preset_seg_indices] = 1
        offset = np.cumsum(offset)[1:]
        attack_range_orig = (
            attack_range[0] + offset[attack_range[0]].item(), attack_range[1] + offset[attack_range[1]-2].item())

        locate_res = LocateResult(attack_range_orig, name=locate_res_no_skip.name)
        preset_seg_ranges = sorted(skip_ranges + [attack_range_orig])
        segs, text_seg_indices, preset_seg_indices = seg_index_utils.text_to_segs_by_char(
            src, preset_seg_ranges=preset_seg_ranges)
        attack_seg_index = preset_seg_indices[preset_seg_ranges.index(attack_range_orig)]

        attack_seg = segs[attack_seg_index]

        skip_seg_indices_set = set(i for i in preset_seg_indices if i != attack_seg_index)

        ctx = Context(history=[attack_seg], rg=rg)
        best_tgt = None
        trans_type_list = []
        for i in range(depth):
            for j in range(try_n):
                src_ = ctx.history[-1]
                tgts, trans_types = self.transformation_manager.transform(src_, tags=tags, ctx=ctx)
                len_tgts = len(tgts)
                if len_tgts > 0:
                    pick_i = rg.integers(0, len_tgts)
                    tgt = tgts[pick_i]
                    trans_type = trans_types[pick_i]
                else:
                    tgt = None
                    trans_type = None

                logger.debug('depth: %d, try_i: %d, trans_type: %s, src: %s, tgt: %s', i, j, trans_type, src_, tgt)
                if tgt is not None:
                    ctx.history.append(tgt)
                    segs_replace = seg_index_utils.replace(segs, attack_seg_index, tgt)
                    tgt_text = seg_index_utils.segs_to_text(segs_replace)
                    tgt_no_skip = seg_index_utils.segs_to_text(
                        [seg for i, seg in enumerate(segs_replace) if i not in skip_seg_indices_set])
                    sim = edit_distance.levenshtein_sim(src_no_skip, tgt_no_skip)
                    if sim >= sim_thres:
                        best_tgt = tgt_text
                        trans_type_list.append(trans_type)
                    break
        return best_tgt, AttackExtInfo(locate_res, trans_type_list)


class TransformationManager(object):
    def __init__(self, rulenames=None, tag_rulenames=None, bijections=None):
        """

        Parameters
        ----------
        rulenames : List[str]
            限定变种算子列表. 为 None 时表示所有
        tag_rulenames : Mapping[str, str]
            tag -> rulenames 映射
        bijections : List[Tuple[str, str]]
        """
        rulenclass_map = transformation_rules.DedupRule.get_registry()
        if rulenames is None:
            rulenames = sorted(rulenclass_map.keys())
        rule_map = {}
        for r in rulenames:
            # if r == transformation_rules.BijectionReplace.__name__ and bijections:
            #     rule = rulenclass_map[r](bijection=bijections)
            # else:
            #     rule = rulenclass_map[r]()
            rule = rulenclass_map[r]()
            rule_map[r] = rule

        self._rules = [rule_map[r] for r in rulenames]

        if tag_rulenames is not None:
            self._tag_rules = {tag: [rule_map[r] for r in _rules] for tag, _rules in tag_rulenames.items()}
        else:
            self._tag_rules = None

    def transform(self, src, tags=None, ctx=None):
        """

        Parameters
        ----------
        src
        tags : sequence of str
        ctx

        Returns
        -------
        list[str] :
            变换后结果
        list[str] :
            变换算子名

        """
        if tags is not None and len(tags) > 0 and self._tag_rules is not None:
            rules = set(self._tag_rules[tags[0]])
            for i in range(1, len(tags)):
                rules.intersection_update(self._tag_rules[tags[i]])
            rules = sorted(rules, key=attrgetter('rule_name'))
        else:
            rules = self._rules

        if ctx is None:
            ctx = context.Context()

        tgts = []
        rule_names = []

        for rule in rules:
            tgt = rule.transform(src, ctx)
            if tgt is not None:
                ctx.history.append(tgt)
                tgts.append(tgt)
                rule_names.append(rule.rule_name)
        return tgts, rule_names


# def build_locater(args):
#     namelist = io.read_jsonline(args.listfile, data_class=Name)
#     return AttackSourceLocater(namelist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', default='-', help='包含 id, text, skip_ranges, depth, sim_thres 列, 制表符分割')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=constants.SEED)
    parser.add_argument('-o', '--outfile', default='-')
    parser.add_argument('-l', '--listfile')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger.debug('args: %s', args)

    rg = np.random.default_rng(args.seed)

    # transformation_manager = TransformationManager(io.read_tuples('data/bijection_v0.txt', constants.TAB))
    # attacker = RuleBasedAttacker(build_locater(args), transformation_manager)
    #
    # infile = sys.stdin if args.infile == '-' else open(args.infile)
    # outfile = sys.stdout if args.outfile == '-' else open(args.outfile, mode='w')
    #
    # with infile, outfile:
    #     for line in infile:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         id_, text, skip_ranges_t, depth_t, sim_thres_t = line.split('\t', 4)
    #         src = np.empty([1], dtype=object)
    #         src[0] = text
    #         skip_ranges = np.empty([1], dtype=object)
    #         skip_ranges[0] = seg_index_utils.parse_ranges(skip_ranges_t)
    #         depth = int(depth_t)
    #         sim_thres = float(sim_thres_t)
    #         tgt, trans_type = attacker.attack(src, rg=rg, skip_ranges=skip_ranges, depth=depth, try_n=5,
    #                                           sim_thres=sim_thres)
    #         tgt = tgt[0] or ''
    #         trans_type = trans_type[0] or ''
    #         outfile.write(f'{id_}\t{tgt}\t{trans_type}\n')


if __name__ == '__main__':
    main()

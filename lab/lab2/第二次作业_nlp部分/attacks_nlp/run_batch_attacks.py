# coding:utf-8

import unittest

import numpy as np
import torch

from attacks import constants
from attacks import text_bugger
# from attacks import tf_utils
from attacks.common_attack import AttackExtInfo
from attacks.common_attack import TransformationManager
from models.text_cnn import TextCNN
from models.text_cnn import Config
from utils import build_batch_predict_from_path, build_single_predict_from_text
from models.predictor import Predictor
from tqdm import tqdm
from attacks import common_attack


class MyTestCase():
    def __init__(self):

        transformation_manager = TransformationManager(['RandomDelete'])
        # transformation_manager = TransformationManager(bijections=bijections)

        PATH = 'datas'
        embedding = 'embedding_pretrain.npz'
        self.config = Config(PATH, embedding)
        self.textcnn = TextCNN(self.config)
        state_dict = torch.load('datas/saved_dict/rawTextCNN.ckpt')
        self.textcnn.load_state_dict(state_dict, strict=False)
        self.textcnn.eval()
        self.predictor = Predictor(self.textcnn,self.config)

        self.attacker = text_bugger.TextBugger(self.predictor, transformation_manager)

    def test_black_box(self):

        file_path = 'datas/data/samples.txt'

        attack_suc = 0
        batch_num = 0
        times_list = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                batch_num += 1
                content, label = lin.split('\t')
                src = np.empty([1], dtype=object)
                src[0] = content
                pred_bfr = self.predictor.predict_raw(src)
                rg = np.random.default_rng(constants.SEED)
                expect = (np.array([content], dtype=object),
                          np.array([AttackExtInfo(locateResult=None, trans_types=['RandomDelete'])], dtype=object))
                times, tgts, actual = self.attacker.attack(src, y=pred_bfr, white_box=False, sim_thres=0.8, rg=rg)
                aa  = np.array(actual)
                if 'success' in str(np.array(actual)):
                    attack_suc += 1
                    times_list.append(times)

        # Calculate final accuracy for this epsilon
        final_acc = attack_suc / float(batch_num)
        print("Black box\ttrans_types: {}\tAttack success rate = {} / {} = {}\tAttack times = {}".format('RandomDelete', attack_suc, batch_num, final_acc, np.mean(times_list)))



    def test_white_box(self):

        file_path = 'datas/data/samples.txt'
        attack_suc = 0
        batch_num = 0
        times_list = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                batch_num += 1
                content, label = lin.split('\t')
                src = np.empty([1], dtype=object)
                src[0] = content
                pred_bfr = self.predictor.predict_raw(src)
                rg = np.random.default_rng(constants.SEED)
                expect = (np.array([content], dtype=object),
                          np.array([AttackExtInfo(locateResult='seccess', trans_types=['RandomDelete'])], dtype=object))
                times, tgts, actual  = self.attacker.attack(src, y=pred_bfr, white_box=True, sim_thres=0.7, rg=rg)

                if 'success' in str(np.array(actual)):
                    attack_suc += 1
                    times_list.append(times)

        # Calculate final accuracy for this epsilon
        final_acc = attack_suc / float(batch_num)
        print("White box\ttrans_types: {}\tAttack success rate = {} / {} = {}\tAttack times = {}".format('RandomDelete', attack_suc, batch_num,
                                                                           final_acc, np.mean(times_list)))


if __name__ == '__main__':
    mytest = MyTestCase()
    mytest.test_black_box()
    mytest.test_white_box()

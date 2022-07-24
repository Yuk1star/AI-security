# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from torch.nn.utils.rnn import pad_sequence
import re
from collections import defaultdict
from models.text_cnn import TextCNN

from models.text_cnn import Config


'''Convolutional Neural Networks for Sentence Classification'''

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def text_to_id(segs, vocab, unk_token=None):
    """

    Parameters
    ----------
    segs : list of str
    vocab : dict of (str, int)
    unk_token : str

    Returns
    -------
    list of int :
    """
    ids = []
    for s in segs:
        if s not in vocab:
            if unk_token is not None:
                ids.append(vocab[unk_token])
        else:
            ids.append(vocab[s])
    return ids

def skip_token(seq, token):
    for i, t in enumerate(seq):
        if t != token:
            return i
    return None

class Predictor(nn.Module):
    def __init__(self, model, config):
        super(Predictor, self).__init__()
        self.model = model
        self.config = config
        self.word_vocab = pkl.load(open(config.vocab_path, 'rb'))
        self.pad_size = config.pad_size

        def _lower(text):
            return text.lower()

        def v_preprocess(text):
            return [t for t in text]

        def _text_to_id(segs):
            return text_to_id(segs, self.word_vocab)

        def v_text_to_id(seq):
            return [_text_to_id(t) for t in seq]

        self._v_lower = np.vectorize(_lower)
        self._v_preprocess = v_preprocess
        self._v_text_to_id = v_text_to_id


    def load_model(self, model_path):

        PATH = 'datas'
        embedding = 'embedding_pretrain.npz'
        self.config = Config(PATH, embedding)
        self.model = TextCNN(self.config)
        # state_dict = torch.load(config.save_path)
        state_dict = torch.load(model_path)
        # print(state_dict)
        # print('state_dict', state_dict.keys())
        # print('model_dict', model.keys())
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def predict(self, x):
        """

        Parameters
        ----------
        x : np.array

        Returns
        -------
        np.array of int :

        """
        outputs = self.model(x)
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        return predic

    def predict_raw(self, raw):
        if len(raw) == 0:
            return None

        x, seg_indices = self.preprocess(raw)
        return self.predict(x)

    def predict_proba_raw(self, raw, y):
        """

        Parameters
        ----------
        raw : np.array of str
        y : np.array of int

        Returns
        -------
        np.array of float :

        """
        x, seg_indices = self.preprocess(raw)
        return self.predict_proba(x, y)
    
    def predict_proba(self, x, y):

        outputs = self.model(x)
        score = outputs.data.cpu().numpy()

        return score[:, y]


    def preprocess(self, text):
        text_bfr = self._v_lower(text)
        text_aft = self._v_preprocess(text_bfr)
        id_ = self._v_text_to_id(text_aft)
        x = []
        for unit in id_:
            s_len = len(unit)  # calculate the length of sequence in each unit
            x1 = np.pad(unit, (0, self.pad_size - s_len), 'constant')
            x.append(x1)

        text_x_indices = np.empty_like(text_bfr, dtype=object)
        for i, text_bfr_i in enumerate(text_bfr):

            x_i = x[i]
            x_i_index = skip_token(x_i, self.word_vocab[PAD])
            x_i_len = len(x_i)
            indices = []
            for j, c in enumerate(text_bfr_i):
                if x_i_index < x_i_len and c in self.word_vocab and self.word_vocab[c] == x_i[x_i_index]:
                    indices.append([x_i_index])
                    x_i_index += 1
                else:
                    indices.append([None])
            text_x_indices[i] = indices
        # print(len(x))
        x = torch.LongTensor(x)
        return x, text_x_indices

    def gradient(self, x, y):

        data, target = torch.LongTensor(x), torch.LongTensor(y)
        output = self.model(data)
        # Calculate the loss
        loss = F.cross_entropy(output, target)
        loss.backward()

        data_grad = self.get_emb_grad(self.model)
        # Zero all existing gradients
        # self.model.zero_grad()

        # Calculate gradients of model in backward pass
        # Collect datagrad
        data_grad = data_grad[x, :]

        return np.sum(data_grad, axis=-1)

    def get_emb_grad(self, model, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # self.backup[name] = param.data.clone()
                return param.grad.data.numpy()

    def gradient_raw(self, text, segs, text_seg_indices, y):
        """

        Parameters
        ----------
        text : np.array of str
            text. shape = [batch_size, ]
        segs : np.array of list of str
            segs. 文本片段表示, shape 同 `text`
        text_seg_indices : np.array of list of list of int
            text_seg_indices. `text_seg_indices[i][j]` 是 `text[i][j]` 对应在 `segs` 中的下标列表
        y : np.array of int

        Returns
        -------
        np.array of list of float :
            gradient. shape 同 `segs`，`gradient[i][j]` 对应 `segs[i][j]` 的梯度

        """
        x, x_seg_indices = self.preprocess(text)
        seg_x_indices = self.get_seg_x_indices(segs, text_seg_indices, x_seg_indices)
        grad = self.gradient(x, y)
        grad_segs = np.empty_like(segs)
        for i, seg_x_indices_i in enumerate(seg_x_indices):
            grad_segs[i] = []
            for j, indices in enumerate(seg_x_indices_i):
                indices_filter_none = list(filter(lambda x: x is not None, indices))
                if len(indices_filter_none) == 0:
                    grad_segs[i].append(None)
                else:
                    pp = np.sum(grad[i, indices_filter_none]).tolist()
                    grad_segs[i].append(pp)
        return grad_segs


    def get_seg_x_indices(self, segs, text_seg_indices, text_x_indices):
        """

        Parameters
        ----------
        segs : np.array of list of str
            segs. 文本片段表示, shape 同 `text`
        text_seg_indices : np.array of list of list of int
            text_seg_indices. `text_seg_indices[i][j]` 是 `text[i][j]` 对应在 `segs` 中的下标列表
        text_x_indices : np.array of list of list of int
            text_x_indices. `text_x_indices[i][j]` 是 `text[i][j]` 对应在 `x` 中的下标列表

        Returns
        -------
        np.array of list of list of int :
            seg_x_indices. `seg_x_indices[i][j]` 是 `segs[i][j]` 对应在 `x` 中的下标列表

        """
        # np.array of list of list of int
        seg_text_indices = np.empty_like(text_seg_indices, dtype=object)
        for i, text_seg_indices_i in enumerate(text_seg_indices):
            seg_to_text_index_map = defaultdict(list)
            for j, indices in enumerate(text_seg_indices_i):
                for index in indices:
                    seg_to_text_index_map[index].append(j)

            seg_text_indices[i] = []
            for j in range(len(segs[i])):
                seg_text_indices[i].append(seg_to_text_index_map[j])

        seg_x_indices = np.empty_like(segs, dtype=object)
        for i, seg_text_indices_i in enumerate(seg_text_indices):
            text_x_indices_i = text_x_indices[i]
            seg_x_indices[i] = []
            for j, indices in enumerate(seg_text_indices_i):
                x_indices = set()
                for index in indices:
                    x_indices_ = text_x_indices_i[index]
                    x_indices.update(filter(lambda x: x is not None, x_indices_))
                seg_x_indices[i].append(sorted(x_indices))
        return seg_x_indices


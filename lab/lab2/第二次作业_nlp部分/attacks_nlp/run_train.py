# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
from models import text_cnn as x

#
# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', default='TextCNN', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# args = parser.parse_args()


if __name__ == '__main__':

    dataset = 'datas'  # 数据集
    embedding = 'embedding_pretrain.npz'
    model_name = 'TextCNN'
    word = False

    from utils import build_dataset, build_iterator, get_time_dif

    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    vocab, train_data, dev_data, test_data = build_dataset(config, word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.TextCNN(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)

import numpy as np
import os
import re
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
import random
import torch
import torch.nn as nn
import argparse


def train_dev_test_split(x, train=.7, dev=.1):
    train_idx = int(len(x) * train)
    dev_idx = int(len(x) * (train + dev))
    return x[:train_idx], x[train_idx:dev_idx], x[dev_idx:]


def to_array(X, n=2):
    return np.array([np.eye(n)[x] for x in X])


def per_class_prec(y, pred):
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    results = []
    for j in range(num_classes):
        class_y = y[:, j]
        class_pred = pred[:, j]
        p = precision_score(class_y, class_pred, average='binary')
        results.append([p])
    return np.array(results)


def per_class_rec(y, pred):
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    results = []
    for j in range(num_classes):
        class_y = y[:, j]
        class_pred = pred[:, j]
        rec = recall_score(class_y, class_pred, average='binary')
        results.append([rec])
    return np.array(results)


def macro_f1(y, pred):
    """Get the macro f1 score"""

    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)

    results = []
    for j in range(num_classes):
        class_y = y[:, j]
        class_pred = pred[:, j]
        f1 = f1_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results).mean()


def get_best_C(dataset, cross_dataset):
    """
    Find the best parameters on the dev set.
    """
    best_f1 = 0
    best_c = 0

    labels = sorted(set(dataset._ytrain))

    test_cs = [0.001, 0.003, 0.006, 0.009,
               0.01, 0.03, 0.06, 0.09,
               0.1, 0.3, 0.6, 0.9,
               1, 3, 6, 9,
               10, 30, 60, 90]
    for i, c in enumerate(test_cs):

        sys.stdout.write('\rRunning cross-validation: {0} of {1}'.format(i + 1, len(test_cs)))
        sys.stdout.flush()

        clf = LinearSVC(C=c)
        h = clf.fit(dataset._Xtrain, dataset._ytrain)
        pred = clf.predict(cross_dataset._Xdev)
        if len(labels) == 2:
            dev_f1 = macro_f1(cross_dataset._ydev, pred)
        else:
            dev_f1 = f1_score(cross_dataset._ydev, pred, labels=labels, average='macro')
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_c = c

    print()
    print('Best F1 on dev data: {0:.3f}'.format(best_f1))
    print('Best C on dev data: {0}'.format(best_c))

    return best_c, best_f1


def get_syn_ant(lang, vecs):
    # This is a quick way to import the sentiment synonyms and antonyms to check their behaviour during training.
    synonyms1 = [l.strip() for l in open(os.path.join('syn-ant', lang, 'syn1.txt')) if l.strip() in vecs._w2idx]
    synonyms2 = [l.strip() for l in open(os.path.join('syn-ant', lang, 'syn2.txt')) if l.strip() in vecs._w2idx]
    neg = [l.strip() for l in open(os.path.join('syn-ant', lang, 'neg.txt')) if l.strip() in vecs._w2idx]
    idx = min(len(synonyms1), len(synonyms2), len(neg))
    return synonyms1[:idx], synonyms2[:idx], neg[:idx]


def get_best_run(weightdir):
    """
    This returns the best dev f1, parameters, and weights from the models
    found in the weightdir.
    """
    best_params = []
    best_f1 = 0.0
    best_weights_path = ''
    print()
    print("get_best_run lokking in dir: {}".format(weightdir))
    print("os.listdir(weightdir): ", os.listdir(weightdir))
    for file in os.listdir(weightdir):
        # print("trying file " + str(file))
        epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
        batch_size = int(re.findall('[0-9]+', file.split('-')[-3])[0])
        alpha = float(re.findall('0.[0-9]+', file.split('-')[-2])[0])
        f1 = float(re.findall('0.[0-9]+', file.split('-')[-1])[0])
        if f1 > best_f1:
            best_params = [epochs, batch_size, alpha]
            best_f1 = f1
            weights = os.path.join(weightdir, file)
            best_weights_path = weights
    return best_f1, best_params, best_weights_path

# def new_get_best_run(weightdir):
#     """
#     This returns the best dev f1, parameters, and weights from the models
#     found in the weightdir.
#     """
#     best_params = []
#     best_f1 = 0.0
#     best_weights_path = ''
#     print()
#     print("get_best_run lokking in dir: {}".format(weightdir))
#     print("os.listdir(weightdir): ", os.listdir(weightdir))
#     for file in os.listdir(weightdir):
#         # print("trying file " + str(file))
#         epochs = int(re.findall('[0-9]+', file.split('-')[-5])[0])
#         batch_size = int(re.findall('[0-9]+', file.split('-')[-4])[0])
#         alpha = float(re.findall('0.[0-9]+', file.split('-')[-3])[0])
#         f1 = float(re.findall('0.[0-9]+', file.split('-')[-2])[0])
#         if f1 > best_f1:
#             best_params = [epochs, batch_size, alpha]
#             best_f1 = f1
#             weights = os.path.join(weightdir, file)
#             best_weights_path = weights
#     return best_f1, best_params, best_weights_path


def get_best_model_params(best_model_file_path):
    only_file_name = best_model_file_path.split('/')[-1]

    print("only_file_name: {}".format(only_file_name))
    params = only_file_name.split('-')
    epochs = int(re.findall('[0-9]+', params[-5])[0])
    batch_size = int(re.findall('[0-9]+', params[-4])[0])
    alpha = float(re.findall('0.[0-9]+', params[-3])[0])
    f1 = float(re.findall('0.[0-9]+', params[-2])[0])
    lr = float(re.findall('0.[0-9]+', params[-1])[0])

    best_params = [epochs, batch_size, alpha, lr]
    return f1, best_params


def print_prediction(model, cross_dataset, outfile):
    prediction = model.predict(cross_dataset._Xtest)
    with open(outfile, 'w') as out:
        for line in prediction:
            out.write('{0}\n'.format(line))


def shuffle_data(class_X, class_Y):
    c = list(zip(class_X, class_Y))
    random.Random(4).shuffle(c)
    class_X, class_Y = zip(*c)
    return np.array(class_X), np.array(class_Y)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mse_loss(x, y):
    # mean squared error loss
    return torch.sum((x - y) ** 2) / x.data.shape[0]


def cosine_loss(x, y):
    c = nn.CosineSimilarity()
    return (1 - c(x, y)).mean()


def sort_batch_by_sent_lens(x_batch, y_batch):
    sorted_by_x_len = sorted(zip(x_batch, y_batch), key=lambda pair: len(pair[0]), reverse=True)
    x_batch, y_batch = map(list, zip(*sorted_by_x_len))

    return x_batch, y_batch


def prepare_batch(x_batch_sorted_by_len):
    # get the length of each sentence
    batch_lengths = [len(sentence) for sentence in x_batch_sorted_by_len]

    # create an empty matrix with padding tokens
    pad_token = 0
    longest_sent = max(batch_lengths)
    batch_size = len(x_batch_sorted_by_len)
    padded_batch = np.ones((batch_size, longest_sent)) * pad_token
    # copy over the actual sequences
    for i, sent_len in enumerate(batch_lengths):
        sequence = x_batch_sorted_by_len[i]
        padded_batch[i, 0:sent_len] = sequence[:sent_len]

    x_batch_sorted_by_len = [torch.cuda.LongTensor(l) for l in padded_batch]
    return torch.stack(x_batch_sorted_by_len), batch_lengths

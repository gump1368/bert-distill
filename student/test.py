#! -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, recall_score

from student.processing import Processing
from student.model import StudentNetwork
from utils.configure import Args

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = Args.args
process = Processing()
process.load_data()

model_path = os.path.join(args['data_dir'], 'models/st_14.model')
model = StudentNetwork(vocab_size=2104, embedding_size=200, hidden_size=512)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.to(device)
model.eval()


def similarity(sentence_a, sentence_b):
    sentence_a = ['bos']+sentence_a.strip().split()+['eos']
    sentence_b = ['bos']+sentence_b.strip().split()+['eos']

    sentence_a = [process.word2id.get(item, 0) for item in sentence_a]
    sentence_b = [process.word2id.get(item, 0) for item in sentence_b]

    sentence_a = torch.LongTensor(sentence_a).view(1, -1)
    sentence_b = torch.LongTensor(sentence_b).view(1, -1)
    with torch.no_grad():
        output = model(sentence_a, sentence_b).squeeze()

        score = F.softmax(output, -1).numpy()[1].item()

    return score


if __name__ == '__main__':
    import time
    s1 = '你 好 琥珀 你 的 男 朋友 是 谁'
    s2 = '琥珀 你 男 朋友 是 谁'
    be = time.time()
    sim = similarity(s1, s2)

    print(time.time()-be, sim)

# examples = process.create_examples()
#
# thresholds = []
# start_threshold = 0.6
# while start_threshold < 0.99:
#     thresholds.append(start_threshold)
#     start_threshold += 0.01
#
# y_pred = []
# y_true = []
# for item in examples:
#     s1, s2, label = item
#
#     sentences_a = torch.LongTensor(s1).view(1, -1)  # .to(device)
#     sentences_b = torch.LongTensor(s2).view(1, -1)  # .to(device)
#     with torch.no_grad():
#         output = model(sentences_a, sentences_b).squeeze()
#
#         score = F.softmax(output).numpy()[1].item()
#         pred = [1 if score >= threshold else 0 for threshold in thresholds]
#         y_true.append(int(label))
#         y_pred.append(pred)
#
# y_pred = np.array(y_pred)
# best_threshold_index = 0
# best_score = 0
# for index in range(len(thresholds)):
#     y_ = y_pred[:, index].tolist()
#     score = accuracy_score(y_true, y_)
#     if score > best_score:
#         best_score = score
#         best_threshold_index = index
#
# best_threshold = thresholds[best_threshold_index]
# print('best threshold: {}, best score:{}'.format(best_threshold, best_score))
#
# y_pred_best = y_pred[:, best_threshold_index].tolist()
# #
# acc = accuracy_score(y_true, y_pred_best)
# f1 = f1_score(y_true, y_pred_best, average='weighted')
# recall = recall_score(y_true, y_pred_best, average='weighted')
# print('accuracy: {}, recall: {}, f1: {}'.format(acc, recall, f1))
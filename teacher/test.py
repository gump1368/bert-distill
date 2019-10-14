#! -*-coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, recall_score

from transformers import BertForSequenceClassification

from teacher.processing import Processing
from utils.configure import Args

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = Args.args

model = BertForSequenceClassification.from_pretrained(args['output_dir'])
model.to(device)
model.eval()

process = Processing()
examples = process.create_examples()

thresholds = []
start_threshold = 0.6
while start_threshold < 0.99:
    thresholds.append(start_threshold)
    start_threshold += 0.01

y_pred = []
y_true = []
for item in examples:
    token_id, token_type, mask, label = item
    token_id = torch.LongTensor([token_id]).to(device)
    token_type = torch.LongTensor([token_type]).to(device)
    mask = torch.LongTensor([mask]).to(device)
    with torch.no_grad():
        output = model(input_ids=token_id,
                       attention_mask=mask,
                       token_type_ids=token_type)

        score = F.sigmoid(output[0]).item()
        pred = [1 if score >= threshold else 0 for threshold in thresholds]
        y_true.append(int(label))
        y_pred.append(pred)

y_pred = np.array(y_pred)
best_threshold_index = 0
best_score = 0
for index in range(len(thresholds)):
    y_ = y_pred[:, index].tolist()
    score = accuracy_score(y_true, y_)
    if score > best_score:
        best_score = score
        best_threshold_index = index

best_threshold = thresholds[best_threshold_index]
print('best threshold: {}'.format(best_threshold))

y_pred_best = y_pred[:, best_threshold_index].tolist()

acc = accuracy_score(y_true, y_pred_best)
f1 = f1_score(y_true, y_pred_best)
recall = recall_score(y_true, y_pred_best)
print('accuracy: {}, recall: {}, f1: {}'.format(acc, recall, f1))
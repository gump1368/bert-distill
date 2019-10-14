#! -*- coding: utf-8 -*-
import os
import math
import logging
import random
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from utils.configure import Args
from student.processing import Processing
from student.model import StudentNetwork

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level='INFO',
                    format='%(asctime)s-%(levelname)s-%(name)s-%(message)s')

args = Args.args
batch_size = args['train_batch_size']
num_train_epochs = 20
learning_rate = 0.001

logging.info('prepare for data...')
process = Processing()
process.load_data()
examples = process.create_examples()
vocab_size = len(process.word2id)
embedding = torch.from_numpy(process.embedding).float()


logging.info('model...')
model = StudentNetwork(vocab_size=vocab_size, embedding_size=200, hidden_size=512, embedded=embedding)
model.to(device)

# loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.001)

total = len(examples)
last_training_loss = 100000000000


total_step = math.ceil(total/batch_size)
for epoch in range(num_train_epochs):
    model.train()
    random.shuffle(examples)
    start = 0
    training_loss = 0
    for _ in tqdm(range(int(total_step)), total=total_step):
        batch = examples[start: start+batch_size]
        start += batch_size

        for line in batch:
            s1, s2, label = line

            sentences_a = torch.LongTensor(s1).view(1, -1).to(device)
            sentences_b = torch.LongTensor(s2).view(1, -1).to(device)
            labels = torch.LongTensor([int(label)]).to(device)

            logits = model(sentences_a, sentences_b)

            # loss = loss_function(logits, labels)
            loss = loss_function(logits, labels)

            training_loss += loss.item()

            loss.backward()

        optimizer.step()
        model.zero_grad()

    logging.info('epoch: {}, training loss: {}'.format(epoch, training_loss/len(examples)))

    #eval
    model.eval()
    y_true = []
    y_pred = []
    for line in examples:
        s1_, s2_, label_ = line

        sentences_a = torch.LongTensor(s1_).view(1, -1).to(device)
        sentences_b = torch.LongTensor(s2_).view(1, -1).to(device)
        labels = torch.LongTensor([int(label_)]).to(device)

        logits = model(sentences_a, sentences_b).squeeze()
        pred_label = torch.argmax(logits).item()

        y_true.append(int(label_))
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    logging.info('accuracy: {}, recall: {}, f1: {}'.format(acc, recall, f1))

    if training_loss < last_training_loss:
        last_training_loss = training_loss
        if epoch >= 10:
            save_path = os.path.join(process.data_dir, 'models/st_{}.model'.format(epoch))
            torch.save(model.state_dict(), save_path)
            logging.info('saving model at {}'.format(save_path))
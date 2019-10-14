#! -*- coding: utf-8 -*-
"""学生网络，采用bilstm对句子进行编码"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, embedded=None):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        if embedded is not None:
            self.embedding.weight = nn.Parameter(embedded)

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, input_seq):
        embedding_seq = self.embedding(input_seq)

        # packed = nn.utils.rnn.pack_padded_sequence(embedding_seq, seq_length)

        output, _ = self.lstm(embedding_seq)

        # unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # output = output[:, -1, :self.hidden_size] + output[:, -1, self.hidden_size:]

        return output[:, -1, :]


class StudentNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, embedded=None):
        super(StudentNetwork, self).__init__()

        self.rnn = RNN(vocab_size=vocab_size, embedding_size=embedding_size,
                       hidden_size=hidden_size, embedded=embedded)

        # self.fc_mul = nn.Sequential(
        #     nn.Linear(hidden_size*2, hidden_size),
        #     nn.ReLU()
        # )
        # self.fc_minus = nn.Sequential(
        #     nn.Linear(hidden_size*4, hidden_size),
        #     nn.ReLU()
        # )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*4*2, hidden_size),
            nn.ReLU()
        )

        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, seq_a, seq_b):
        rnn_a = self.rnn(seq_a)
        rnn_b = self.rnn(seq_b)

        mul = rnn_a.mul(rnn_b)
        # mul_feature = self.fc_mul(mul)

        minus = rnn_a.add(-rnn_b)
        # minus_feature = self.fc_minus(minus)

        logits = self.fc(torch.cat([rnn_a, rnn_b, mul, minus], -1))

        return self.classifier(logits)


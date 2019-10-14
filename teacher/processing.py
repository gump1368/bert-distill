#! -*- cosing: utf-8 -*-
"""bert 数据预处理"""
import os
import csv
from utils.configure import Args
from transformers import BertTokenizer

config = Args.args


class Processing(object):
    def __init__(self):
        self.data_dir = config.get('data_dir', '')
        self.bert_model_dir = config.get('bert_model_dir', '')
        self.max_seq_length = config.get('max_seq_length', '')
        self.tokenizer = None

        if self.bert_model_dir:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir)

    def read_data(self):
        if not self.data_dir:
            raise FileNotFoundError('Can\'t get data dir')

        file_path = os.path.join(self.data_dir, 'sim_source.csv')
        with open(file_path, encoding='utf-8') as f:
            file = csv.reader(f)
            next(file)  # 去除头

            for line in file:
                s1 = line[0]
                s2 = line[1]
                label = line[3]

                yield (s1, s2, label)

    def build_example(self, s1, s2, is_padded=False):
        assert self.tokenizer is not None

        s1 = self.tokenizer.tokenize(s1)
        s2 = self.tokenizer.tokenize(s2)

        s = ['[CLS]'] + s1 + ['[SEP]'] + s2 + ['[SEP]']

        token_id = self.tokenizer.convert_tokens_to_ids(s)
        token_type = [0]*(len(s1)+2)+[1]*(len(s2)+1)
        mask = [1]*len(token_id)

        assert len(token_id) == len(token_type) == len(mask) <= self.max_seq_length

        if is_padded:
            padding = [0]*(self.max_seq_length-len(token_id))
            token_id += padding
            token_type += padding
            mask += padding

        return token_id, token_type, mask

    def create_examples(self):
        data = self.read_data()
        examples = []
        for line in data:
            s1, s2, label = line
            token_id, token_type, mask = self.build_example(s1, s2, is_padded=True)
            examples.append((token_id, token_type, mask, label))
        return examples

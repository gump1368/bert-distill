#! -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import requests
import collections

from utils.configure import Args

args = Args.args


class Processing(object):
    def __init__(self):
        self.data_dir = args.get('data_dir', '')
        self.word2id = collections.defaultdict(int)
        self.id2word = collections.defaultdict(str)
        self.embedding = None

        self.word2id['unk'] = 0
        self.id2word[str(0)] = 'unk'

    def load_data(self):
        vocab_path = os.path.join(self.data_dir, 'vocabulary.txt')
        if os.path.isfile(vocab_path):
            with open(vocab_path, encoding='utf-8') as f:
                vocabulary = f.readlines()
                for index, item in enumerate(vocabulary):
                    item = item.strip()
                    self.word2id[item] = index
                    self.id2word[str(index)] = item
        else:
            print('can\'t find vocabulary file!')

        embedding_path = os.path.join(self.data_dir, 'embedding.npz')
        if os.path.isfile(embedding_path):
            self.embedding = np.load(embedding_path)['embedding']
        else:
            print('can\'t find embedding file!')

    def read_data(self):
        if not self.data_dir:
            raise FileNotFoundError('Can\'t get data dir')

        file_path = os.path.join(self.data_dir, 'sim_source.csv')
        with open(file_path, encoding='utf-8') as f:
            file = csv.reader(f)
            next(file)  # 去除头

            for line in file:
                s1 = self.machinery_segment(line[0])
                s2 = self.machinery_segment(line[1])
                label = line[3]

                yield (s1, s2, label)

    def build_vocabulary(self, data, eos_bos_unk=True, save=False):
        vocabulary = collections.defaultdict(int)
        for line in data:
            if not isinstance(line, list):
                line = line.strip().split()
            for item in line:
                vocabulary[item] += 1

        vocabulary = collections.OrderedDict(sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))

        num = 1
        if eos_bos_unk:
            self.word2id['bos'] = 1
            self.word2id['eos'] = 2
            self.id2word['1'] = 'bos'
            self.id2word['2'] = 'eos'
            num += 2

        for item in vocabulary.keys():
            self.word2id[item] = num
            self.id2word[str(num)] = item
            num += 1

        if save:
            with open(os.path.join(self.data_dir, 'vocabulary.txt'), 'w', encoding='utf-8') as f:
                for item in self.word2id.keys():
                    f.write(item + '\n')

    def build_embedding(self, save=True):
        embedding = np.zeros((len(self.word2id), 200))
        for index, word in enumerate(self.word2id.keys()):
            word_embedding = self.post_embedding(word)
            if word_embedding:
                embedding[index] = np.array(word_embedding)
        embedding[0] = 0
        self.embedding = embedding
        if save:
            np.savez_compressed(os.path.join(self.data_dir, 'embedding.npz'), embedding=embedding)

    @classmethod
    def post_embedding(cls, word):
        url = 'http://172.27.1.206:11109/vector'
        data = {
            'type': 'word_vector',
            'data': {
                'word_list': [word]
            }
        }
        try:
            resp = requests.post(url, json=data).json()
            vector = resp['data'].get(word, [])
            not vector and print(f'{word} has no vector')
            return vector
        except:
            return []

    @classmethod
    def machinery_segment(cls, sentence):
        url = 'http://172.19.1.23:3333/machinery_segment/'
        data = {
            "sentence": sentence
        }
        try:
            resp = requests.post(url, json=data).json()
        except Exception as e:
            print(e)
            return []

        result = resp.get('data', '').get('seg', [])
        if result:
            return result
        return []

    def build_example(self, sentence):
        if not isinstance(sentence, list):
            sentence = sentence.strip.split()

        sentence = ['bos'] + sentence + ['eos']
        sentence2id = [self.word2id[item] for item in sentence]
        return sentence2id

    def create_examples(self):
        data = self.read_data()
        examples = []
        for line in data:
            s1, s2, label = line
            id_s1 = self.build_example(s1)
            id_s2 = self.build_example(s2)
            examples.append((id_s1, id_s2, label))
        return examples


# if __name__ == '__main__':
#     process = Processing()
#     data = process.read_data()
#     all_data = []
#     for line in data:
#         s1, s2, _ = line
#         all_data.append(s1)
#         all_data.append(s2)
#
#     process.build_vocabulary(all_data, save=True)
#     process.build_embedding()
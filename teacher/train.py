#! -*- coding: utf-8 -*-
import os
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, WarmupLinearSchedule, BertForSequenceClassification

from utils.configure import Args
from teacher.processing import Processing

config = Args.args
logging.info('configure: {}'.format(config))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level='INFO',
                    format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%m/%d/%Y %H:%M:S')

logging.info('prepare data ....')
processing = Processing()
train_examples = processing.create_examples()
data_loader = DataLoader(train_examples, batch_size=config['train_batch_size'], shuffle=True)

logging.info('create model')
model = BertForSequenceClassification.from_pretrained(config['bert_model_dir'], num_labels=1)
model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]
t_total = len(data_loader) // config['gradient_accumulation_steps'] * config['num_train_epochs']
optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config['warmup_steps'], t_total=t_total)

model.train()
# batch_size = config['train_batch_size']
last_training_loss = 10000000000000000
for epoch in range(config['num_train_epochs']):
    training_loss = 0
    for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc='training'):
        # for item in batch:
        token_id, token_type, mask, label = batch
        size = token_id[0].shape[0]
        # token_ids.append(token_id)
        # token_types.append(token_type)
        # masks.append(mask)
        # labels.append(label)
        token_ids = torch.cat([item.view(size, -1) for item in token_id], -1).to(device)
        token_types = torch.cat([item.view(size, -1) for item in token_type], -1).to(device)
        masks = torch.cat([item.view(size, -1) for item in mask], -1).to(device)
        labels = torch.FloatTensor([int(item) for item in label]).to(device)

        outputs = model(input_ids=token_ids,
                        attention_mask=masks,
                        token_type_ids=token_types,
                        labels=labels)

        loss = outputs[0]
        if config['gradient_accumulation_steps'] > 1:
            loss = loss / config['gradient_accumulation_steps']

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        training_loss += loss.item()

        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            optimizer.step()

            scheduler.step()  # Update learning rate schedule

            model.zero_grad()

    logging.info('epoch: {}, training loss: {}'.format(epoch, training_loss/len(train_examples)))

    if training_loss < last_training_loss:
        last_training_loss = training_loss
        logging.info('saving model......')
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(config['output_dir'])
        # output_model_file = os.path.join(config['output_dir'], 'model_' + str(epoch) + '.bin')
        # torch.save(model_to_save.state_dict(), output_model_file)
        # output_config_file = os.path.join(config['output_dir'], 'config_' + str(epoch) + '.json')
        # with open(output_config_file, 'w') as f:
        #     f.write(model_to_save.config.to_json_string())



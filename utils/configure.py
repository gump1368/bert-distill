#! -*-coding: utf-8-*-
class Args:

    args = {
        "data_dir": '/data/gump/project-data/sentence_similarity',
        "bert_model_dir": '/data/gump/bert_chinese/chinese_L-12_H-768_A-12/',
        "task_name": "sentence_similarity",
        "output_dir": '/data/gump/project-data/sentence_similarity/models/',

        "cache_dir": None,
        "max_seq_length": 50,
        "do_train": True,
        "do_eval": True,
        "do_lower_case": True,
        "train_batch_size": 64,
        "eval_batch_size": 32,
        "learning_rate": 3e-5,
        "num_train_epochs": 10,
        "warmup_proportion": 0.1,
        "no_cuda": False,
        "local_rank": -1,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": False,
        "fp16": False,
        "loss_scale": 0,
        "server_ip": '',
        "server_port": '',
        "layers": -1,
        'weight_decay': 0.0,
        'adam_epsilon': 1e-8,
        'warmup_steps': 0,
        'max_grad_norm': 1.0
    }
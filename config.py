from pathlib import Path


def get_config():
    return {
        "batch_size":8,
        "num_epochs":20,
        "lr":10**-4,
        "seq_len":350,
        "d_model":512,
        "lang_src":"en",
        "lang_tgt":"it",
        "model_folder":"weights",
        "model_filename":"tmodel_",
        "preload":None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/tmodel",

        # 固定训练/验证集的划分，让 test.py 评估的是训练时留出的那批句子
        "seed":42,
        # 每轮验证最多跑多少个 batch。验证集有 3000+ 句，
        # 全跑一遍比一个训练 epoch 还慢，抽一部分估个趋势就够了
        "val_batches":50,
    }
def get_weight_file_path(config,epoch:str):
    model_folder=config['model_folder']
    model_basename=config['model_filename']
    model_filename=f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

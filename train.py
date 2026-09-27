import csv
import time
import warnings

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BillingualDataset,causal_mask
from model import Transformer, build_transformer
from config import get_weight_file_path,get_config
from tqdm import tqdm

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path=Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    #数据集名由源语言和目标语言拼出来，注意中间是 '-' 不是减号
    #必须写全名 namespace/name：新版 huggingface_hub 不再自动补全短名，写 'opus_books' 会报 HfUriError
    ds_raw=load_dataset('Helsinki-NLP/opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')

    #build tokenizer
    tokenizer_src=get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt=get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    #random_split 第二个参数是每个子集长度的 list
    # 固定 seed：否则每次跑 train.py 留出的验证集都不一样，test.py 的 BLEU 也没法跨次比较
    generator=torch.Generator().manual_seed(config['seed'])
    train_ds_raw,val_ds_raw=random_split(ds_raw,[train_ds_size,val_ds_size],generator=generator)

    train_ds=BillingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds=BillingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src=0
    max_len_tgt=0
    for item in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids=tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

    print(f'max len of src sentence: {max_len_src}')
    print(f'max len of tgt sentence: {max_len_tgt}')

    train_dataloader=DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    # 验证集不 shuffle：同一轮的 val_loss 才有可比性。batch 也不用压到 1，纯属浪费
    val_dataloader=DataLoader(val_ds,batch_size=config['batch_size'],shuffle=False)
    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt
def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model

def run_validation(model,val_dataloader,loss_fn,tokenizer_tgt,config,device):
    """在验证集的一个子集上算平均损失。

    max_batches 是为了控制耗时：验证集 3000+ 句，全跑完比一个训练 epoch 还慢。
    """
    model.eval()
    total_loss=0.0
    n_batches=0
    with torch.no_grad():
        for batch in val_dataloader:
            if n_batches>=config['val_batches']:
                break

            encoder_input=batch['encoder_input'].to(device)
            decoder_input=batch['decoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)
            decoder_mask=batch['decoder_mask'].to(device)
            label=batch['label'].to(device)

            encoder_output=model.encode(encoder_input,encoder_mask)
            decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            proj_output=model.project(decoder_output)

            loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
            total_loss+=loss.item()
            n_batches+=1

    model.train()
    return total_loss/max(n_batches,1)

def train_model(config):
    #
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt=get_ds(config)
    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    #TensorBorad
    writer=SummaryWriter(config['experiment_name'])

    # 同时落一份 CSV：plot_loss.py 不依赖 tensorboard 也能画图，
    # 而且远端跑完直接把 CSV 拉回本机就行。用追加模式，接 preload 续跑不会覆盖之前的记录
    run_dir=Path(config['experiment_name'])
    run_dir.mkdir(parents=True,exist_ok=True)
    step_log=open(run_dir/'loss_history.csv','a',newline='')
    step_writer=csv.writer(step_log)
    if step_log.tell()==0:
        step_writer.writerow(['step','epoch','loss','timestamp'])
    epoch_log=open(run_dir/'epoch_metrics.csv','a',newline='')
    epoch_writer=csv.writer(epoch_log)
    if epoch_log.tell()==0:
        epoch_writer.writerow(['epoch','train_loss','val_loss','elapsed_sec','lr'])

    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)
    initial_epoch=0
    global_step=0
    if config['preload']:
        model_filename=get_weight_file_path(config,config['preload'])
        print(f'Preloading model{model_filename}')
        state=torch.load(model_filename)
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state['global_step']

    #label 是由 tgt 分词器生成的，所以 ignore_index 也要用 tgt 分词器的 [PAD] id
    loss_fn=CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        epoch_start=time.time()
        running_loss=0.0
        n_batches=0
        batch_iterator=tqdm(train_dataloader,desc=f'Processing epoch{epoch:02d}')
        for batch in batch_iterator:

            encoder_input=batch['encoder_input'].to(device) #(b,Seq_len)
            decoder_input=batch['decoder_input'].to(device)#(b,Seq_len)
            encoder_mask=batch['encoder_mask'].to(device)#(B,1,1,Seq_len)
            decoder_mask=batch['decoder_mask'].to(device)#(B,1,Seq_len,Seq_len)

            #encode/decode 内部会做词嵌入和位置编码，不要直接调 model.encoder/model.decoder
            encoder_output=model.encode(encoder_input,encoder_mask)#(B,seq_len,d_model)
            decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)#(B,seq_len,d_model)
            proj_output=model.project(decoder_output)

            label=batch['label'].to(device)#(B,Seq_len)
            #(B,Seq_len,tgt_vocab_size)-->(B * seq_len,tgt_vocab_size)
            loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))

            running_loss+=loss.item()
            n_batches+=1
            batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss',loss.item(),global_step)
            writer.flush()
            step_writer.writerow([global_step,epoch,f'{loss.item():.6f}',f'{time.time():.1f}'])

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            global_step+=1

        # ---- 每轮结束：跑一遍验证集，把三个量都记下来 ----
        train_loss_epoch=running_loss/max(n_batches,1)
        val_loss=run_validation(model,val_dataloader,loss_fn,tokenizer_tgt,config,device)
        elapsed=time.time()-epoch_start
        lr=optimizer.param_groups[0]['lr']

        writer.add_scalar('train_loss_epoch',train_loss_epoch,epoch)
        writer.add_scalar('val_loss',val_loss,epoch)
        writer.flush()

        epoch_writer.writerow([epoch,f'{train_loss_epoch:.6f}',f'{val_loss:.6f}',f'{elapsed:.1f}',f'{lr:.3e}'])
        step_log.flush()
        epoch_log.flush()

        print(f'epoch {epoch:02d} | train_loss {train_loss_epoch:.4f} | '
              f'val_loss {val_loss:.4f} | {elapsed:.0f}s')

        #save the model
        model_filename=get_weight_file_path(config,f'{epoch:02d}')
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'global_step':global_step
        },model_filename)

    step_log.close()
    epoch_log.close()
    writer.close()

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    config=get_config()
    train_model(config)

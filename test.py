"""测试/评估训练好的 Transformer（英->意）。

三种模式，可以只跑其中一种，也可以组合：

  --dry-run        不加载权重，用随机初始化的模型跑一次前向+解码，
                   只为验证环境、维度和数据流打通（几十秒，不用等训练完）
  --text "句子"    翻译给定的英文句子，可重复传多次
  --bleu N         在验证集上抽 N 句做贪心解码，算语料级 BLEU 并打印对照样例

用法:
    python test.py --dry-run
    python test.py --text "I love you." --text "The cat is on the table."
    python test.py --bleu 100 --out test_results.json
    python test.py --checkpoint weights/tmodel_19.pt --text "Good morning."
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import get_config
from dataset import BillingualDataset, causal_mask
from model import build_transformer


# --------------------------------------------------------------------------
# 模型 / 权重
# --------------------------------------------------------------------------
def pick_latest_checkpoint(config):
    """weights/ 下形如 tmodel_07.pt，按轮次取最大的那个。"""
    folder = Path(config['model_folder'])
    pattern = re.compile(re.escape(config['model_filename']) + r'(\d+)\.pt$')
    candidates = []
    for path in folder.glob(f'{config["model_filename"]}*.pt'):
        match = pattern.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates)[1]


def build_model(config, device):
    """词表大小从 tokenizer 文件读，不需要下载数据集。"""
    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    model = build_transformer(
        tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
        config['seq_len'], config['seq_len'], config['d_model'],
    ).to(device)
    return model, tokenizer_src, tokenizer_tgt


# --------------------------------------------------------------------------
# 解码
# --------------------------------------------------------------------------
def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    """逐 token 取概率最大的那个，遇到 [EOS] 或到 max_len 就停。"""
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)  # 编码器只跑一次，解码循环里复用
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # 只看最后一个位置的输出，那就是下一个 token 的预测
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1,
        )
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)


def encode_source(text, tokenizer_src, seq_len, device):
    """把一句英文按训练时的同一套规则补齐成 (1, seq_len)。"""
    sos = tokenizer_src.token_to_id('[SOS]')
    eos = tokenizer_src.token_to_id('[EOS]')
    pad = tokenizer_src.token_to_id('[PAD]')

    ids = tokenizer_src.encode(text).ids
    if len(ids) > seq_len - 2:          # 超长就截断，别让测试脚本直接崩
        ids = ids[:seq_len - 2]
    tokens = [sos] + ids + [eos] + [pad] * (seq_len - len(ids) - 2)

    source = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)
    source_mask = (source != pad).unsqueeze(0).unsqueeze(0).int().to(device)
    return source, source_mask


@torch.no_grad()
def translate(model, text, tokenizer_src, tokenizer_tgt, seq_len, device, max_decode_len):
    model.eval()
    source, source_mask = encode_source(text, tokenizer_src, seq_len, device)
    out = greedy_decode(model, source, source_mask, tokenizer_tgt, max_decode_len, device)
    return tokenizer_tgt.decode(out.detach().cpu().numpy(), skip_special_tokens=True)


# --------------------------------------------------------------------------
# 验证集 BLEU
# --------------------------------------------------------------------------
def get_val_dataset(config):
    """复现 train.py 里那个划分。两边都用 config['seed'] 固定，所以是同一批句子。"""
    from datasets import load_dataset
    from torch.utils.data import random_split

    ds_raw = load_dataset(
        'Helsinki-NLP/opus_books',
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split='train',
    )
    n_train = int(0.9 * len(ds_raw))
    generator = torch.Generator().manual_seed(config['seed'])
    _, val_raw = random_split(ds_raw, [n_train, len(ds_raw) - n_train], generator=generator)

    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    val_ds = BillingualDataset(
        val_raw, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len'],
    )
    return val_ds, tokenizer_src, tokenizer_tgt


def run_bleu(model, config, device, n_sentences, max_decode_len, stride):
    val_ds, tokenizer_src, tokenizer_tgt = get_val_dataset(config)

    predictions, references, examples = [], [], []
    n_sentences = min(n_sentences, len(val_ds))

    # 顺序抽句子；opus_books 是按书排列的，取太靠前会全是同一本书的同一批题材
    for i in range(n_sentences):
        item = val_ds[(i * stride) % len(val_ds)]
        src_text, tgt_text = item['src_text'], item['tgt_text']
        predicted = translate(model, src_text, tokenizer_src, tokenizer_tgt,
                              config['seq_len'], device, max_decode_len)
        predictions.append(predicted)
        references.append(tgt_text)
        if len(examples) < 10:
            examples.append({'src': src_text, 'ref': tgt_text, 'pred': predicted})

    from sacrebleu.metrics import BLEU
    score = BLEU().corpus_score(predictions, [references])
    return score, examples, predictions, references


# --------------------------------------------------------------------------
# dry-run：不依赖权重，验证整条链路
# --------------------------------------------------------------------------
def dry_run(model, config, device, tokenizers):
    tokenizer_src, tokenizer_tgt = tokenizers
    print('[dry-run] 随机初始化权重，只为验证维度和数据流，输出没有意义\n')

    text = 'Hello, how are you today?'
    t0 = time.time()
    source, source_mask = encode_source(text, tokenizer_src, config['seq_len'], device)
    print(f'  encoder_input  {tuple(source.shape)}  (期望 (1, {config["seq_len"]}))')
    print(f'  encoder_mask   {tuple(source_mask.shape)}  (期望 (1, 1, 1, {config["seq_len"]}))')

    model.eval()
    with torch.no_grad():
        enc_out = model.encode(source, source_mask)
        print(f'  encoder_output {tuple(enc_out.shape)}  '
              f'(期望 (1, {config["seq_len"]}, {config["d_model"]}))')

        tgt_len = 16
        decoder_input = torch.randint(0, tokenizer_tgt.get_vocab_size(), (1, tgt_len), device=device)
        decoder_mask = causal_mask(tgt_len).type_as(source_mask).to(device)
        dec_out = model.decode(enc_out, source_mask, decoder_input, decoder_mask)
        proj = model.project(dec_out)
        print(f'  decoder_output {tuple(dec_out.shape)}')
        print(f'  proj_output    {tuple(proj.shape)}  (期望 (1, {tgt_len}, {tokenizer_tgt.get_vocab_size()}))')

        assert dec_out.shape == (1, tgt_len, config['d_model'])
        assert proj.shape == (1, tgt_len, tokenizer_tgt.get_vocab_size())
        # log_softmax 的输出必须归一化到 log 概率，exp 之后和要为 1
        prob_sum = proj.exp().sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4), \
            f'projection 层没做归一化: {prob_sum.flatten()[:5]}'
        print(f'  softmax 归一化检查通过 (exp 后求和 ≈ 1)')

        out = greedy_decode(model, source, source_mask, tokenizer_tgt, max_len=12, device=device)
        print(f'  greedy_decode  {tuple(out.shape)}  tokens={out.cpu().tolist()[:12]}')

    print(f'\n[dry-run] 全部通过，耗时 {time.time() - t0:.1f}s')
    print('[dry-run] 注意：权重是随机初始化的，翻译结果是乱码属正常。')


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='测试/评估 Transformer 英->意翻译')
    parser.add_argument('--checkpoint', default=None,
                        help='权重路径，默认自动取 weights/ 下轮次最大的')
    parser.add_argument('--text', action='append', default=[],
                        help='要翻译的英文句子，可重复传')
    parser.add_argument('--bleu', type=int, default=0, metavar='N',
                        help='在验证集上抽 N 句算 BLEU')
    parser.add_argument('--dry-run', action='store_true',
                        help='不加载权重，用随机模型验证链路（不想等训练完就用这个）')
    parser.add_argument('--stride', type=int, default=17,
                        help='抽验证集句子时的步长，默认 17')
    parser.add_argument('--max-decode-len', type=int, default=128,
                        help='解码最长 token 数，默认 128。贪心解码是逐 token 串行跑的，'
                             '调到 350(seq_len) 会让每句话慢好几倍')
    parser.add_argument('--out', default=None, help='把结果写到 JSON')
    args = parser.parse_args()

    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[info] device = {device}')
    if device.type == 'cuda':
        print(f'[info] gpu = {torch.cuda.get_device_name(0)}')

    model, tokenizer_src, tokenizer_tgt = build_model(config, device)

    if args.dry_run:
        dry_run(model, config, device, (tokenizer_src, tokenizer_tgt))

    checkpoint_loaded = False
    path = None
    if not args.dry_run or args.text or args.bleu:
        path = Path(args.checkpoint) if args.checkpoint else pick_latest_checkpoint(config)
        if path is None:
            if args.text or args.bleu:
                print(f'[error] 找不到权重文件（{config["model_folder"]}/{config["model_filename"]}*.pt）。'
                      f'训练还没跑完就先 `python test.py --dry-run` 验证链路。', file=sys.stderr)
                return 1
        else:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            checkpoint_loaded = True
            print(f'[info] 已加载 {path}  (epoch {state.get("epoch")}, '
                  f'global_step {state.get("global_step")})')

    results = {'device': str(device), 'checkpoint': str(path) if checkpoint_loaded else None}

    if args.text and checkpoint_loaded:
        print()
        for text in args.text:
            translation = translate(model, text, tokenizer_src, tokenizer_tgt,
                                    config['seq_len'], device, args.max_decode_len)
            print(f'  EN: {text}')
            print(f'  IT: {translation}\n')
            results.setdefault('translations', []).append({'src': text, 'pred': translation})

    if args.bleu and checkpoint_loaded:
        print(f'[info] 开始在验证集上评估 {args.bleu} 句（贪心解码是串行的，请耐心等）…')
        t0 = time.time()
        score, examples, predictions, references = run_bleu(
            model, config, device, args.bleu, args.max_decode_len, args.stride)
        elapsed = time.time() - t0

        print(f'\n===== BLEU =====')
        print(f'  {score}')
        print(f'  ({args.bleu} 句，耗时 {elapsed:.1f}s，'
              f'{elapsed / max(args.bleu, 1):.2f}s/句)\n')

        print('===== 样例对照 =====')
        for i, ex in enumerate(examples[:10], 1):
            print(f'  [{i}] EN : {ex["src"]}')
            print(f'      REF: {ex["ref"]}')
            print(f'      HYP: {ex["pred"]}\n')

        results['bleu'] = score.score
        results['n_sentences'] = args.bleu
        results['examples'] = examples
        results['predictions'] = predictions
        results['references'] = references

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f'[ok] 结果已写入 {args.out}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

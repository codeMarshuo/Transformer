# Transformer 英→意机器翻译

从零手写实现的 Transformer（*Attention Is All You Need*），不依赖 `torch.nn.Transformer`，所有组件——多头注意力、位置编码、残差+层归一化、编码器/解码器堆叠——都是自己实现并在 `model.py` 里逐行注释。

任务是在 `Helsinki-NLP/opus_books` 的 `en-it` 子集上做英译意。

![训练损失曲线](assets/loss_curve.png)

## 实测结果

在 RTX 3080 上训练 20 个 epoch（72760 步，约 2.6 小时），交叉熵损失从 **10.04 降到 2.24**。

实际翻译效果：

| 英文 | 输出 |
|---|---|
| I love you. | Io amo . |
| The cat is on the table. | Il gatto è sulla tavola . |
| He did not know what to say. | Egli non sapeva quello che diceva . |
| She went to the market to buy some bread. | Entrò nelle cose necessarie a preparare il pane . |
| The night was dark and the wind was cold. | La notte era fredda e il vento era fredda . |

短句基本正确；长句会明显劣化，偶尔出现性数一致错误（如 `vento` 阳性配了 `fredda` 阴性）。

**关于 BLEU：** 实测 50.91（100 句），但**这个数字不能当留出集成绩**。早期版本的 `train.py` 里 `random_split` 没有固定随机种子，验证集是每次运行随机划分的、不可复现；而现在用固定 seed 划出的「验证集」与训练集高度重叠（约 90% 的句子模型见过）。要拿到干净的留出集指标，需要用固定 seed 重新训练——`config.py` 里的 `seed` 已经加好，`train.py` 会用它。

## 项目结构

```
model.py        Transformer 全部组件的实现（含中文注释）
dataset.py      双语数据集、因果掩码、padding
train.py        训练循环，含每轮验证集损失
test.py         评估：贪心解码 / BLEU / 样例翻译
plot_loss.py    损失曲线（自带 tfevents 解析器，不依赖 tensorboard）
config.py       超参数集中配置
```

模型配置：`d_model=512`、`N=6` 层、`h=8` 头、`d_ff=2048`、`dropout=0.1`、`seq_len=350`、词表约 22463。

## 环境准备

`requirements.txt` 里**故意没有列 torch**。torch 必须按机器的 CUDA 版本安装，写一个不带版本号的 torch 会让 `pip install -r requirements.txt` 把它覆盖成默认构建，可能直接丢掉 GPU 支持。按官方指引单独装：

<https://pytorch.org/get-started/locally/>

其余依赖：

```bash
pip install -r requirements.txt
```

## 使用

### 训练

```bash
python train.py
```

会先下载数据集并训练分词器（首次运行较慢），权重存到 `weights/`，每轮的指标同时写入 `runs/tmodel/` 下的 TensorBoard 事件文件和两份 CSV（`loss_history.csv`、`epoch_metrics.csv`）。

### 测试

不加载权重、只验证环境和数据流是否打通（几十秒，训练没跑完也能用）：

```bash
python test.py --dry-run
```

翻译指定句子：

```bash
python test.py --text "I love you." --text "Good morning."
```

在验证集上算 BLEU：

```bash
python test.py --bleu 100 --out test_results.json
```

默认自动加载 `weights/` 下轮次最大的权重，也可以用 `--checkpoint` 指定。贪心解码是逐 token 串行执行的，`--max-decode-len`（默认 128）会显著影响速度。

### 损失曲线

```bash
python plot_loss.py                      # 默认读 runs/tmodel
python plot_loss.py --smooth 0.5 --dark
```

`plot_loss.py` 自带一个约 60 行的 protobuf/TFRecord 解析器，**不需要安装 tensorboard**，只要 matplotlib。数据源优先用 `loss_history.csv`，没有就回退解析 `events.out.tfevents.*`；同一目录下有多个互相重叠的事件文件时，会自动选取点数最多的那个。

## 在远端 GPU 服务器上运行

本项目主要在租用的 AutoDL 实例上训练。通过 PyCharm 的 SSH 终端（`Tools → Start SSH Session`）运行，**不要用 PyCharm 的 Run 按钮**——它每次会把整个项目上传到 `/tmp/<随机名>/` 再执行，实例重启即清空，长时间训练的 checkpoint 会白跑。

```bash
export HF_ENDPOINT=https://hf-mirror.com        # huggingface.co 国内不通，只有镜像可达
M=~/miniconda3; export PATH=$M/bin:$PATH        # 非交互 SSH 的 PATH 里没有 conda
cd /root/autodl-tmp/Transformer                 # 放数据盘，实例重启不丢

python train.py 2>&1 | tee train.log
python test.py --bleu 100
python plot_loss.py
```

## 说明

- `weights/`、`runs/`、`huggingface_cache/` 都在 `.gitignore` 里。权重大约 900MB/个（含优化器状态），不适合进仓库，需要的话请自行训练。
- `tokenizer_en.json` / `tokenizer_it.json` 已提交（约 900KB），这样 `test.py --text` 不必先下载数据集就能跑。它们由 `train.py` 首次运行时生成，删掉会自动重建。

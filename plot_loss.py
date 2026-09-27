"""画训练损失曲线，输出 PNG。

数据来源按优先级自动挑选：
  1. runs/<experiment>/loss_history.csv  —— train.py 每步写一行(step,loss)，
     若存在 epoch_metrics.csv 还会带上每轮的平均 train_loss / val_loss
  2. runs/<experiment>/events.out.tfevents.*  —— TensorBoard 事件文件

自带一个极简的 tfevents 解析器，所以远端不装 tensorboard 也能画图。

用法:
    python plot_loss.py                          # 自动用 runs/tmodel
    python plot_loss.py --run runs/tmodel --out loss_curve.png
    python plot_loss.py --smooth 0.9             # 调 EMA 平滑强度
    python plot_loss.py --dark                   # 出深色版
"""
import argparse
import csv
import glob
import os
import struct
import sys

import matplotlib
matplotlib.use('Agg')  # 远端没有显示器，必须在 import pyplot 之前设成无头后端
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---- 配色（已用 dataviz 校验器验证：CVD ΔE 24.7 / 常视 ΔE 33.6，全部 PASS）----
THEME = {
    'light': {
        'surface': '#fcfcfb', 'ink': '#0b0b0b', 'ink2': '#52514e',
        'muted': '#898781', 'grid': '#e1e0d9', 'axis': '#c3c2b7',
        'train': '#2a78d6', 'val': '#eb6834',
    },
    'dark': {
        'surface': '#1a1a19', 'ink': '#ffffff', 'ink2': '#c3c2b7',
        'muted': '#898781', 'grid': '#2c2c2a', 'axis': '#383835',
        'train': '#3987e5', 'val': '#d95926',
    },
}


# --------------------------------------------------------------------------
# 极简 protobuf / TFRecord 解析：只取 Event.step 和 Summary.Value.simple_value
# --------------------------------------------------------------------------
def _read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf, pos, end):
    """按 protobuf wire format 逐个吐 (field_number, wire_type, value)。"""
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:                      # varint
            value, pos = _read_varint(buf, pos)
        elif wire == 1:                    # 64-bit
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == 2:                    # length-delimited
            length, pos = _read_varint(buf, pos)
            value, pos = buf[pos:pos + length], pos + length
        elif wire == 5:                    # 32-bit
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f'unsupported wire type {wire}')
        yield field, wire, value


def _parse_summary_value(buf):
    """Summary.Value -> (tag, simple_value)"""
    tag, value = None, None
    for field, wire, raw in _iter_fields(buf, 0, len(buf)):
        if field == 1 and wire == 2:
            tag = raw.decode('utf-8', 'replace')
        elif field == 2 and wire == 5:     # simple_value 是 float
            value = struct.unpack('<f', raw)[0]
    return tag, value


def read_tfevents(path):
    """读一个 tfevents 文件，返回 {tag: [(step, value), ...]}"""
    series = {}
    with open(path, 'rb') as fh:
        while True:
            header = fh.read(8)
            if len(header) < 8:
                break
            length = struct.unpack('<Q', header)[0]
            fh.read(4)                     # length 的 crc
            data = fh.read(length)
            fh.read(4)                     # data 的 crc
            if len(data) < length:
                break

            step = 0
            for field, wire, raw in _iter_fields(data, 0, len(data)):
                if field == 2 and wire == 0:
                    step = raw
                elif field == 5 and wire == 2:
                    for sub, sub_wire, sub_raw in _iter_fields(raw, 0, len(raw)):
                        if sub == 1 and sub_wire == 2:
                            tag, value = _parse_summary_value(sub_raw)
                            if tag and value is not None:
                                series.setdefault(tag, []).append((step, value))
    return series


# --------------------------------------------------------------------------
# 数据源
# --------------------------------------------------------------------------
def load_from_csv(run_dir):
    """读 train.py 落盘的 CSV。返回 (steps, train, epochs, train_mean, val_mean)"""
    steps, train = [], []
    history = os.path.join(run_dir, 'loss_history.csv')
    if os.path.exists(history):
        with open(history, newline='') as fh:
            for row in csv.DictReader(fh):
                steps.append(int(row['step']))
                train.append(float(row['loss']))

    epochs, train_mean, val_mean = [], [], []
    metrics = os.path.join(run_dir, 'epoch_metrics.csv')
    if os.path.exists(metrics):
        with open(metrics, newline='') as fh:
            for row in csv.DictReader(fh):
                epochs.append(int(row['epoch']))
                train_mean.append(float(row['train_loss']))
                val_mean.append(float(row['val_loss']))
    return steps, train, epochs, train_mean, val_mean


def _pick_or_merge_event_files(paths):
    """一个 run 目录下可能有多个事件文件。

    步数区间不重叠 => 是同一次训练的续跑，直接拼起来；
    区间重叠     => 是几次互相独立的运行，取点数最多的那个（通常是跑完整的那次）。
    """
    loaded = []
    for path in paths:
        series = read_tfevents(path)
        points = series.get('train_loss', [])
        if points:
            loaded.append((path, points))
    if not loaded:
        return None, None

    loaded.sort(key=lambda item: item[1][0][0])   # 按起始步数排序
    merged, prev_end = [], None
    for path, points in loaded:
        if prev_end is not None and points[0][0] <= prev_end:
            merged = None                         # 区间重叠 -> 放弃合并
            break
        merged = (merged or []) + points
        prev_end = points[-1][0]

    if merged is not None:
        return merged, [p for p, _ in loaded]

    path, points = max(loaded, key=lambda item: len(item[1]))
    print(f'[warn] 发现多个互相重叠的事件文件，只用点数最多的: {os.path.basename(path)}')
    for other, pts in loaded:
        if other != path:
            print(f'       已忽略 {os.path.basename(other)} ({len(pts)} 点)')
    return points, [path]


def load_from_tfevents(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, 'events.out.tfevents.*')))
    if not paths:
        return None
    points, used = _pick_or_merge_event_files(paths)
    if not points:
        print(f'[warn] {run_dir} 下的事件文件里没有 train_loss')
        return None
    for path in used:
        print(f'[info] 读取事件文件 {os.path.basename(path)}')
    return points


# --------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------
def ema(values, weight):
    """指数滑动平均，weight 越接近 1 越平滑。"""
    out, last = [], None
    for value in values:
        last = value if last is None else last * weight + value * (1 - weight)
        out.append(last)
    return out


def make_plot(steps, train, epochs, train_mean, val_mean, args):
    theme = THEME['dark' if args.dark else 'light']
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=args.dpi)
    fig.patch.set_facecolor(theme['surface'])
    ax.set_facecolor(theme['surface'])

    # 逐 step 的原始 loss：同一个实体的更细粒度视图，用同色低透明度画，不占独立系列
    ax.plot(steps, train, color=theme['train'], linewidth=0.8, alpha=0.28, zorder=2)

    # 平滑后的训练曲线
    smoothed = ema(train, args.smooth)
    ax.plot(steps, smoothed, color=theme['train'], linewidth=2.0,
            label='Train loss (smoothed)', zorder=3)

    # 每轮平均值：有的话用点画出来，比平滑曲线更能说明趋势
    if epochs:
        mean_x = [steps[-1] * (e / max(epochs)) for e in epochs] if steps else epochs
        ax.plot(mean_x, train_mean, color=theme['train'], linewidth=0,
                marker='o', markersize=5, markeredgecolor=theme['surface'],
                markeredgewidth=1.4, label='Train loss (epoch mean)', zorder=4)
        if any(v == v for v in val_mean):
            ax.plot(mean_x, val_mean, color=theme['val'], linewidth=2.0,
                    marker='o', markersize=5, markeredgecolor=theme['surface'],
                    markeredgewidth=1.4, label='Validation loss', zorder=5)

    # 直接标注末值，省得读者去对刻度
    end_labels = []
    if steps:
        end_labels.append([smoothed[-1], steps[-1], f'{smoothed[-1]:.3f}'])
    if epochs and any(v == v for v in val_mean):
        end_labels.append([val_mean[-1], mean_x[-1], f'{val_mean[-1]:.3f}'])

    # 训练后期两条线末值常常挨得很近，两个数字会叠在一起，按 y 值上下错开
    offsets = [0] * len(end_labels)
    y_lo, y_hi = ax.get_ylim()
    if len(end_labels) == 2:
        end_labels.sort(key=lambda item: item[0])
        if (y_hi - y_lo) > 0 and abs(end_labels[0][0] - end_labels[1][0]) < 0.05 * (y_hi - y_lo):
            offsets = [-9, 9]

    for (value, x_pos, label), dy in zip(end_labels, offsets):
        ax.annotate(label, (x_pos, value), textcoords='offset points', xytext=(8, dy),
                    color=theme['ink2'], fontsize=10, va='center')

    ax.set_xlabel('Training step', color=theme['muted'], fontsize=11)
    ax.set_ylabel('Cross-entropy loss (nats)', color=theme['muted'], fontsize=11)
    ax.set_title(args.title or 'Transformer en→it training loss',
                 color=theme['ink'], fontsize=14, pad=14, loc='left')

    ax.grid(True, color=theme['grid'], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(theme['axis'])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme['muted'], labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    # 系列 ≥2 才上图例；只有一条训练曲线时，标题已经说明它是什么了
    if len(ax.get_legend_handles_labels()[0]) >= 2:
        legend = ax.legend(frameon=False, fontsize=10, loc='upper right')
        for text in legend.get_texts():
            text.set_color(theme['ink2'])

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='画 Transformer 训练损失曲线')
    parser.add_argument('--run', default='runs/tmodel',
                        help='实验目录，默认 runs/tmodel（也接受直接给 .csv 或 tfevents 文件）')
    parser.add_argument('--out', default=None, help='输出 PNG 路径，默认 <run>/loss_curve.png')
    parser.add_argument('--smooth', type=float, default=0.9,
                        help='EMA 平滑系数 0~1，默认 0.9；给 0 就是不平滑')
    parser.add_argument('--dark', action='store_true', help='出深色版')
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--title', default=None)
    args = parser.parse_args()

    run_dir = args.run
    if os.path.isfile(run_dir):
        run_dir = os.path.dirname(run_dir) or '.'
    if not os.path.isdir(run_dir):
        print(f'[error] 目录不存在: {run_dir}', file=sys.stderr)
        return 1

    steps, train, epochs, train_mean, val_mean = load_from_csv(run_dir)

    if not steps:
        points = load_from_tfevents(run_dir)
        if points is None:
            print(f'[error] {run_dir} 下既没有 loss_history.csv 也没有事件文件。'
                  f'先跑一次 train.py 吧。', file=sys.stderr)
            return 1
        steps = [p[0] for p in points]
        train = [p[1] for p in points]

    print(f'[info] {len(steps)} 个训练点，loss {train[0]:.4f} -> {train[-1]:.4f}')
    if epochs:
        print(f'[info] {len(epochs)} 轮，val_loss 末值 {val_mean[-1]:.4f}')

    out = args.out or os.path.join(run_dir, 'loss_curve_dark.png' if args.dark else 'loss_curve.png')
    fig = make_plot(steps, train, epochs, train_mean, val_mean, args)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[ok] 已保存 {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

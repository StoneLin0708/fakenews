import numpy as np
import re
import os


def markdown_pred_summary(x):
    return '|input|output|target|\n|-|-|-|\n' + '\n'.join(
        map(lambda i: f'|{i[0]}|{i[1]}|{i[2]}|', x)
    )


def peek(data, n, seed):
    return [data[i] for i in np.random.default_rng(seed=seed).choice(len(data), n)]


def find_latest_ckpt(folder, pattern):
    pt = sorted(list(filter(
        lambda x: x is not None,
        [re.match(pattern, i) for i in os.listdir(folder)])
    ), key=lambda x: int(x.group(1)))[-1].group(0)
    return os.path.join(folder, pt)

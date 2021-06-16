import numpy as np
import re
import os
from collections import namedtuple


def markdown_pred_summary(x):
    x = re.sub(r'<', r'&lt', x)
    x = re.sub(r'>', r'&gt', x)
    return '|input|output|target|\n|-|-|-|\n' + '\n'.join(
        map(lambda i: f'|{i[0]}|{i[1]}|{i[2]}|', x)
    )


def peek(data, n, seed):
    return [data[i] for i in np.random.default_rng(seed=seed).choice(len(data), n)]


def find_latest_ckpt(folder, pattern):
    d = os.listdir(folder)
    pt = sorted(list(filter(
        lambda x: x is not None,
        [re.match(pattern, i) for i in d])
    ), key=lambda x: int(x.group(1)))
    if len(pt) == 0:
        return None
    pt = pt[-1]
    return namedtuple('ckpt', ['step', 'path'])(int(pt.group(1)), os.path.join(folder, pt.group(0)))

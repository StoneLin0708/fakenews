# In[]
from tqdm import tqdm
from src.dataset import NewsDataset
from argparse import ArgumentParser
import numpy as np
import os


def getargs():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--sample', type=float, nargs='+', default=None)
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()


def gen(datasets, samples):
    for ds, sample in zip(datasets, samples):
        for idx in np.random.choice(len(ds), int(len(ds)*sample)):
            i = ds.data[idx]
            a = '\n'.join(
                map(lambda x: i[2][x:x+4096], range(0, len(i[2]), 4096)))
            a = f'{i[1]}\n{a}\n'
            if max(map(len, a.split('\n'))) > 4096:
                print(max(map(len, a.split('\n'))))
            yield a


if __name__ == '__main__':
    args = getargs()
    if args.sample is None:
        args.sample = [1.] * len(args.data)
    if os.path.exists(args.out):
        print(f'corpus {args.out} exists')
        exit()
    ds = list(map(NewsDataset, args.data))
    open(args.out, 'w').writelines(
        tqdm(gen(ds, args.sample), total=sum(
            map(lambda x: int(x[0]*x[1]), zip(map(len, ds), args.sample))))
    )

# print(max(map(len, open('data/corpus.txt').readlines())))

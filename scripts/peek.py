import argparse
import numpy as np
from src.dataset import NewsDataset
from src.utils import peek

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-n', type=int, default=1)
    parser.add_argument('-a', type=float, default=0)
    parser.add_argument('-b', type=int, default=64)
    return parser.parse_args()

if __name__ == '__main__':
    args = getargs()
    ds = NewsDataset(args.dataset, args.a, args.b)
    for idx, x, y in peek(ds.data, args.n):
        print(f'[{idx}]')
        print('-'*20+'x'+'-'*20)
        print(x)
        print('-'*20+'y'+'-'*20)
        print(y)
        print('='*41)
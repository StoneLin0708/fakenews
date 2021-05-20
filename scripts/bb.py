import argparse
import random
import timeit

import numpy as np
from tqdm import tqdm
import torch

from src.decode import beam_search_v2
from src.dataset import Arithmetic
from src.transformer_model import TransformerModel
from src.tokenizer import Tokenizer



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/arithmetic.csv')
    parser.add_argument('--tokenizer',default='data/atk', type=str)
    parser.add_argument('--seed', default=8787, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--ckpt', default='test/backward.pt', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    return parser.parse_args()


def main(args):
    set_seed(args.seed)

    tk = Tokenizer(args.tokenizer)

    ds = Arithmetic(args.data)

    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ds.get_collate_fn(tk)
    )

    model = TransformerModel(
        d_model=32,
        d_ff=64,
        dropout=.0,
        layers=3,
        heads=4,
        d_emb=-1,
        pad_token_id=tk.pad_id,
        vocab_size=tk.vocab_size
    )

    device = torch.device(args.device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
    model.to(device)

    start = timeit.default_timer()
    total = 0
    tp = 0
    for x, y in tqdm(dl):
        p = beam_search_v2(model, x, tk,
        lambda b, nx, ny: (nx + ny) * b > 4096 * 6 * 64, 1, device, 10)
        r = list(map(lambda i:i[0]==i[1], zip(p, y)))
        total += len(r)
        tp += np.count_nonzero(r)


    print((timeit.default_timer()-start))
    print(tp/total)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    main(parse_args())

import argparse
import random
import timeit

import numpy as np
import torch

from src.decode import beam_search_v2
from src.dataset import NewsDataset
from src.transformer_model import TransformerModel
from src.tokenizer import Tokenizer
from src.utils import peek, find_latest_ckpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inseq', type=str, nargs='+')
    parser.add_argument('--data', type=str, default='data/news_dataset.db')
    parser.add_argument('-a', type=float, default=0)
    parser.add_argument('-b', type=int, default=64)

    parser.add_argument('--peek', type=int, default=1)
    parser.add_argument('--hide_tgt', action='store_true')
    parser.add_argument('--markdown', action='store_true')
    parser.add_argument('--tokenizer', default='data/tk', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--ckpt', default='latest', type=str)
    parser.add_argument('--ckpt_pattern', default=r'(\d+).pt', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    return parser.parse_args()


def main(args):
    set_seed(args.seed) if args.seed is not None else None

    tk = Tokenizer(args.tokenizer)

    model = TransformerModel(
        d_model=768,
        d_ff=1024,
        dropout=.1,
        layers=5,
        heads=8,
        d_emb=-1,
        pad_token_id=tk.pad_id,
        vocab_size=tk.vocab_size
    )

    if args.inseq is not None:
        r = beam_search_v2(model, tk.tokenize(args.inseq), tk,
                       lambda b, nx, ny: (nx + ny) * b > 128 * 64, 4, args.device, 64)
    else:
        if args.peek == 0:
            return
        ds = NewsDataset(args.data,args.a,args.b)

        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=args.peek,
            shuffle=True,
            collate_fn=ds.get_collate_fn(tk),
        )

        device = torch.device(args.device)

        if args.ckpt == 'latest':
            args.ckpt = find_latest_ckpt(args.model_dir, args.ckpt_pattern)
        model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
        model.to(device)
        data = next(iter(dl))
        start = timeit.default_timer()
        pred = beam_search_v2(model, data[0], tk,
                       lambda b, nx, ny: (nx + ny) * b > 128 * 64, 4, args.device, 64)
        for t, a, p in zip(data[0], data[1], pred):
            if args.markdown:
                print(f'|{tk.detokenize(t)}|',end='')
            else:
                print(tk.detokenize(t))
            if not args.hide_tgt:
                print('-' * 25 + 'text' + '-' * 25)
                print(tk.detokenize(a))
            if args.markdown:
                print(f'{tk.detokenize(p)}|')
            else:
                print('-' * 25 + 'pred' + '-' * 25)
                print(tk.detokenize(p))
                print('=' * 50)

        print((timeit.default_timer()-start))


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

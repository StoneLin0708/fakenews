import argparse
import random
import timeit

import numpy as np
import torch
import pandas as pd

from src.decode import beam_search_v2, topk
from src.dataset import NewsDataset
from src.transformer_model import TransformerModel
from src.tokenizer import Tokenizer
from src.utils import peek, find_latest_ckpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inseq', type=str, nargs='+')
    parser.add_argument('--aids', type=int, nargs='+')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('-a', type=float, default=0)
    parser.add_argument('-b', type=int, default=64)

    parser.add_argument('--layer', default=5, type=int)
    parser.add_argument('--heads', default=12, type=int)
    parser.add_argument('--peek', type=int, default=1)
    parser.add_argument('--tokenizer', required=True, type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--maxlen', type=int, default=64)
    parser.add_argument('--inplace', action='store_true', default=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--sample', default=-1, type=int)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--topk', type=int, nargs='+', default=[])
    parser.add_argument('--beam', type=int, nargs='+', default=[])
    parser.add_argument('--ckpt', default='latest', type=str)
    parser.add_argument('--ckpt_pattern', default=r'(\d+).pt', type=str)
    parser.add_argument('--device', default='cpu', type=str)

    parser.add_argument('--output', type=str)
    return parser.parse_args()


def main(args):
    set_seed(args.seed) if args.seed is not None else None

    tk = Tokenizer(args.tokenizer)

    model = TransformerModel(
        d_model=768,
        d_ff=1024,
        dropout=0,
        layers=args.layer,
        heads=args.heads,
        d_emb=-1,
        pad_token_id=tk.pad_id,
        vocab_size=tk.vocab_size
    )

    if args.inseq is not None:
        r = beam_search_v2(model, tk.tokenize(args.inseq), tk,
                           lambda b, nx, ny: (nx + ny) * b > 128 * 64, 4, args.device, args.maxlen)
    else:
        if args.peek == 0:
            return
        ds = NewsDataset(args.data, args.a, args.b,
                         inplace=args.inplace, sample=args.sample, seed=args.seed)

        device = torch.device(args.device)

        if args.ckpt == 'latest':
            args.ckpt = find_latest_ckpt(args.model_dir, args.ckpt_pattern)
        model.load_state_dict(torch.load(
            args.ckpt, map_location=device)['model'])
        model.to(device)
        if len(args.aids) == 0:
            ids, inseq, outseq = ds.get_collate_fn(
                tk, getid=True)(peek(ds, args.peek, args.seed))
        else:
            data = list(filter(lambda x: x[0] in args.aids, ds.data))
            sdata = []
            if len(data) != len(args.aids):
                raise Exception(f'only got {list(zip(*data))[0]}')
            for i in args.aids:
                for x in data:
                    if x[0] == i:
                        sdata.append(x)
            ids, inseq, outseq = ds.get_collate_fn(
                tk, getid=True)(sdata)

        start = timeit.default_timer()
        preds = []
        for beam_n in args.beam:
            p = beam_search_v2(model, inseq, tk,
                               lambda b, nx, ny: (nx + ny) * b > 128 * 64,
                               beam_n, args.device, args.maxlen)
            preds.append((f'beam{beam_n}', p))
        for topk_k in args.topk:
            p = topk(model, outseq, tk, topk_k, args.device, args.maxlen)
            preds.append((f'topk{topk_k}', p))

        results = []
        for idx in range(len(inseq)):
            results.append((
                tk.detokenize(inseq[idx]),
                *[tk.detokenize(p[idx]) for _, p in preds],
                tk.detokenize(outseq[idx][:args.maxlen+1])
            ))
        df = pd.DataFrame(results,
                          columns=['input', *[n for n, _ in preds], 'target'])
        if args.output is None:
            print(df)
        else:
            df.to_csv(args.output)
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

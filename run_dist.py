import argparse
import random
import timeit
import pickle

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from src.decode import beam_search_v2, topk
from src.dataset import NewsDataset
from src.transformer_model import TransformerModel
from src.tokenizer import Tokenizer
from src.utils import peek, find_latest_ckpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)

    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--sample', default=-1, type=int)
    parser.add_argument('-a', type=float, default=0)
    parser.add_argument('-b', type=int, default=64)
    parser.add_argument('--inplace', action='store_true', default=False)

    parser.add_argument('--tokenizer', required=True, type=str)

    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--ckpt', default='latest', type=str)
    parser.add_argument('--ckpt_pattern', default=r'(\d+).pt', type=str)

    parser.add_argument('--layer', default=5, type=int)
    parser.add_argument('--heads', default=12, type=int)

    parser.add_argument('--batch_size', default=64, type=int)
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

    ds = NewsDataset(args.data, args.a, args.b,
                     inplace=args.inplace, sample=args.sample, seed=args.seed)

    device = torch.device(args.device)

    @torch.no_grad()
    def find_dist(model):
        model.eval()
        probs = [[] for i in range(tk.vocab_size)]
        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=ds.get_collate_fn(tk),
            num_workers=2,
        )

        def pad(inseq, max_len):
            return torch.nn.utils.rnn.pad_sequence(
                list(map(lambda x: torch.LongTensor(x[:max_len]), inseq)),
                padding_value=0,
                batch_first=True).to(device)
        tkcount = 0
        for x, y in tqdm(dl, smoothing=0):
            tkcount += sum([len(i) for i in y])
            x = pad(x, 512)
            y = pad(y, 513)

            # B S V
            p = torch.nn.functional.softmax(model(x, y[:, :-1]), dim=-1)
            # B S
            r = torch.gather(
                p,
                -1,
                y[:, 1:].unsqueeze(-1)
            ).squeeze(-1).cpu()

            for vids, ps in zip(y[:, 1:], r):
                for vid, p in zip(vids, ps):
                    probs[vid].append(float(p))
        return probs

    if args.ckpt == 'latest':
        args.ckpt = find_latest_ckpt(args.model_dir, args.ckpt_pattern)
    model.load_state_dict(torch.load(
        args.ckpt, map_location=device)['model'])
    model.to(device)
    results = find_dist(model)
    pickle.dump(results, open(args.output, 'wb'))


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

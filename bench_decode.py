import argparse
import random
import timeit
import os

import numpy as np
import torch
from torch.nn.modules import Transformer

from src.dataset import dataset
from src.tokenizer import Tokenizer
from src.transformer_model import TransformerModel
from src.train import train
from src.decode import beam_search, beam_search_v2, beam_search_v3


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--results', default='test/decoded.txt', type=str)
    # parser.add_argument('--m_ratio', default=0.001, type=float)
    parser.add_argument('--results', default='test/dec_m.txt', type=str)
    parser.add_argument('--m_ratio', default=0.00002, type=float)
    parser.add_argument('--tokenizer', default='model/tk', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=8787, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--ckpt', default='test/backward.pt', type=str)
    parser.add_argument('--beam', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=64)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)

    tk = Tokenizer(args.tokenizer)

    model = TransformerModel(
        d_model=768,
        d_ff=1024,
        dropout=.1,
        layers=6,
        heads=8,
        d_emb=-1,
        pad_token_id=tk.pad_id,
        vocab_size=tk.vocab_size
    )

    ds = dataset(0)

    device = torch.device(args.device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])

    ds.set_mono_ratio(args.m_ratio)
    if not os.path.exists(args.results):
        start = timeit.default_timer()
        ds.generate(lambda x: [beam_search(
            model=model.to(device),
            input_sequence=torch.LongTensor(tk.tokenize(x)).to(device),
            bos_id=tk.bos_id,
            eos_id=tk.eos_id,
            beam_width=args.beam,
            device=device,
            max_seq_len=64)],
            max_input_len=64)
        end = timeit.default_timer()
        print(f'{end-start:.2f} sec')
        open(args.results,'w').writelines('\n'.join(tk.detokenize(ds.synthetic[1:])))
    else:
        start = timeit.default_timer()
        ds.generate(lambda x: beam_search_v2(
            model=model.to(device),
            input_sequence=tk.tokenize(x),
            tokenizer=tk,
            is_full=lambda b, nx, ny: (nx + ny * 1.5) * b > 256 * 64,
            beam_width=args.beam,
            device=device,
            max_seq_len=64),
            max_input_len=64,
            batch_size=64)
        end = timeit.default_timer()

        s = tk.detokenize(ds.synthetic[1:])
        open(args.results+'_2', 'w').writelines('\n'.join(s))
        r = open(args.results).readlines()
        if len(s) != len(r):
            raise Exception(f'result should be length of {len(r)} but got {len(s)}')
        for i, j in zip(s, r):
            if i != j.strip():
                print(f'---------------\n"{i}"\n!=\n"{j.strip()}"')

        print(f'{end-start:.2f} sec')


if __name__ == '__main__':
    main(parse_args())

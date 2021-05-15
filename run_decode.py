import argparse
import random
import timeit

import numpy as np
import torch

from src.decode import beam_search_v2
from src.dataset import NewsDataset
from src.transformer_model import TransformerModel
from src.tokenizer import Tokenizer



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inseq', default='瞞騙父母種水耕蔬菜 他3年將自家冰庫打造成2千萬大麻花園',type=str)

    parser.add_argument('--data', type=str, default='data/news_dataset.db')
    parser.add_argument('--tokenizer',default='data/tk', type=str)
    parser.add_argument('--seed', default=8787, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--ckpt', default='', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    return parser.parse_args()


def main(args):
    set_seed(args.seed)

    tk = Tokenizer(args.tokenizer)

    ds = NewsDataset(args.data)

    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ds.get_collate_fn(tk)
    )

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

    device = torch.device(args.device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
    model.to(device)
    start = timeit.default_timer()
    # r = beam_search(model, torch.LongTensor(tk.tokenize([args.inseq])).to(
    r = beam_search_v2(model, tk.tokenize([args.inseq]), tk,
        lambda b, nx, ny: (nx + ny) * b > 128 * 64, 4, device, 64)
    print((timeit.default_timer()-start))
    print(tk.detokenize(r))


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

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules import Transformer

from src.dataset import NewsDataset
from src.tokenizer import Tokenizer
from src.transformer_model import TransformerModel
from src.train import train
from src.decode import beam_search


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/news_dataset_clean.db')
    parser.add_argument('--tokenizer', type=str, default='data/tk')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--seed', default=8787, type=int)
    parser.add_argument('--save_epoch', default=1, type=float)
    parser.add_argument('--summary_step', default=50, type=int)
    parser.add_argument('--batch_size', default=14, type=int)
    parser.add_argument('--model_dir', default='model/0', type=str)
    parser.add_argument('--ckpt', type=str)
    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

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

    ds = NewsDataset(args.data)

    print(
        f'model size = {sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024:.2f} M trainable parameters')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(
        model=model,
        dataset=ds,
        batch_size=args.batch_size,
        device=device,
        tokenizer=tk,
        epochs=args.epochs,
        model_dir=args.model_dir,
        save_epoch=args.save_epoch,
        summary_step=args.summary_step)


if __name__ == '__main__':
    main(parse_args())

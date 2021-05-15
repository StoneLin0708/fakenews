import sentencepiece as sp
import argparse
import os


class Tokenizer:
    def __init__(self, model_path):
        self.model = sp.SentencePieceProcessor()
        self.vocab_size = 0
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        if model_path != None and os.path.exists(f'{model_path}.model'):
            self.load(model_path)

    def train(self, data_path, max_vocab,
              model_path, coverage, max_len):
        sp.SentencePieceTrainer.Train(
            f'--input={data_path} '
            f'--model_prefix={model_path} '
            f'--vocab_size={max_vocab} '
            f'--pad_id={self.pad_id} '
            f'--bos_id={self.bos_id} '
            f'--eos_id={self.eos_id} '
            f'--unk_id={self.unk_id} '
            f'--max_sentence_length={20480} '
            f'--character_coverage={coverage} '
            f'--max_sentencepiece_length={max_len} '
        )

    def load(self, model_path):
        self.model.Load(f'{model_path}.model')
        self.vocab_size = self.model.GetPieceSize()
        print(f'loaded {self.vocab_size}')

    def is_loaded(self):
        return self.model.GetPieceSize() > 0

    def tokenize(self, sentence, bos=True, eos=True):
        return self.model.Encode(sentence, add_bos=bos, add_eos=eos)

    def detokenize(self, token):
        return self.model.Decode(token)


def main(args):
    tk = Tokenizer(args.o)
    if not tk.is_loaded():
        tk.train(
            model_path=args.o,
            data_path=args.d,
            max_vocab=args.vocab,
            coverage=args.coverage,
            max_len=args.max_len
        )
    else:
        token = tk.tokenize('今天不想上班')
        print(token)
        print(tk.detokenize(token))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str)
    parser.add_argument('-o', type=str)
    parser.add_argument('--vocab', type=int)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--coverage', type=str)
    args = parser.parse_args()
    main(args)

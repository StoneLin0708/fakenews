import sqlite3
import torch
import pandas as pd
import numpy as np


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, db_path: str, a=0, b=0, inplace=False, sample=None, seed=87):
        super().__init__()

        # Connect to DB.
        conn = sqlite3.connect(db_path)

        # Get database cursor.
        cursor = conn.cursor()

        # Get all news title and article.
        cursor.execute("SELECT id, title, article FROM news_dataset;")
        data = cursor.fetchall()
        if sample is not None and sample > 0:
            N = len(data)
            idx = np.random.default_rng(seed=seed).choice(
                N, sample, replace=False)
            data = [data[i] for i in idx]
        self.data = data
        self.a = a
        self.b = b
        self.inplace = inplace

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_collate_fn(self, tk):
        def collect_fn(seq):
            _, x, y = zip(*seq)
            x, y = tk.tokenize(list(x)), tk.tokenize(list(y))
            if self.a > 0:
                idxs = np.random.choice(len(x), int(
                    len(x) * self.a), replace=False)
                if self.inplace:
                    for idx in idxs:
                        x[idx] += y[idx][:self.b]
                        y[idx] = y[idx][self.b:]
                else:
                    for idx in idxs:
                        x.append(x[idx] + y[idx][:self.b])
                        y.append(y[idx][self.b:])
            return x, y
        return collect_fn


class Arithmetic(torch.utils.data.Dataset):
    def __init__(self, datapath, srcfilter=lambda x: x):
        super().__init__()
        d = pd.read_csv(datapath, dtype=(str, str)).values.tolist()
        self.data = list(filter(lambda x: srcfilter(x[0]), d))

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_collate_fn(tk):
        def collect_fn(seq):
            x, y = zip(*seq)
            return tk.tokenize(list(x)), tk.tokenize(list(y))
        return collect_fn


if __name__ == "__main__":
    # Initial example.
    dataset = NewsDataset(db_path='news.db')

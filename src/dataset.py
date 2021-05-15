import sqlite3
import torch
import pandas as pd


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, db_path: str):
        super().__init__()

        # Connect to DB.
        conn = sqlite3.connect(db_path)

        # Get database cursor.
        cursor = conn.cursor()

        # Get all news title and article.
        cursor.execute("SELECT id, title, article FROM news_dataset;")
        self.data = cursor.fetchall()

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_collate_fn(tk):
        def collect_fn(seq):
            _, x, y  = zip(*seq)
            return tk.tokenize(list(x)), tk.tokenize(list(y))
        return collect_fn

class Arithmetic(torch.utils.data.Dataset):
    def __init__(self, datapath, srcfilter=lambda x:x):
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
            x, y  = zip(*seq)
            return tk.tokenize(list(x)), tk.tokenize(list(y))
        return collect_fn

if __name__ == "__main__":
    # Initial example.
    dataset = NewsDataset(db_path='news.db')

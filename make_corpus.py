# In[]
from tqdm import tqdm
from src.dataset import NewsDataset


def gen(datasets):
    for ds in datasets:
        for i in ds.data:
            a = '\n'.join(
                map(lambda x: i[2][x:x+4096], range(0, len(i[2]), 4096)))
            a = f'{i[1]}\n{a}\n'
            if max(map(len, a.split('\n'))) > 4096:
                print(max(map(len, a.split('\n'))))
            yield a


open('data/corpus.txt', 'w').writelines(
    tqdm(gen([
        NewsDataset('data/news_dataset_clean.db'),
        NewsDataset('data/wiki.db'),
    ]))
)

# print(max(map(len, open('data/corpus.txt').readlines())))

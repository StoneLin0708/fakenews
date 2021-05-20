# In[]
import unicodedata
import re
from src.dataset import NewsDataset
import sqlite3
import os
import numpy as np
import argparse

original = 'data/news_v1.4.1.db'
dataset = 'data/news_dataset_v1.4.1.db'
cleandataset = 'data/news_dataset_clean_200_v1.4.1.db'


def extract(source, target):
    con = sqlite3.connect(source)
    c = con.cursor()
    c.execute("SELECT id, title, article FROM news_table")

    dataset = sqlite3.connect(target)
    dc = dataset.cursor()
    dc.execute("""
    CREATE TABLE IF NOT EXISTS news_dataset (
        id interger PRIMARY KEY,
        title text NOT NULL,
        article text NOT NULL);
    """)

    N = 0
    while True:
        i = c.fetchmany(1000)
        N += len(i)
        dc.executemany(
            'INSERT INTO news_dataset (id, title, article) VALUES (?,?,?);',
            i)
        dataset.commit()
        if len(i) < 1000:
            break
    print(N)


if not os.path.exists(dataset):
    extract(original, dataset)
else:
    print('dataset exist')
# In[]
ds = NewsDataset(dataset)
N = len(ds.data)

t_len = list(map(lambda x: len(x[1]), ds.data))
a_len = list(map(lambda x: len(x[2]), ds.data))


def analysis(d):
    s = np.array(sorted(d))
    return (np.average(s),
            np.std(s),
            s[0],
            s[int(len(s)*0.25)],
            s[int(len(s)*0.5)],
            s[int(len(s)*0.75)],
            s[-1])


print(analysis(t_len))
print(analysis(a_len))


# In[] peek dataset
peek = [ds.data[i] for i in np.random.choice(N, 1)]
for idx, title, article in peek:
    print(f'[{idx}] : {title}')
    print(article)

# In[] find out prefix
prefix = {}
for idx, title, article in ds.data:
    p = article.strip()[:11]
    if p in prefix:
        prefix[p] += 1
    else:
        prefix[p] = 1

sp = sorted(prefix.items(), key=lambda x: x[1], reverse=True)
print('\n'.join(map(lambda x: f'|{x[0]}|{x[1]}|', sp[:25])))
# In[] clean up test
ad = 0
for idx, title, article in ds.data:
    if '▪整理包/110國中教育會考衝刺大補帖 各科必考重點一次看 ▪活動/讀書累了嗎?30秒偷看漫畫舒壓一下 看這間學校有多鬧!' in article:
        ad += 1
print(ad)

url = re.compile(r'https?://[a-zA-Z0-9/\?\=\-\.]+')
torm = [
    '▪整理包/110國中教育會考衝刺大補帖 各科必考重點一次看 ▪活動/讀書累了嗎?30秒偷看漫畫舒壓一下 看這間學校有多鬧!',
]
rmurl = 0
rmrule = 0
for idx, title, article in ds.data:
    nurl = len(url.findall(article))
    if nurl > 0:
        article = url.sub('', article)
        rmurl += nurl
    for i in torm:
        if i in article:
            rmrule += 1
        article = article.replace(i, '')

print(rmurl)
print(rmrule)


def cleanup(article):
    article = url.sub('', article)
    for i in torm:
        article = article.replace(i, '')
    return article


# In[]

def makeclean(cleandataset):
    dataset = sqlite3.connect(cleandataset)
    dc = dataset.cursor()

    dc.execute("""
    CREATE TABLE IF NOT EXISTS news_dataset (
        id interger PRIMARY KEY,
        title text NOT NULL,
        article text NOT NULL);
    """)

    dc.executemany(
        'INSERT INTO news_dataset (id, title, article) VALUES (?,?,?);',
        filter(lambda i: len(i[2]) >= 200,
               map(lambda x: (x[0], x[1], cleanup(x[2])),
                   ds.data)
               ))

    dataset.commit()


if not os.path.exists(cleandataset):
    makeclean(cleandataset)
    print('clean')
else:
    print('clean exist')

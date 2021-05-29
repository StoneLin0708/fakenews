# In[]
import unicodedata
import re
from src.dataset import NewsDataset
import sqlite3
import os
import numpy as np
import argparse
from tqdm import tqdm

original = 'data/news_v2.1.db'
dataset = 'data/news_dataset_v2.1.db'
cleantagdataset = 'data/news_dataset_tag10_v2.1.db'
cleandataset = 'data/news_dataset_clean_200_v1.6.2.db'


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
            filter(lambda x:len(x[1])>0, i))
        dataset.commit()
        if len(i) < 1000:
            break
    print(N)


if not os.path.exists(dataset):
    extract(original, dataset)
else:
    print('dataset exist')

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
            s[int(len(s)*0.95)],
            s[-1])


print(analysis(t_len))
print(analysis(a_len))
# In[]
for i, t, a in ds.data:
    if len(t) < 3:
        print(i)
        print(len(t))
        print(t)
        break

# In[]
for _, t, a in ds.data:
    if '<en>>' in a:
        print(a)
        break
# In[]
tagreg = re.compile(r'(<...?([0-9]+)>)')

alltags = [tagreg.findall(t + a) for idx, (_id, t, a)
           in tqdm(enumerate(ds.data), total=len(ds.data))]

alltaglist = [j for i in filter(lambda x:len(x) > 0, alltags) for j in i]
# In[]


def unique(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


# alltagset = unique(alltaglist)
# print(set(alltaglist))
print(set(map(lambda x: x[0][:5], alltaglist)))
# In[]


def replace_tag(t, a, tagseq, newtagpat, limitpat, limit=10):
    for i, n in enumerate(tagseq):
        tmptag = f'<tmptag{i}>'
        if i < limit:
            t = re.sub(n, tmptag, t)
            a = re.sub(n, tmptag, a)
        else:
            t = re.sub(n, limitpat, t)
            a = re.sub(n, limitpat, a)

    t = re.sub(r'<tmptag([0-9]+)>', newtagpat, t)
    a = re.sub(r'<tmptag([0-9]+)>', newtagpat, a)
    return t, a


def ordertag():
    for idx, (_id, t, a) in tqdm(enumerate(ds.data), total=len(ds.data)):
        tags = tagreg.findall(t + a)
        tags = [(n, int(c) if len(c) > 0 else None) for n, c in tags]
        if len(tags) == 0:
            continue

        for tagpat, rep, lmpt in [
            ('<org', r'<org\1>', '<org>'),
            ('<loc', r'<loc\1>', '<loc>'),
            ('<per', r'<per\1>', '<per>'),
        ]:
            cur_tags = unique(
                list(filter(lambda x: x[0].startswith(tagpat), tags)))
            t, a = replace_tag(t, a, [n for n, _ in cur_tags], rep, lmpt)

        ds.data[idx] = (_id, t, a)


ordertag()

# In[]
if True:
    def test2():
        for idx, (_id, t, a) in tqdm(enumerate(ds.data), total=len(ds.data)):
            tags = tagreg.findall(t + a)
            tags = [(n, int(c) if len(c) > 0 else None) for n, c in tags]
            if len(tags) == 0:
                continue
            for tagpat, rep in [
                ('<org', r'<org\1>'),
                ('<loc', r'<loc\1>'),
                ('<per', r'<per\1>'),
            ]:
                cur_tags = unique(
                    list(filter(lambda x: x[0].startswith(tagpat), tags)))
                for i, (n, c) in enumerate(cur_tags):
                    if i != c:
                        print('error')
                        return
    test2()

# In[]
ll = []
for idx, (_id, t, a) in tqdm(enumerate(ds.data), total=len(ds.data)):
    tags = tagreg.findall(t + a)
    tags = [(n, int(c) if len(c) > 0 else None) for n, c in tags]
    if len(tags) == 0:
        continue
    l = [len(unique(list(filter(lambda x: x[0].startswith(tagpat), tags))))
         for tagpat in ['<org', '<loc', '<per']]
    ll.append(l)
lorg, lloc, lper = zip(*ll)

# In[]
print(analysis(lorg))
print(analysis(lloc))
print(analysis(lper))

# In[]


def makecleantag(cleantagdataset):
    dataset = sqlite3.connect(cleantagdataset)
    dc = dataset.cursor()

    dc.execute("""
    CREATE TABLE IF NOT EXISTS news_dataset (
        id interger PRIMARY KEY,
        title text NOT NULL,
        article text NOT NULL);
    """)

    dc.executemany(
        'INSERT INTO news_dataset (id, title, article) VALUES (?,?,?);',
        filter(lambda i: len(i[2]) >= 200, ds.data))

    dataset.commit()


if not os.path.exists(cleantagdataset):
    makecleantag(cleantagdataset)
    print('clean tag')
else:
    print('clean tag exist')


# print(sorted(t_orgs,key=lambda x:x[1]))
# In[] peek dataset
peek = [ds.data[i] for i in np.random.choice(N, 1)]
for idx, title, article in peek:
    print(f'[{idx}] : {title}')
    print(article)

# In[] find out prefix
prefix = {}
for idx, title, article in ds.data:
    p = article.strip()[:30]
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

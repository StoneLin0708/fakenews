# In[]
import sqlite3
import xml.etree.ElementTree as et
import numpy as np
import os
import json

# https://dumps.wikimedia.org/zhwiki/latest/

# In[]
if False:
    data = []
    for i in range(1, 7):
        fn = f'../data/zhwiki-latest-abstract-zh-tw{i}.xml'
        elems = []
        for idx, (e, d) in enumerate(et.iterparse(fn, events=['start', 'end'])):
            if e == 'start':
                elems.append(d)
            elif e == 'end':
                if d in elems and d.tag == 'doc':
                    t = d.findall('title')[0].text
                    a = d.findall('abstract')[0].text
                    if t == None or a == None:
                        continue
                    data.append((t, a))
                elems.pop()

    t = np.array(list(map(lambda x: len(x[0]), data)))
    a = np.array(list(map(lambda x: len(x[1]), data)))
    print(np.average(a))

# In[]
from opencc import OpenCC

def listfiles(folder, fileonly=False):
    for root, folders, files in os.walk(folder):
        f = files if fileonly else folders + files
        for filename in f:
            yield os.path.join(root, filename)

cc = OpenCC(config='s2t')

dbname = '../data/wiki.db'
db = sqlite3.connect(dbname)
dc = db.cursor()
dc.execute("""
CREATE TABLE IF NOT EXISTS news_dataset (
    id interger PRIMARY KEY,
    title text NOT NULL,
    article text NOT NULL);
""")
cvt = lambda x: cc.convert(x)

from tqdm import tqdm

for i in tqdm(list(listfiles('../data/text', True))):
    j = json.loads('[' + ','.join(open(i).readlines()) + ']')
    dc.executemany(
        'INSERT INTO news_dataset (id, title, article) VALUES (?,?,?);',
        map(lambda d: (d['id'], cvt(d['title']), cvt(d['text'])), j))
    db.commit()

#In[]
db.close()
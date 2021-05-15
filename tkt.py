# In[]
from src.tokenizer import Tokenizer
from src.dataset import NewsDataset
from collections import Counter
import pandas as pd
import numpy as np

d = list(
    map(lambda x: (x[0], float(x[1])),
        map(lambda x: x.strip().split('\t'),
        filter(len, open('data/tk.vocab').readlines()))))

# In[]
w, f = zip(*d)
f = np.array(f)
print(w[:4])
print(np.exp(f[4:]).sum())

l = list(map(len, w[4:]))
freq = sorted(dict(Counter(l)).items(), key=lambda x: x[0])
print('\n'.join(f'|{i}|{j}|' for i, j in freq))
# In[]
ds = NewsDataset('data/news_dataset_clean.db')
tk = Tokenizer('data/tk')

# In[]
ll = list(map(lambda x:len(x[2]), ds.data))
# In[]
import numpy as np
sl = sorted(ll)
# In[]
print(sl[int(len(sl)*0.25)])
print(sl[int(len(sl)*0.5)])
print(sl[int(len(sl)*0.75)])

# In[]
tk.tokenize('<unk>adsf')
# In[]

N = 0
UNK = set()


for idx, t, a in ds.data:
    N += len(t) + len(a)
    tt = tk.tokenize([i for i in t], bos=False, eos=False)
    ta = tk.tokenize([i for i in a], bos=False, eos=False)
    for idx, i in enumerate(tt):
        if isinstance(i, list):
            if len(i) == 0:
                continue
            i = i[-1]
        if i == 3:
            UNK.add(t[idx])
    for idx, i in enumerate(ta):
        if isinstance(i, list):
            if len(i) == 0:
                continue
            i = i[-1]
        if i == 3:
            UNK.add(a[idx])

#In[]
for idx, t, a in ds.data:
    tt = tk.tokenize(t, bos=False, eos=False)
    ta = tk.tokenize(a, bos=False, eos=False)

# In[]
import json
# print(UNK)
print(N)
print(len(UNK))
# json.dump(list(UNK), open('ov.json','w'),ensure_ascii=False)

# In[]
N = 0
UNK = 0

for idx, t, a in ds.data:
    tt = tk.tokenize(t, bos=False, eos=False)
    ta = tk.tokenize(a, bos=False, eos=False)
    N += len(tt) + len(ta)
    UNK += len(list(filter(lambda x: x == 3, tt)))
    UNK += len(list(filter(lambda x: x == 3, ta)))

# In[]
print(N)
print(UNK)
print((N-UNK)/N)

# In[]
ch = set()
for idx, t, a in ds.data:
    ch |= set(t)
    ch |= set(a)
print(len(ch))
# In[]
wl = len(list(filter(lambda x: x == 1, map(len, w))))
print(wl)

# In[]
to = sum(map(lambda x:len(x[1])+len(x[2]), ds.data))
print((to-UNK)/to)

# tk.tokenize([j for i in ds.data for j in i[1:]], bos=False, eos=False)
# In[]
freq = {i:0 for i in ch}
for _, t, a in ds.data:
    for i in t:
        freq[i] += 1
    for i in a:
        freq[i] += 1

# In[]
s = sorted(freq.items(), key=lambda x:x[1])
lfw = set()

for idx, (c, n) in enumerate(s):
    if n == 11:
        break
    lfw.add(c)

print(idx)
print(len(lfw))
print(len(UNK))
print(s[idx])
print(UNK - lfw)
print(lfw - UNK)
k = list(map(lambda i : freq[i], lfw - UNK))
print(len(k))
print(max(k))
print(min(k))

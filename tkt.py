# In[]
from src.tokenizer import Tokenizer
from src.dataset import NewsDataset
from collections import Counter
import pandas as pd
import numpy as np

d = list(
    map(lambda x: (x[0], float(x[1])),
        map(lambda x: x.strip().split('\t'),
        filter(len, open('data/tk1.6.2_200_tag10.vocab').readlines()))))

# In[]
w, f = zip(*d)
f = np.array(f)
print(np.exp(f[4:]).sum())

l = list(map(len, w[4:]))
freq = sorted(dict(Counter(l)).items(), key=lambda x: x[0])
print('\n'.join(f'|{i}|{j}|' for i, j in freq))
# In[]
ds = NewsDataset('data/news_dataset_200_tag10_v1.6.2.db')
# ds = NewsDataset('data/wiki.db')
tk = Tokenizer('data/tk1.6.2_200_tag10')

# In[]
from src.utils import peek
d = peek(ds.data, 1)
print(d[0][2])
print(tk.detokenize(tk.tokenize(d[0][2])))
# In[]
ll = list(map(lambda x:len(x[2]), ds.data))
sl = sorted(ll)
print(sl[0])
print(sl[int(len(sl)*0.25)])
print(sl[int(len(sl)*0.5)])
print(sl[int(len(sl)*0.75)])
print(sl[-1])

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
from tqdm import tqdm
tl = []
al = []
for idx, t, a in tqdm(ds.data):
    tl.append(len(tk.tokenize(t, bos=False, eos=False)))
    al.append(len(tk.tokenize(a, bos=False, eos=False)))

tl=np.array(tl)
al=np.array(al)

#In[]
import matplotlib.pyplot as plt

ll = list(filter(lambda x:x[0]!=0 and x[1]!=0, zip(tl, al)))

ll = np.array(ll)

#In[]
plt.hist(ll[:,0], bins=31, range=(0,30))
plt.show()
# plt.savefig('title.jpg', bins=200)
# plt.close()

plt.hist(ll[:,1], bins=1000, range=(0, 1000))
plt.show()
# plt.savefig('article.jpg')
# plt.close()

#In[]
def a(ll):
    sl = sorted(ll)
    print(np.mean(sl))
    print(np.std(sl))
    print(sl[0])
    print(sl[-1])
    print(sl[int(len(sl)*0.25)])
    print(sl[int(len(sl)*0.5)])
    print(sl[int(len(sl)*0.75)])
a(tl)
a(al)

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

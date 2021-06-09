# In[]
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
from collections import namedtuple
Result = namedtuple('Result', ['data', 'mean', 'std'])

r = pickle.load(open('probs1.pickle', 'rb'))
v = list(
    map(lambda x: (x[0], float(x[1])),
        map(lambda x: x.strip().split('\t'),
        filter(len, open('data/tk2.1sntag.vocab').readlines()))))

print(len(r))
print(len(v))

# In[]


def ana(x):
    if len(x) == 0:
        return Result(data=np.array([]), mean=0, std=0)
    x = np.sort(np.array(x))
    # return x
    return Result(
        data=x,
        mean=np.average(x),
        std=np.std(x))


z = list(zip(v, r))
res = []
for vocab, probs in tqdm(z[4:]):
    res.append((vocab, ana(probs)))
N = sum([len(r.data) for v,r in res])
#In[]
c = {}
for v, r in res:
    n = len(r.data)
    if n in c:
        c[n].append((v, r))
    else:
        c[n] = [(v, r)]
# In[]
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x, y = zip(*list(map(lambda x: (x[0] / N, np.mean([r.mean for v, r in x[1]])),
    sorted(c.items(), key=lambda x: x[0]),
    )))
plt.figure(figsize=(8,8))
ax = plt.subplot(1, 1, 1)
ax.set_xlabel('a')
ax.scatter(x,y)
# for xi, yi in zip(x,y):
#     if xi > 0.007:
#         ax.annotate('你', (xi,yi), fontproperties=font)
plt.show()

# In[]


def out(r):
    print('\n'.join([f'|{v}|{c}|{r}|' for v,p,c,r in r]))


f = filter(lambda x: x[1].mean > 0, res)
out([(*v, len(r.data), r.mean)
     for v, r in sorted(f, key=lambda x: x[1].mean)[:100]])

print('-'*30)

f = filter(lambda x: x[1].mean > 0, res)
out([(*v, len(r.data), r.mean)
     for v, r in sorted(f, key=lambda x: x[1].mean, reverse=True)[:50]])

import subprocess
import os

tkname = 'data/tk1.6.2_200_tag10'
ds = 'data/news_dataset_200_tag10_v1.6.2.db'

confs = [
    {
        'name': 'model/v141_200_6l_w3_1000_a0:100',
        'warnup': 1,
        'alpha':1,
        'inplace' : True,
        'lr':'1e-3',
        'dropout': 0,
        'heads': 12,
        'batch_size' : 12,
        'sample' : 1000,
        'save_epoch': 4,
        'summary_step': 20,
        'epoch': 20,
    },
    {
        'name': 'model/v141_200_6l_c4_1000_a0:100',
        'warnup': -1,
        'alpha':1,
        'inplace' : True,
        'lr':'1e-4',
        'dropout': 0,
        'heads': 12,
        'batch_size' : 12,
        'sample' : 1000,
        'save_epoch': 4,
        'summary_step': 1000,
        'epoch': 20,
    },
    {
        'name': 'model/v141_200_5l_w24_all',
        'warnup': 1,
        'alpha': 0,
        'beta': 128,
        'inplace' : True,
        'lr':'2e-4',
        'dropout': 0.1,
        'batch_size' : 12,
        'heads': 8,
        'layer': 5,
        'sample' : -1,
        'save_epoch': 2,
        'summary_step': 1000,
        'epoch': 10,
    },
]

for conf in confs:
    if os.path.exists(conf['name']):
        continue
    cmd = ['python','main.py',
                '--data', ds,
                '--tokenizer', tkname,
                '--model_dir', conf['name'],
                '--epoch', conf['epoch'],
                '--save_epoch', conf['save_epoch'],
                '--summary_step', conf['summary_step'],
                '--lr', conf['lr'],
                '--warnup', conf['warnup'],
                '--layer', conf['layer'],
                '--dropout', conf['dropout'],
                '--alpha', conf['alpha'],
                '--beta', conf['beta'],
                '--seed', 8787,
                '--batch_size', conf['batch_size'],
                '--sample', conf['sample']
                ]
    if conf['inplace']:
        cmd.append('--inplace')
    subprocess.run(list(map(str, cmd)))

exit()

for conf in confs:
    cmd = list(map(str, ['python','run_decode.py',
                '--data', ds,
                '--tokenizer', tkname,
                '--model_dir', conf['name'],
                '--peek', 10,
                '-a', 0,
                '-b', 128,
                '--layer', 6,
                '--seed', str(8787),
                '--inplace',
                '--markdown',
                '--device', 'cuda',
                ]))

    subprocess.run(cmd)

    cmd = list(map(str, ['python','run_decode.py',
                '--data', ds,
                '--tokenizer', tkname,
                '--model_dir', conf['name'],
                '--peek', 10,
                '-a', 1,
                '-b', 128,
                '--layer', 6,
                '--seed', str(8787),
                '--inplace',
                '--markdown',
                '--device', 'cuda',
                '--inplace'
                ]))

    subprocess.run(cmd)
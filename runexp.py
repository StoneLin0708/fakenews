import subprocess
import os

tkname = 'data/tk1.6.2_200_tag10'
ds = 'data/news_dataset_200_tag10_v1.6.2.db'

confs = [
    {
        'name': 'model/v162_3l8h_c24_2k_a0',
        'warnup': -1,
        'alpha': 0,
        'beta': 128,
        'inplace' : True,
        'lr':'2e-4',
        'dropout': 0,
        'batch_size' : 16,
        'heads': 8,
        'layer': 3,
        'sample' : 2000,
        'save_epoch': 8,
        'summary_step': 25,
        'epoch': 40,
    },
    {
        'name': 'model/v162_3l8h_c24_2k_a5i',
        'warnup': -1,
        'alpha': 0.5,
        'beta': 128,
        'inplace' : True,
        'lr':'2e-4',
        'dropout': 0,
        'batch_size' : 16,
        'heads': 8,
        'layer': 3,
        'sample' : 2000,
        'save_epoch': 8,
        'summary_step': 25,
        'epoch': 40,
    },
    {
        'name': 'model/v162_3l8h_c24_2k_a10i',
        'warnup': -1,
        'alpha': 1,
        'beta': 128,
        'inplace' : True,
        'lr':'2e-4',
        'dropout': 0,
        'batch_size' : 16,
        'heads': 8,
        'layer': 3,
        'sample' : 2000,
        'save_epoch': 8,
        'summary_step': 25,
        'epoch': 40,
    },
    {
        'name': 'model/v162_3l8h_c25_all_a10i',
        'warnup': -1,
        'alpha': 1,
        'beta': 128,
        'inplace' : True,
        'lr':'2e-5',
        'dropout': 0,
        'batch_size' : 16,
        'heads': 8,
        'layer': 3,
        'sample' : -1,
        'save_epoch': 4,
        'summary_step': 2000,
        'epoch': 20,
    },
    {
        'name': 'model/v162_3l8h_c15_all_a10i',
        'warnup': -1,
        'alpha': 1,
        'beta': 128,
        'inplace' : True,
        'lr':'1e-5',
        'dropout': 0,
        'batch_size' : 12,
        'heads': 8,
        'layer': 4,
        'sample' : -1,
        'save_epoch': 8,
        'summary_step': 500,
        'epoch': 40,
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
                '--heads', conf['heads'],
                '--seed', 8787,
                '--batch_size', conf['batch_size'],
                '--sample', conf['sample']
                ]
    if conf['inplace']:
        cmd.append('--inplace')
    subprocess.run(list(map(str, cmd)))

for conf in confs:
    cmd = list(map(str, ['python','run_decode.py',
                '--data', ds,
                '--tokenizer', tkname,
                '--model_dir', conf['name'],
                '--peek', 3,
                '-a', 0,
                '-b', 128,
                '--layer', conf['layer'],
                '--heads', conf['heads'],
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
                '--peek', 3,
                '-a', 1,
                '-b', 128,
                '--layer', conf['layer'],
                '--heads', conf['heads'],
                '--seed', str(8787),
                '--inplace',
                '--markdown',
                '--device', 'cuda',
                ]))

    subprocess.run(cmd)
import subprocess
import os

confs = [
    {
        'ds': 'data/news_dataset_tag10_v2.1.db',
        'name': 'model/v21_w2_5e4_40k_a1i',
        'tk': 'data/tk2.1s',
        'warnup': 2,
        'alpha': 1,
        'beta': 64,
        'inplace': True,
        'lr': '2e-4',
        'dropout': 0.05,
        'batch_size': 12,
        'heads': 8,
        'layer': 4,
        'sample': 40000,
        'save_epoch': 5,
        'summary_step': 500,
        'epoch': 40,
        'out': 'results/0.csv',
        'rp': 'results/0.html',
        'beam': [],
        'topk': [1, 5, 10],
        'tag': True,
    },
    {
        'ds': 'data/news_dataset_tag10_v2.1.db',
        'name': 'model/v21_w2_5e4_40k_a1i_c1',
        'tk': 'data/tk2.1s_c1',
        'warnup': 2,
        'alpha': 1,
        'beta': 64,
        'inplace': True,
        'lr': '2e-4',
        'dropout': 0.05,
        'batch_size': 12,
        'heads': 8,
        'layer': 4,
        'sample': 40000,
        'save_epoch': 5,
        'summary_step': 500,
        'epoch': 40,
        'out': 'results/1.csv',
        'rp': 'results/1.html',
        'beam': [],
        'topk': [1, 5, 10],
        'tag': True,
    },
    {
        'ds': 'data/news_v2.1_ntag.db',
        'name': 'model/v21_w2_5e4_40k_a1i_c9999_ntag',
        'tk': 'data/tk2.1sntag',
        'warnup': 2,
        'alpha': 1,
        'beta': 64,
        'inplace': True,
        'lr': '2e-4',
        'dropout': 0.05,
        'batch_size': 12,
        'heads': 8,
        'layer': 4,
        'sample': 40000,
        'save_epoch': 5,
        'summary_step': 500,
        'epoch': 40,
        'out': 'results/2.csv',
        'rp': 'results/2.html',
        'beam': [],
        'topk': [1, 5, 10],
        'tag': False,
    },
]

if True:
    for conf in confs:
        if os.path.exists(conf['name']):
            continue
        print(f'run : {conf["name"]}')
        cmd = ['python', 'main.py',
               '--data', conf['ds'],
               '--tokenizer', conf['tk'],
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
    # cmd = list(map(str, ['python','run_decode.py',
    #             '--data', ds,
    #             '--tokenizer', conf['tk'],
    #             '--model_dir', conf['name'],
    #             '--peek', 5,
    #             '-a', 0,
    #             '-b', 128,
    #             '--layer', conf['layer'],
    #             '--heads', conf['heads'],
    #             '--seed', str(8787),
    #             '--inplace',
    #             '--markdown',
    #             '--device', 'cpu',
    #             ]))

    # subprocess.run(cmd)

    cmd = ['python', 'run_decode.py',
                         '--data', conf['ds'],
                         '--tokenizer', conf['tk'],
                         '--model_dir', conf['name'],
                         '--peek', 5,
                         '-a', 1,
                         '-b', 64,
                         '--aids', 31580, 71360, 110562, 86595, 88860,
                         '--layer', conf['layer'],
                         '--heads', conf['heads'],
                         '--seed', 8787,
                         '--inplace',
                         '--device', 'cuda',
                         '--output', conf['out']
                         ]

    if 'beam' in conf and len(conf['beam']) > 0:
        cmd += ['--beam', *conf['beam']]
    if 'topk' in conf and len(conf['topk']) > 0:
        cmd += ['--topk', *conf['topk']]

    o = subprocess.check_output(list(map(str,cmd)))

    cmd = ['python', 'genrp.py',
           '--input', conf['out'],
           '--output', conf['rp']
           ]
    print(o.decode('utf-8'))
    if conf['tag']:
        cmd.append('--tag')

    subprocess.run(list(map(str, cmd)))

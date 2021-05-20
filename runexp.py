import subprocess

tkname = 'data/tk1.4.1_200'
ds = 'data/news_dataset_clean_200_v1.4.1.db'
sample = 1000
batch_size = 12
save_epoch= 2
summary_step= 20

confs = [
    {
        'name': 'model/v141_200_6l_w3_1000_a0:100',
        'warnup': 1,
        'alpha':1,
        'inplace' : True,
        'lr':'1e-3',
        'batch_size' : batch_size,
        'sample' : sample,
        'save_epoch': save_epoch,
        'summary_step': summary_step,
    }
]

for conf in confs:
    cmd = ['python','main.py',
                '--data', ds,
                '--tokenizer', tkname,
                '--model_dir', conf['name'],
                '--epoch', '20',
                '--save_epoch', str(conf['save_epoch']),
                '--summary_step', str(conf['summary_step']),
                '--lr', conf['lr'],
                '--warnup', str(conf['warnup']),
                '--layer', '6',
                '--alpha', str(conf['alpha']),
                '--beta', '64',
                '--batch_size', str(conf['batch_size']),
                '--sample', str(conf['sample'])
                ]
    if conf['inplace']:
        cmd.append('--inplace')
    subprocess.run(cmd)
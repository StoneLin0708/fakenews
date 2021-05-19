import os
import re
import torch
from torch.utils import tensorboard
from tqdm import tqdm
from src.decode import beam_search
from src.utils import markdown_pred_summary


class Optimizer:
    def __init__(self, init_lrs, optimizer, d_model, warnup, total_steps, init_step=1):
        self.lrs = init_lrs
        self.opti = optimizer
        self._step = init_step
        self.warnup = warnup
        self.d_model = d_model
        self.total_steps = total_steps

    def get_lr_ratio(self):
        if self._step < self.warnup:
            return float(self._step) / float(max(1, self.warnup))
        return max(0.,
                   float(self.total_steps - self._step) / float(max(1, self.total_steps - self.warnup)))

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [lr * ratio for lr in self.lrs]

    def step(self):
        for p, lr in zip(self.opti.param_groups, self.get_lr()):
            p['lr'] = lr
        self._step += 1
        self.opti.step()
        self.opti.zero_grad()

    def load_state_dict(self, d, step):
        self._step = step
        self.opti.load_state_dict(d)

    def state_dict(self):
        return self.opti.state_dict()


def train(
        model,
        dataset,
        batch_size,
        epochs,
        device,
        tokenizer,
        model_dir,
        save_epoch,
        summary_step,
        lr=1e-5,
        warnup=1.
):

    def decode(seq):
        return tokenizer.detokenize(seq)

    def sample_fristk(src, tgt, pred, k=3):
        return list(zip(*list(map(lambda i: decode(i[:k, :].cpu().tolist()),
                        [src,
                         torch.argmax(torch.nn.functional.softmax(
                             pred, dim=-1), dim=-1),
                         tgt]))))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(tokenizer)
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = Optimizer(
        init_lrs=[lr],
        optimizer=torch.optim.Adam([{'params': model.parameters()}], lr=lr),
        d_model=model.d_model,
        warnup=len(dataloader)*warnup,
        total_steps=len(dataloader)*epochs)

    summary = tensorboard.SummaryWriter(os.path.join(model_dir, 'log'))

    def load_ckpt():
        if ckpt is not None:
            step = int(re.match(r'.+/(\d+).pt', ckpt).group(1))
            ckpt = torch.load(ckpt)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'], step)

    save_step = save_epoch * len(dataloader)

    global_steps = 0

    def pad(inseq, max_len):
        return torch.nn.utils.rnn.pad_sequence(
            list(map(lambda x: torch.LongTensor(x[:max_len]), inseq)),
            padding_value=0,
            batch_first=True).to(device)

    for e in tqdm(range(epochs)):
        model.to(device)
        model.train()
        model.zero_grad()
        epoch_loss = 0.
        epoch_it = tqdm(dataloader)

        for local_step, (src, tgt) in enumerate(epoch_it, start=1):
            global_steps += 1

            src = pad(src, 32)
            tgt = pad(tgt, 513)

            pred = model(src=src, tgt=tgt[:, :-1])

            loss = criterion(
                pred.reshape(-1, tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
            loss.backward()

            optimizer.step()
            step_loss = loss.item()
            epoch_loss += step_loss
            epoch_it.set_description(
                f'[{global_steps}]{epoch_loss/local_step:.5f}')

            if global_steps % summary_step == 0:
                summary.add_scalar(
                    f'loss', step_loss, global_step=global_steps)
                summary.add_scalar(
                    f'lr', optimizer.get_lr()[0], global_step=global_steps)
                summary.add_text('text',
                                 markdown_pred_summary(
                                     sample_fristk(src, tgt, pred)),
                                 global_step=global_steps)

            if global_steps % save_step == 0:
                for s, p, t in sample_fristk(src, tgt, pred):
                    print(f'{s}\n{t}\n{p}\n-------------------')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, f'{global_steps}.pt'))

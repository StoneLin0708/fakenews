import os
import re
import torch
from torch.utils import tensorboard
import tqdm
from src.decode import beam_search
from src.utils import markdown_pred_summary


class Optimizer:
    def __init__(self, optimizer, d_model, warnup, init_step=1):
        self.opti = optimizer
        self._step = init_step
        self.warnup = warnup
        self.d_model = d_model

    def get_lr(self):
        return (self.d_model ** -.5) * min(self._step ** -.5, self._step * self.warnup ** -1.5)

    def step(self):
        lr = self.get_lr()
        for p in self.opti.param_groups:
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
        lr = 1e-5
        ):

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(tokenizer)
    )

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = Optimizer(
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        d_model=model.d_model,
        warnup=len(dataloader)*0.5)

    summary = tensorboard.SummaryWriter(os.path.join(model_dir, 'log'))

    def load_ckpt():
        if ckpt is not None:
            step = int(re.match(r'.+/(\d+).pt', ckpt).group(1))
            ckpt = torch.load(ckpt)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'], step)

    save_step = save_epoch * len(dataloader)

    global_steps = 0

    for e in range(epochs):
        model.to(device)
        model.train()
        model.zero_grad()
        epoch_loss = 0.
        epoch_it = tqdm.tqdm(dataloader)

        for local_step, (src, tgt) in enumerate(epoch_it, start=1):
            global_steps += 1

            src = torch.nn.utils.rnn.pad_sequence(
                list(map(lambda x: torch.LongTensor(x[:32]), src)),
                padding_value=tokenizer.model.pad_id(),
                batch_first=True).to(device)
            tgt = torch.nn.utils.rnn.pad_sequence(
                list(map(lambda x: torch.LongTensor(x[:513]), tgt)),
                padding_value=tokenizer.model.pad_id(),
                batch_first=True).to(device)

            pred = model(src=src, tgt=tgt[:, :-1])

            loss = criterion(
                pred.reshape(-1, tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
            loss.backward()

            optimizer.step()
            step_loss = loss.item()
            epoch_loss += step_loss
            epoch_it.set_description(
                f'[{global_steps}][{e}]{epoch_loss/local_step:.5f}')

            if global_steps % summary_step == 0:
                summary.add_scalar(
                    f'loss', step_loss, global_step=global_steps)
                summary.add_scalar(
                    f'lr', optimizer.get_lr(), global_step=global_steps)
                summary.add_text('text', markdown_pred_summary(
                    zip(
                        tokenizer.detokenize(src[:3, :].cpu().tolist()),
                        tokenizer.detokenize(torch.argmax(torch.nn.functional.softmax(pred, dim=-1), dim=-1)[:3, :].cpu().tolist()),
                        tokenizer.detokenize(tgt[:3, :].cpu().tolist()))
                ), global_step=global_steps)

            if global_steps % save_step == 0:
                for s, t, p in zip(tokenizer.detokenize(src[:3, :].cpu().tolist()), tokenizer.detokenize(tgt[:3, :].cpu().tolist()), tokenizer.detokenize(torch.argmax(torch.nn.functional.softmax(pred, dim=-1), dim=-1)[:3, :].cpu().tolist())):
                    print(f'{s}\n{t}\n{p}\n-------------------')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, f'{global_steps}.pt'))

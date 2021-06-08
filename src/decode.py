from typing import List

import torch


@torch.no_grad()
def beam_search(
        model,
        input_sequence,
        bos_id,
        eos_id,
        beam_width: int,
        device: torch.device,
        max_seq_len: int,
) -> List[str]:

    model.eval()

    accum_prob = [0] * beam_width
    outputs = [[bos_id]]

    while True:
        # Model prediction has shape (B, S, V).
        pred = torch.nn.functional.softmax(
            model(input_sequence, torch.LongTensor(outputs).to(device)),
            dim=-1)

        # top k (B, S, beam_width).
        # only last predicted token is needed (B, beam_width).
        probs, indices = pred[:, -1, :].cpu().topk(k=beam_width, dim=-1)
        probs = probs.log()

        beams = []
        for output, acc_prob, prob, index in zip(outputs, accum_prob, probs, indices):
            beams += [(output + [int(i)], p)
                      for p, i in zip(acc_prob - prob, index)]

        beams = sorted(beams, key=lambda x: x[1])[:beam_width]

        outputs, accum_prob = list(zip(*beams))
        if input_sequence.size(0) == 1:
            input_sequence = torch.cat([input_sequence] * beam_width)

        if beams[0][0][-1] == eos_id:
            break
        if len(outputs[0]) == max_seq_len:
            break

    return outputs[0]


@torch.no_grad()
def beam_search_v2(
        model,
        input_sequence,
        tokenizer,
        is_full,
        beam_width: int,
        device: torch.device,
        max_seq_len: int,
) -> List[str]:

    model.eval()

    class myq():
        def __init__(self):
            self.q = []

        def enq(self, value):
            self.q.append(value)

        def deq(self):
            return self.q.pop(0)

        def peek(self):
            return self.q[0]

        def __len__(self):
            return len(self.q)

    queue = myq()

    for i in input_sequence:
        queue.enq((i, [tokenizer.bos_id], 0.))

    def collect_data(queue, is_full):
        batch_size = 0
        nx = 0
        ny = 0

        data = []

        while len(queue) > 0:
            x, y, p = queue.peek()
            batch_size += 1
            nx, ny = max(nx, len(x)), max(ny, len(y))
            if is_full(batch_size, nx,  ny):
                break
            data.append(queue.deq())
        if len(data) == 0:
            raise Exception('not single element can fit memory')

        return list(zip(*data))

    def pad(x):
        return torch.nn.utils.rnn.pad_sequence(
            list(map(torch.LongTensor, x)),
            padding_value=tokenizer.pad_id,
            batch_first=True).to(device)

    beams = []
    outputs = []

    eos_id = tokenizer.eos_id

    first = True
    while len(queue) > 0:
        # collect data
        inseq, outseq, acc_probs = collect_data(
            queue, is_full)

        # Model prediction has shape (B, S, V).
        pred = torch.nn.functional.softmax(
            model(pad(inseq), pad(outseq)), dim=-1)

        probs, indices = pred[:, -1, :].cpu().topk(k=beam_width, dim=-1)
        probs = probs.log()

        def proc_beam(beams, queue, outputs):
            beams = sorted(beams, key=lambda x: x[2])[:beam_width]
            if beams[0][1][-1] == eos_id or len(beams[0][1]) == max_seq_len:
                outputs.append(beams[0])
            else:
                for b in beams:
                    queue.enq(b)

        # TODO this assum all data can fit in first batch
        k = beam_width
        if first:
            k = 1
            first = False

        for ins, outs, acc_prob, prob, index in zip(inseq, outseq, acc_probs, probs, indices):
            for p, i in zip(acc_prob - prob, index):
                curr_beam = (ins, outs + [int(i)], p)
                beams.append(curr_beam)
                if len(beams) == k * beam_width:
                    proc_beam(beams, queue, outputs)
                    beams = []

        if len(queue) > 0 and len(beams) > 0 and (len(beams[0][1]) < len(queue.peek()[1])):
            proc_beam(beams, queue, outputs)
            beams = []

    result = []
    # O(N^2) search
    for i in input_sequence:
        r = -1
        for o in range(len(outputs)):
            if outputs[o][0] == i:
                r = o
                break
        result.append(outputs[r][1])
    # print(result)
    return result


@torch.no_grad()
def beam_search_v3(
        model,
        input_sequence,
        tokenizer,
        beam_width: int,
        device: torch.device,
        max_seq_len: int,
) -> List[str]:

    model.eval()

    batch_size = len(input_sequence)

    outputs = torch.LongTensor(
        [[tokenizer.pad_id] * max_seq_len]*batch_size).to(device)
    acc_probs = torch.Tensor([[0]*beam_width]*batch_size)
    input_sequence = torch.nn.utils.rnn.pad_sequence(
        list(map(torch.LongTensor, input_sequence)),
        padding_value=tokenizer.pad_id,
        batch_first=True).to(device)

    input_sequence[:, 0] = tokenizer.bos_id

    for seq_len in range(1, max_seq_len):
        # Model prediction has shape (B * W, S, V).
        # print(input_sequence.shape)
        # print(outputs.shape)
        pred = torch.nn.functional.softmax(
            model(input_sequence, outputs), dim=-1).cpu()
        # print(pred.shape)

        # top k (B, S, W).
        # only last predicted token is needed (B * W, W).
        probs, indices = pred[:, -1, :].topk(k=beam_width, dim=-1)
        # print(probs.shape)
        probs = probs.log()
        # indices = indices.reshape(batch_size, beam_width, beam_width)
        if input_sequence.size(0) == batch_size:
            batch_input = []
            for i in input_sequence:
                batch_input += [i.unsqueeze(0)] * beam_width
            input_sequence = torch.cat(batch_input).to(device)

        genseq = torch.zeros(
            (batch_size*beam_width, max_seq_len), dtype=torch.long)

        #       (1),  (S),      (W),  (W),   (W)                      (B),  ( B,S),     (B,W), (B,W),    (B,W)
        for batch_n, outs, acc_prob, prob, index in zip(range(batch_size), outputs, acc_probs, probs, indices):
            beams = [(outs[:seq_len].tolist() + [int(i)], float(p))
                     for p, i in zip(acc_prob - prob, index)]
            beams = sorted(beams, key=lambda x: x[1])[:beam_width]

            for beam_n, (o, p) in zip(range(beam_width), beams):
                genseq[batch_n * beam_width + beam_n,
                       : seq_len+1] = torch.LongTensor(o)
                acc_probs[batch_n, beam_n] = p
                # TODO eos early stop ?
        outputs = genseq.to(device)

    return outputs[::beam_width, :].cpu().tolist()


@torch.no_grad()
def topk(
        model,
        input_sequence,
        tokenizer,
        k: int,
        device: torch.device,
        max_seq_len: int,
) -> List[str]:

    model.eval()

    def pad(x):
        return torch.nn.utils.rnn.pad_sequence(
            list(map(torch.LongTensor, x)),
            padding_value=tokenizer.pad_id,
            batch_first=True).to(device)

    B = len(input_sequence)
    outputs = [[tokenizer.bos_id] for i in range(B)]

    eos_id = tokenizer.eos_id

    while True:
        # Model prediction has shape (B, S, V).
        pred = torch.nn.functional.softmax(
            model(pad(input_sequence), pad(outputs)), dim=-1)
        # (B, V) -> (B, K)
        probs, indices = pred[:, -1, :].cpu().topk(k=k, dim=-1)
        nextid = torch.gather(indices, -1,
                              torch.multinomial(
                                  torch.nn.functional.softmax(probs, dim=-1), 1)
                              )
        for o, n in zip(outputs, nextid):
            o.append(int(n))

        if len(outputs[0]) > max_seq_len or \
                all(map(lambda x: x[-1] == eos_id or x[-1] == tokenizer.pad_id, outputs)):
            break
    #print(outputs)

    return outputs

r"""Transformer Model

DESCRIPTION
    A torch implementation of transformer model [1],
    Code is based on The Annotated Transformer [2] from Harvard NLP.

    [1] Vaswani, Ashish, et al. "Attention is all you need."
    Advances in neural information processing systems. 2017.
    https://arxiv.org/pdf/1706.03762.pdf

    [2] The Annotated Transformer
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import math
import torch
import numpy as np


class SubsequentMask(torch.nn.Module):
    r"""
    Generate subsequent mask for attention layer.
    """

    def __init__(self, pad_id: int):
        r"""
        Args:
            pad_id:
                Id to be mask out.
        """
        super().__init__()
        self.pad_id = pad_id

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        r"""
        """
        src_mask = (src != self.pad_id).unsqueeze(-2)

        tgt_mask = (tgt != self.pad_id).unsqueeze(-2)
        in_size = tgt.size(1)
        subseq_mask = (torch.triu(torch.ones(
            (1, in_size, in_size)), 1) == 0).to(tgt.device)

        return src_mask, tgt_mask & subseq_mask


class PositionalEncoding(torch.nn.Module):
    r"""PositionalEncoding

    Generate a fixed sequence of vector and apply to input

    Copy from https://nlp.seas.harvard.edu/2018/04/03/attention.html

    Fomula:
        PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})
        PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})

    Examples:
        >>> batch_size, seq_len = 32, 128
        >>> x = torch.rand((batch_size, seq_len, 10))
        >>> pe = PositionalEncoding(d_model=10, dropout=.1)
        >>> encoded_x = pe(x)
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        r"""

        Args:
            d_model:
                Dimension of each vector.
            dropout:
                Dropout probability.
            max_len:
                Max length of input sequence. Defaults to 5000.
        """

        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        r"""Forward pass

        Args:
            input_sequence:
                input sequence of shape (batch size, sequence len, d_model)

        Returns:
            torch.Tensor:
                positional encoded input
        """
        input_sequence += self.positional_encoding[:, :input_sequence.size(-2)]
        return self.dropout(input_sequence)


class Attention(torch.nn.Module):
    r"""
    Attention Layer

    Perform a learnable linear transform on all input tensor (`Query`, `Key` ,`Value`).
    After that, create a score by `Query` and `Key`
    which decide how much `Value` should pass through.

    Fomula:
        Attention(Query, Key, Value, Mask, Weight)
        =Score\times Value'
        =Softmax(Query'\times Key'^T)\times Value'
        =Softmax(W_{Q}(Query)\times W_{K}(Key)^T)\times W_{V}(Value)

    """

    def __init__(self, d_model: int, d_k: int):
        r"""
        Args:
            d_model:
                input size
            d_k:
                linear transform output size
        """
        super().__init__()
        self.w_q = torch.nn.Linear(d_model, d_k)
        self.w_k = torch.nn.Linear(d_model, d_k)
        self.w_v = torch.nn.Linear(d_model, d_k)
        self.d_k = d_k

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor):
        r"""
        Args:
            query:
                Input query.
            key:
                Input key.
            value:
                Input value.
            mask:
                Whenever position of mask is false,
                set corresponding score to -1e9 then apply softmax.
                Simplified fomula : Softmax( mask(Q x K) )
        """
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return torch.nn.functional.softmax(scores, dim=-1) @ value


class MultiHeadAttention(torch.nn.Module):
    r"""
    Multi Head Attention Layer

    Parallel apply multiple different attention to the same input,
    combine results by a linear transform.

    """

    def __init__(self, d_model: int, heads: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            heads:
                Number of different attention.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.attentions = torch.nn.ModuleList(
            [Attention(d_model, d_model // heads) for _ in range(heads)])
        self.w_output = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor):
        r"""
        Concatnate result of all attention output and transform back to original shape.

        Args:
            query:
                Input query.
            key:
                Input key.
            value:
                Input value.
            mask:
                Whenever position of mask is false,
                set corresponding score to -1e9 then apply softmax.
                Simplified fomula : Softmax( mask(Q x K) )
        """
        output = torch.cat([att(query, key, value, mask)
                            for att in self.attentions], dim=-1)
        return self.dropout(self.w_output(output))


class FeedForward(torch.nn.Module):
    r"""
    Feed Forward Layer

    Stack two layer of linear transform combine with relu activation function.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            d_ff:
                Linear dimension.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.w_in = torch.nn.Linear(d_model, d_ff)
        self.w_out = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):
        r"""
        Args:
            input:
                Input tensor.
        """
        return self.dropout(self.w_out(self.dropout(torch.nn.functional.relu(self.w_in(input)))))


class AddNorm(torch.nn.Module):
    r"""
    Add two tensor and perform Layer Normalize
    """

    def __init__(self, d_model: int):
        r"""
        Args:
            d_model:
                Input size.
        """
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, input: torch.Tensor, sub: torch.Tensor):
        r"""
        Args:
            input:
                Input tensor.
            sub:
                Input tensor.
        """

        return self.norm(input + sub)


class EncoderLayer(torch.nn.Module):
    r"""
    Encode input by attention and feed forward layer
    """

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            heads:
                Number of different attention.
            d_ff:
                Feed forward layer dimension.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model=d_model, heads=heads, dropout=dropout)
        self.addnorm1 = AddNorm(d_model=d_model)
        self.addnorm2 = AddNorm(d_model=d_model)
        self.feedforward = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        r"""
        Args:
            input:
                Input tensor.
            mask:
                Mask for attention layer.
        """
        out = self.addnorm1(input, self.attention(input, input, input, mask))
        return self.addnorm2(out, self.feedforward(out))


class Encoder(torch.nn.Module):
    r"""
    Stack encoder layers
    """

    def __init__(self, d_model: int, heads: int,  d_ff: int, layers: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            heads:
                Number of different attention.
            d_ff:
                Feed forward layer dimension.
            layers:
                Number of decoder layers.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(
            d_model=d_model, d_ff=d_ff, heads=heads, dropout=dropout) for _ in range(layers)])

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        r"""
        Args:
            input:
                Input tensor to be decode.
            mask:
                Mask for attention layer.
        """
        for encoder in self.layers:
            input = encoder(input, mask)
        return input


class DecoderLayer(torch.nn.Module):
    r"""
    Decode input by attention and feed forward layer
    """

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            heads:
                Number of different attention.
            d_ff:
                Feed forward layer dimension.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.attention1 = MultiHeadAttention(
            d_model=d_model, heads=heads, dropout=dropout)
        self.attention2 = MultiHeadAttention(
            d_model=d_model, heads=heads, dropout=dropout)
        self.addnorm1 = AddNorm(d_model=d_model)
        self.addnorm2 = AddNorm(d_model=d_model)
        self.addnorm3 = AddNorm(d_model=d_model)
        self.feedforward = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, state: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        r"""
        """
        tgt = self.addnorm1(tgt, self.attention1(tgt, tgt, tgt, tgt_mask))
        tgt = self.addnorm2(tgt, self.attention2(tgt, state, state, src_mask))
        return self.addnorm3(tgt, self.feedforward(tgt))


class Decoder(torch.nn.Module):
    r"""
    Stack decoder layers
    """

    def __init__(self, d_model: int, heads: int,  d_ff: int, layers: int, dropout: float):
        r"""
        Args:
            d_model:
                Input size.
            heads:
                Number of different attention.
            d_ff:
                Feed forward layer dimension.
            layers:
                Number of decoder layers.
            dropout:
                Dropout probability.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([DecoderLayer(
            d_model=d_model, d_ff=d_ff, heads=heads, dropout=dropout) for _ in range(layers)])

    def forward(self, tgt: torch.Tensor, state: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        r"""
        """
        for decoder in self.layers:
            tgt = decoder(tgt, state, src_mask, tgt_mask)
        return tgt


def gen_mask(src, tgt, pad_id):
    src_mask = (src == pad_id)

    tgt_mask = (tgt == pad_id).unsqueeze(1)
    in_size = tgt.size(1)
    subseq_mask = (torch.triu(torch.ones(
        (1, in_size, in_size)), 1) != 0).to(tgt.device)
    return src_mask, tgt_mask & subseq_mask

class TransformerModel(torch.nn.Module):
    r"""
    A torch implementation of transformer model,
    Code is based on The Annotated Transformer from Harvard NLP.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        d_ff: int,
        heads: int,
        layers: int,
        pad_token_id: int,
        vocab_size: int,
        d_emb: int = -1,
    ):
        r"""
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.heads = heads
        self.layers = layers

        super().__init__()

        self.subseqmask = SubsequentMask(pad_id=pad_token_id)
        self.pad_id = pad_token_id

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model if d_emb <= 0 else d_emb,
            padding_idx=pad_token_id
        )

        if d_emb > 0:
            self.v2elayer = torch.nn.Linear(d_emb, d_model, bias=False)

            self.v2e = lambda x: self.v2elayer(self.embedding(x))
            self.e2v = lambda x: x @ self.v2elayer.weight @ self.embedding.weight.transpose(
                0, 1)
        else:
            self.v2e = lambda x: self.embedding(x)
            self.e2v = lambda x: x @ self.embedding.weight.transpose(0, 1)

        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(
            d_model=d_model,
            layers=layers,
            d_ff=d_ff,
            heads=heads,
            dropout=dropout
        )

        self.decoder = Decoder(
            d_model=d_model,
            layers=layers,
            d_ff=d_ff,
            heads=heads,
            dropout=dropout
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        r"""
        """
        src_mask, tgt_mask = self.subseqmask(src, tgt)
        src = self.positional_encoding(self.v2e(src))
        tgt = self.positional_encoding(self.v2e(tgt))
        return self.e2v(self.decoder(
            tgt=tgt,
            state=self.encoder(src, src_mask),
            src_mask=src_mask,
            tgt_mask=tgt_mask))

    def predict(self, input: torch.Tensor):
        r"""
        Run forward and convert output to probability by apply softmax.

        Args:
            input:
                Batch input to predict next word.
        """
        return torch.nn.functional.softmax(self(input), dim=-1)


if __name__ == '__main__':
    model = TransformerModel(
        d_model=512,
        d_ff=1024,
        dropout=.1,
        heads=8,
        layers=6,
        pad_token_id=0,
        vocab_size=10
    )
    model.to('cuda')
    model.train()
    model(torch.zeros(32, 64, dtype=torch.long).to('cuda'),
          torch.zeros(32, 64, dtype=torch.long).to('cuda'))

#!/usr/bin/env python3
"""iTransformer (ICLR 2024) plus an EO-v4 chain-of-encoder refinement.

THE BASELINE IS THE PAPER'S MODEL, NOT A PARAPHRASE
---------------------------------------------------
`DataEmbedding_inverted`, `FullAttention`, `AttentionLayer`, `EncoderLayer`,
`Encoder` and `ITransformer.forecast` are transcribed from `thuml/iTransformer`
(`model/iTransformer.py`, `layers/*.py`) so that a baseline run reproduces the
published numbers rather than approximating them. `precheck.py` PINS that: it
builds this model and the official one from the same seed and asserts the two
forward passes agree bit-for-bit, and refuses to pass when it cannot.

Nothing here imports tsm-trainer. The refinement below is modelled on EO v4's
chain-of-encoder, and the option names are EO v4's so the two can be compared,
but the code is standalone (`exp007 is independent of tsm_trainer`).

THE REFINEMENT (`coe_enabled=True`)
-----------------------------------
EO v4 runs one weight-shared encoder N times, writing each pass's forecast back
into the positions it is asked to predict, so a later pass starts from a better
guess. iTransformer has no masked value channel — its token for variate v IS
that variate's whole window — so the write-back becomes a widened window:

    pass i   token_v = Linear([ x_v (seq_len) ; A_{i-1,v} (pred_len) ])

with `A_0 = 0`. The value fed back is the pass's own forecast, the observed half
is never overwritten, and the whole recursion runs in the normalized space with
loc/scale fixed from the context — all three as in EO v4.

`coe_residual=True` (the default, and the point of the experiment) makes each
pass emit a DELTA: the reported forecast of pass n is `A_{n-1} + raw_n`, and
`A_n` is that sum. Pass 1 has `A_0 = 0`, so a single pass is bit-identical to
`coe_residual=False` — the same invariant EO v4 documents.

`coe_bottleneck=False` is EO v4's ablation: embed once, run the encoder N times
in hidden space (`h_i = encoder(h_{i-1}) + h_{i-1}` under `coe_residual`),
project once. It does not widen the embedding, so no forecast is ever fed back.

WHAT IS DELIBERATELY NOT PORTED
-------------------------------
`coe_repeat_encoding` and the `init_*` warm-up family have no home here: both
write into EO v4's MASK channel / masked value region, and an inverted
embedding has neither. `coe_early_stop` is inference instrumentation for a
sample-level path that does not exist in this experiment.

ARCHITECTURE IS A FLAG, NOT AN INFERENCE
----------------------------------------
`coe_enabled` — not `coe_train_depth_max > 1` — decides whether the embedding is
widened, because the depth is allowed to differ between training and eval and a
derived architecture would silently mean "the checkpoint cannot run the depth
you asked for". A baseline checkpoint asked for depth > 1 is refused, loudly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ModelConfig:
    """Everything the weights depend on. Saved beside every checkpoint."""

    seq_len: int = 96
    pred_len: int = 96
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True
    #: Extra tokens the embedding carries (timestamp features). 0 for the
    #: datasets the paper runs mark-free (Solar, PEMS).
    n_marks: int = 0

    # --- EO v4 chain-of-encoder ---------------------------------------
    coe_enabled: bool = False
    coe_train_depth_max: int = 1
    coe_eval_depth: int = 1
    coe_residual: bool = True
    coe_bottleneck: bool = True
    coe_stochastic_repeat: bool = True
    coe_backprop: str = "last"          # "last" | "all"
    coe_internal_loss: bool = False
    #: Give the widened window the forecast region's real timestamps. EO v4's
    #: time channel does span the forecast region, so True is the faithful
    #: setting; False zero-fills it, which is what makes a depth-1 forward
    #: bit-identical to the baseline (precheck pins that case).
    coe_future_marks: bool = True

    def __post_init__(self) -> None:
        if not self.coe_enabled:
            # Refusing rather than clamping: a baseline checkpoint has no slot
            # to write a forecast back into, so "depth 3" would quietly be
            # depth 1 and the run would look like a null result.
            if self.coe_train_depth_max > 1 or self.coe_eval_depth > 1:
                raise ValueError(
                    "coe_train_depth_max/coe_eval_depth > 1 needs "
                    "coe_enabled=True: without it the embedding has no "
                    "forecast slot and every pass would see the same input.")
        if self.coe_train_depth_max < 1 or self.coe_eval_depth < 1:
            raise ValueError("COE depths are >= 1")
        if self.coe_backprop not in ("last", "all"):
            raise ValueError("coe_backprop is 'last' or 'all'")
        if self.coe_internal_loss and self.coe_backprop != "all":
            # EO v4's rule: under "last" the intermediate passes are grad-free,
            # so a loss on them would contribute nothing while reporting a
            # different number.
            raise ValueError(
                "coe_internal_loss requires coe_backprop='all' — under 'last' "
                "passes 1..N-1 run in no_grad and carry no gradient.")

    @property
    def context_width(self) -> int:
        """Input width of the inverted embedding's Linear."""
        if self.coe_enabled and self.coe_bottleneck:
            return self.seq_len + self.pred_len
        return self.seq_len

    @property
    def variant(self) -> str:
        if not self.coe_enabled:
            return "baseline"
        parts = [f"coe{self.coe_train_depth_max}"]
        parts.append("res" if self.coe_residual else "nores")
        parts.append("bn" if self.coe_bottleneck else "hidden")
        return "-".join(parts)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# layers — transcribed from thuml/iTransformer
# ---------------------------------------------------------------------------
class DataEmbedding_inverted(nn.Module):
    """`layers/Embed.py`. Timestamps enter as extra TOKENS, not extra width."""

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)                       # [B, Variate, Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(
                torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)                       # [B, Variate, d_model]


class FullAttention(nn.Module):
    """`layers/SelfAttention_Family.py`, with the unused masking path dropped:
    iTransformer constructs it with `mask_flag=False` and attends over variates,
    where a causal mask is meaningless."""

    def __init__(self, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values)
        return self.out_projection(out.view(B, L, -1))


class EncoderLayer(nn.Module):
    """`layers/Transformer_EncDec.py`."""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = x + self.dropout(self.attention(x, x, x))
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    """`layers/Transformer_EncDec.py`. The `conv_layers` distilling branch is
    Informer's; iTransformer never builds one."""

    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class ITransformer(nn.Module):
    """iTransformer, optionally refined by an EO-v4 chain of encoders."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.use_norm = cfg.use_norm
        self.enc_embedding = DataEmbedding_inverted(
            cfg.context_width, cfg.d_model, cfg.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(attention_dropout=cfg.dropout),
                    cfg.d_model, cfg.n_heads),
                cfg.d_model, cfg.d_ff,
                dropout=cfg.dropout, activation=cfg.activation)
             for _ in range(cfg.e_layers)],
            norm_layer=nn.LayerNorm(cfg.d_model))
        self.projector = nn.Linear(cfg.d_model, cfg.pred_len, bias=True)

    # -- depth ----------------------------------------------------------
    def resolve_depth(self, override: int | None = None) -> int:
        """Passes for one forward, EO v4's `_resolve_num_repeats`.

        Training draws `N ~ Uniform{1..K}` (or fixed K); eval takes the
        override, else `coe_eval_depth`, which MAY exceed K — that is the
        test-time scaling this experiment measures. Floor of 1 only.
        """
        cfg = self.cfg
        if not cfg.coe_enabled:
            return 1
        if self.training and override is None:
            if not cfg.coe_stochastic_repeat:
                return cfg.coe_train_depth_max
            return int(torch.randint(1, cfg.coe_train_depth_max + 1, (1,)).item())
        n = cfg.coe_eval_depth if override is None else int(override)
        return max(1, n)

    # -- one encoder pass ------------------------------------------------
    def _one_pass(self, x, marks, n_var):
        enc_out = self.enc_embedding(x, marks)
        enc_out = self.encoder(enc_out)
        # [B, N, E] -> [B, N, S] -> [B, S, N], dropping the timestamp tokens.
        return self.projector(enc_out).permute(0, 2, 1)[:, :, :n_var]

    def _marks(self, x_mark_enc, y_mark_fut, batch):
        """Timestamp tokens spanning the (possibly widened) window."""
        if x_mark_enc is None:
            return None
        if not (self.cfg.coe_enabled and self.cfg.coe_bottleneck):
            return x_mark_enc
        if self.cfg.coe_future_marks and y_mark_fut is not None:
            return torch.cat([x_mark_enc, y_mark_fut], dim=1)
        pad = x_mark_enc.new_zeros(
            (batch, self.pred_len, x_mark_enc.shape[-1]))
        return torch.cat([x_mark_enc, pad], dim=1)

    # -- the chains ------------------------------------------------------
    def _bottleneck_chain(self, x_enc, marks, n_var, n_passes, want_all):
        """Write each pass's forecast back into the widened window.

        `future` is A_i in EO v4's notation: the current estimate of the
        forecast region, zero on entry (the normalized mean) and never
        anything but this model's own output afterwards.
        """
        cfg = self.cfg
        B = x_enc.shape[0]
        future = x_enc.new_zeros((B, self.pred_len, n_var))
        backprop_all = cfg.coe_backprop == "all"
        outs = []
        for i in range(n_passes):
            use_grad = self.training and (backprop_all or i == n_passes - 1)
            ctx = contextlib.nullcontext() if use_grad else torch.no_grad()
            with ctx:
                raw = self._one_pass(
                    torch.cat([x_enc, future], dim=1), marks, n_var)
            # coe_residual: the head emits a delta and `future` is what it is a
            # delta against, so the reported forecast is A_{i-1} + raw.
            pred = raw + future if cfg.coe_residual else raw
            if want_all or i == n_passes - 1:
                outs.append(pred)
            future = pred
        return outs

    def _hidden_chain(self, x_enc, marks, n_var, n_passes, want_all):
        """EO v4's `coe_bottleneck=False`: embed once, encoder xN, project once.

        `coe_backprop` and `coe_internal_loss` do not apply — there is no
        per-pass head to supervise and nothing is fed back — which is exactly
        what EO v4's `_coe_hidden_chain` says about them.
        """
        cfg = self.cfg
        h = self.enc_embedding(x_enc, marks)
        outs = []
        for i in range(n_passes):
            out = self.encoder(h)
            if cfg.coe_residual:
                out = out + h               # h_i = encoder(h_{i-1}) + h_{i-1}
            h = out
            if want_all or i == n_passes - 1:
                outs.append(self.projector(h).permute(0, 2, 1)[:, :, :n_var])
        return outs

    # -- forward ---------------------------------------------------------
    def forward(self, x_enc, x_mark_enc=None, y_mark_fut=None, *,
                depth: int | None = None, return_all_passes: bool = False):
        """`x_enc` [B, seq_len, N] -> [B, pred_len, N].

        With `return_all_passes` the result is a LIST, pass 1 first, every
        entry denormalized — that is what `coe_internal_loss` supervises and
        what the depth sweep in `run_eval.py` reads from a single forward.
        """
        if self.use_norm:
            # Non-stationary normalization; loc/scale come from the context
            # ONCE and stay fixed, so every pass of the chain speaks the same
            # units (EO v4: the recursion runs in the normalized space).
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        B, _, N = x_enc.shape
        if x_mark_enc is not None and x_mark_enc.shape[-1] != self.cfg.n_marks:
            # The mark tokens are dropped by position (`[:, :, :N]`), so a
            # count that disagrees with the config is silently absorbed by the
            # projector's output slice instead of failing: a checkpoint trained
            # at one --freq would keep scoring at another.
            raise ValueError(
                f"this model was built for n_marks={self.cfg.n_marks} but was "
                f"given {x_mark_enc.shape[-1]} timestamp features — the "
                f"--freq of the run does not match the checkpoint's")
        n_passes = self.resolve_depth(depth)
        marks = self._marks(x_mark_enc, y_mark_fut, B)

        if not self.cfg.coe_enabled:
            outs = [self._one_pass(x_enc, marks, N)]
        elif self.cfg.coe_bottleneck:
            outs = self._bottleneck_chain(
                x_enc, marks, N, n_passes, return_all_passes)
        else:
            outs = self._hidden_chain(
                x_enc, marks, N, n_passes, return_all_passes)

        if self.use_norm:
            scale = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            shift = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            outs = [o * scale + shift for o in outs]
        return outs if return_all_passes else outs[-1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

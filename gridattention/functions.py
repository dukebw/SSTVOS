import math
import os

import torch

import torch.autograd as autograd
import torch.cuda.comm as comm
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.utils import cpp_extension

_gridattn = None


def init_gridattn(cfg):
    global _gridattn

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    _src_path = os.path.join(curr_dir, "src")
    _build_path = os.path.join(curr_dir, "build")
    os.makedirs(_build_path, exist_ok=True)

    _gridattn = cpp_extension.load(
        name="gridattn",
        extra_cflags=["-O3"],
        build_directory=_build_path,
        verbose=True,
        sources=[os.path.join(_src_path, f) for f in ["lib_cffi.cpp", "gridattn.cu"]],
        extra_cuda_cflags=["--expt-extended-lambda", "-O3", "--use_fast_math"],
    )


def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class GridAttnWeightFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, strides=None):
        query = query.float()
        key = key.float()

        # Save context
        n, c, t, h, w = query.size()
        size = (n, h + w + t - 2, t, h, w)
        attnscores = torch.zeros(
            size,
            dtype=query.dtype,
            layout=query.layout,
            device=query.device,
        )

        _gridattn.gridattn_forward_cuda(query, key, attnscores)

        # Output
        ctx.save_for_backward(query, key)

        return attnscores

    @staticmethod
    @once_differentiable
    def backward(ctx, dattn):
        query, key = ctx.saved_tensors

        dquery = torch.zeros_like(query)
        dkey = torch.zeros_like(key)

        _gridattn.gridattn_backward_cuda(dattn.contiguous(), query, key, dquery, dkey)

        _check_contiguous(dquery, dkey)

        return dquery, dkey, None


class GridAttnMapFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attention, value, strides=None):
        assert attention.shape[1] == (sum(value.shape[2:]) - 2)
        attention = attention.float()
        value = value.float()

        # Save context
        out = torch.zeros_like(value)
        _gridattn.gridattn_map_forward_cuda(attention, value, out)

        # Output
        ctx.save_for_backward(attention, value)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        attention, value = ctx.saved_tensors

        dattn = torch.zeros_like(attention)
        dval = torch.zeros_like(value)

        _gridattn.gridattn_map_backward_cuda(
            dout.contiguous(), attention, value, dattn, dval
        )

        _check_contiguous(dattn, dval)

        return dattn, dval, None


gridattn_weight = GridAttnWeightFunc.apply
gridattn_map = GridAttnMapFunc.apply


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        torch.nn.Module.__init__(self)
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)


class GridAttentionWeight(torch.nn.Module):
    """NOTE(brendan): W and H should be the max. size in the x and y
    dimensions, respectively, of the tensor input to attention.
    """

    def __init__(self, in_dim, cfg):
        torch.nn.Module.__init__(self)

        if cfg.MODEL.RESNETS.STAGE_WITH_DCN[-1]:
            stride = 8
        else:
            stride = 16
        H = cfg.INPUT.MAX_SIZE_TRAIN // stride
        W = cfg.INPUT.MAX_SIZE_TRAIN // stride
        self.max_frame_count = cfg.INPUT.MAX_FRAME_COUNT
        self.num_heads = cfg.MODEL.GRID_ATTENTION.DIM_REDUCE_FACTOR
        self.embinit = cfg.MODEL.GRID_ATTENTION.EMBEDDING_INITIALIZATION
        self.is_position_sincos = (
            cfg.MODEL.GRID_ATTENTION.POSITIONAL_ENCODING == "Sinusoidal"
        )

        projdim = in_dim // self.num_heads
        self.query_conv = torch.nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.key_conv = torch.nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )

        self.x_embeddings = torch.nn.Parameter(torch.Tensor((2 * W) - 1, projdim))
        self.y_embeddings = torch.nn.Parameter(torch.Tensor((2 * H) - 1, projdim))
        if cfg.MODEL.GRID_ATTENTION.POSITIONAL_ENCODING == "LearnedRelative":
            self.t_embeddings = torch.nn.Parameter(
                torch.Tensor((2 * self.max_frame_count) - 1, projdim)
            )
        elif cfg.MODEL.GRID_ATTENTION.POSITIONAL_ENCODING == "Sinusoidal":
            self.t_embeddings = PositionalEncoding(
                projdim, max_len=(2 * self.max_frame_count) - 1
            )
        else:
            assert False

        self.reset_parameters()

    def reset_parameters(self):
        if self.embinit == "Normal":
            torch.nn.init.normal_(self.x_embeddings)
            torch.nn.init.normal_(self.y_embeddings)
            if not self.is_position_sincos:
                torch.nn.init.normal_(self.t_embeddings)
        elif self.embinit == "Zeros":
            torch.nn.init.zeros_(self.x_embeddings)
            torch.nn.init.zeros_(self.y_embeddings)
            if not self.is_position_sincos:
                torch.nn.init.zeros_(self.t_embeddings)
        else:
            assert False

    def forward(self, query, key, frm_indices):
        projquery = self.query_conv(query)
        n, c, t, h, w = projquery.shape
        assert (frm_indices.shape[0] == n) and (
            frm_indices.shape[1] == t
        ), f"{frm_indices.shape}, {n}, {t}"

        projquery = projquery.view(n * self.num_heads, c // self.num_heads, t, h, w)

        projkey = self.key_conv(key)
        projkey = projkey.view(n * self.num_heads, c // self.num_heads, t, h, w)

        if self.is_position_sincos:
            # TODO(brendan): try dropout
            # t_embeddings = self.t_embeddings.dropout(self.t_embeddings.pe)
            t_embeddings = self.t_embeddings.pe
        else:
            t_embeddings = self.t_embeddings
        energy = gridattn_weight(
            projquery,
            projkey,
            self.x_embeddings,
            self.y_embeddings,
            t_embeddings,
            frm_indices.repeat(self.num_heads, 1),
        )

        attn = F.softmax(energy, dim=1)

        return attn.view(n, self.num_heads, t + h + w - 2, t, h, w)


class GridAttentionMap(torch.nn.Module):
    def __init__(self, in_dim, dim_reduce_factor, cfg):
        torch.nn.Module.__init__(self)

        self.num_heads = dim_reduce_factor

        self.value_conv = torch.nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )

    def forward(self, attention, value):
        projvalue = self.value_conv(value)
        n, c, t, h, w = projvalue.shape
        projvalue = projvalue.view(n * self.num_heads, c // self.num_heads, t, h, w)

        attended_val = gridattn_map(
            attention.view(n * self.num_heads, t + h + w - 2, t, h, w), projvalue
        )

        return attended_val.view(n, self.num_heads, c // self.num_heads, t, h, w)


class GridAttention(torch.nn.Module):
    """NOTE(brendan): See GridAttentionWeight for W, H definition."""

    def __init__(self, in_dim, cfg):
        torch.nn.Module.__init__(self)

        self.num_heads = cfg.MODEL.GRID_ATTENTION.DIM_REDUCE_FACTOR

        self.attnweight = GridAttentionWeight(in_dim, cfg)
        self.attnmap = GridAttentionMap(in_dim, self.num_heads, cfg)

        self.outconv = torch.nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )

    def forward(self, query, key, value, frm_indices):
        n, c, t, h, w = value.shape
        attention = self.attnweight(query, key, frm_indices)

        attended_val = self.attnmap(attention, value)
        attended_val = attended_val.view(n, c, t, h, w)

        return self.outconv(attended_val)

"""Unit tests for grid attention"""
import argparse

import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

from gridattention import gridattn_weight, gridattn_map, init_gridattn
from sst.configs import cfg


def _testgrad_gridattnfunc(gridattnfunc, variables):
    for i, var in enumerate(variables):
        if var.dtype == torch.float32:
            var = var.double().cuda()
            var.requires_grad = True
            variables[i] = var
        elif var.dtype == torch.int32:
            var = var.cuda()
            var.requires_grad = False
            variables[i] = var

    if torch.autograd.gradcheck(gridattnfunc, variables, eps=1e-4, atol=1e-2):
        print("Ok")
    else:
        print("Not ok")


def _test_grad():
    N = 1
    C = 4
    T = 5
    H = 6
    W = 6
    query = torch.randn((N, C, T, H, W))
    key = torch.randn((N, C, T, H, W))

    variables = [query, key]
    _testgrad_gridattnfunc(gridattn_weight, variables)

    attention = torch.randn((N, W + H + T - 2, T, H, W))
    value = torch.randn((N, C, T, H, W))

    variables = [attention, value]
    _testgrad_gridattnfunc(gridattn_map, variables)


def test_gridattention():
    parser = argparse.ArgumentParser(description="Test GridAttention")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    init_gridattn(cfg)

    N = 2
    C = 8
    H = 10
    W = 10
    T = 6

    query = torch.zeros(N, C, T, H, W).cuda() + 1.1
    key = torch.zeros(N, C, T, H, W).cuda() + 2.0
    val = torch.zeros(N, C, T, H, W).cuda() + 3.0

    attnscores = gridattn_weight(query, key)
    attn = F.softmax(attnscores, dim=1)
    out = gridattn_map(attn, val)

    _test_grad()


if __name__ == "__main__":
    test_gridattention()

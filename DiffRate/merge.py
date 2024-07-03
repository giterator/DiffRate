

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn

from DiffRate.utils import ste_min

def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = ste_min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean", training=True) -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if training:
            return torch.cat([unm, dst, src], dim=1) 
            #src needs to be masked out in training -> need to change mask ?
            
    
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None, training= True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum", training=training)
    size = merge(size, mode="sum", training=training)

    x = x / size
    return x, size

def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None, training=False
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax", training=training)
    return source

# def get_merge_func(metric: torch.Tensor, kept_number: int, class_token: bool = True):
#     with torch.no_grad():
#         metric = metric/metric.norm(dim=-1, keepdim=True)
#         unimportant_tokens_metric = metric[:, kept_number:]
#         compress_number = unimportant_tokens_metric.shape[1]
#         important_tokens_metric = metric[:,:kept_number]
#         similarity = unimportant_tokens_metric@important_tokens_metric.transpose(-1,-2)
#         if class_token:
#             similarity[..., :, 0] = -math.inf
#         node_max, node_idx = similarity.max(dim=-1)
#         dst_idx = node_idx[..., None]
#     def merge(x: torch.Tensor, mode="mean", training=False) -> torch.Tensor:
#         src = x[:,kept_number:]
#         dst = x[:,:kept_number]
#         n, t1, c = src.shape
#         dst = dst.scatter_reduce(-2, dst_idx.expand(n, compress_number, c), src, reduce=mode) 
#         if training:
#             return torch.cat([dst, src], dim=1)
#         else:
#             return dst
#     return merge, node_max

# def uncompress(x, source):
#     '''
#     input: 
#         x: [B, N', C]
#         source: [B, N', N]
#         size: [B, N', 1]
#     output:
#         x: [B, N, C]
#         source: [B, N, N]
#         size: [B, N, 1]
#     '''
#     index = source.argmax(dim=1)
#     # print(index)
#     uncompressed_x = torch.gather(x, dim=1, index=index.unsqueeze(-1).expand(-1,-1,x.shape[-1]))
#     return uncompressed_x

# def tokentofeature(x):
#     B, N, C = x.shape
#     H = int(N ** (1/2))
#     x = x.reshape(B, H, H, C)
#     return x
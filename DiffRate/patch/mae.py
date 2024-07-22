# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# mae: https://github.com/facebookresearch/mae
# --------------------------------------------------------


import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer


from .deit import DiffRateBlock, DiffRateAttention

from DiffRate.utils import ste_min




def make_diffrate_class(transformer_class):
    class DiffRateVisionTransformer(transformer_class):
        def forward(self, x, return_flop=True) -> torch.Tensor:
            B = x.shape[0]
            self._diffrate_info["size"] = torch.ones([B,self.patch_embed.num_patches+1,1], device=x.device)
            self._diffrate_info["mask"] =  torch.ones((B,self.patch_embed.num_patches+1),device=x.device)
            # self._diffrate_info["prune_kept_num"] = []
            self._diffrate_info["merge_kept_num"] = []
            self._diffrate_info["merge_kept_num_prob"] = []
            self._diffrate_info["merge_decision"] = []
            self._diffrate_info["merge_prob"] = []
            if self._diffrate_info["trace_source"]:
                self._diffrate_info["source"] = torch.eye(self.patch_embed.num_patches+1, device=x.device)[None, ...].expand(B, self.patch_embed.num_patches+1, self.patch_embed.num_patches+1)
            x = super().forward(x)
            if return_flop:
                if self.training:
                    flops = self.calculate_flop_training()
                    etrr = self.calc_etrr()
                else:
                    flops = self.calculate_flop_inference()
                    etrr = self.calc_etrr()
                return x, flops, self._diffrate_info["merge_kept_num"], self._diffrate_info["merge_decision"], etrr, self._diffrate_info["merge_kept_num_prob"]
            else:
                return x
            
        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            # From the MAE implementation
            B = x.shape[0]
            x = self.patch_embed(x)

            T = x.shape[1]

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            if self.global_pool:
                if self.training:
                    mask = self._diffrate_info["mask"][...,None]  # [B, N, 1]
                    num = (self._diffrate_info["size"] * mask)[:, 1:, :].sum(dim=1) # [B,1]
                    x = (x * self._diffrate_info["size"] * mask)[:, 1:, :].sum(dim=1) / num.detach()
                    outcome = self.fc_norm(x)
                else:
                    T = self._diffrate_info["size"][:, 1:, :].sum(dim=1)
                    if self._diffrate_info["size"] is not None:
                        x = (x * (self._diffrate_info["size"]))[:, 1:, :].sum(dim=1) / T
                    else:
                        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                    outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                x = self.pre_logits(x)
                outcome = x[:, 0]            

            return outcome
        
        def parameters(self, recurse=True):
            # original network parameter
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    continue
                params.append(m)
            return iter(params)    
        
        def arch_parameters(self):
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    params.append(m)
            return iter(params)    
    

        def get_dec(self):
            dec = []
            for block in self.blocks:
                # prune_kept_num.append(int(block.prune_ddp.kept_token_number))
                dec.append(int(block.merge_ddp.merge_decision))
            return dec #prune_kept_num, merge_kept_num
        
        def get_merge_prob(self):
            merge_prob = []
            for block in self.blocks:
                # prune_kept_num.append(int(block.prune_ddp.kept_token_number))
                merge_prob.append(float(block.merge_ddp.merge_prob))
            return merge_prob #prune_kept_num, merge_kept_num
        
        def get_kept_num(self):
            # prune_kept_num = []
            merge_kept_num = []
            for block in self.blocks:
                # prune_kept_num.append(int(block.prune_ddp.kept_token_number))
                merge_kept_num.append(int(block.merge_ddp.kept_token_number))
            return merge_kept_num
        
        def set_kept_num(self, merge_kept_numbers):
            # assert len(prune_kept_numbers) == len(self.blocks) and len(merge_kept_numbers) == len(self.blocks)
            for block, merge_kept_number in zip(self.blocks, merge_kept_numbers):
                # block.prune_ddp.kept_token_number = prune_kept_number
                block.merge_ddp.kept_token_number = merge_kept_number
        
        def calculate_flop_training(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1, device=self.blocks[0].merge_ddp.selected_probability.device)
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for merge_kept_number in self._diffrate_info["merge_kept_num"]:
                    # translate fp16 to fp32 for stable training    
                    merge_kept_number = merge_kept_number.float()
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = ste_min(N, merge_kept_number)
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops

        def calculate_flop_inference(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1, device=self.blocks[0].merge_ddp.selected_probability.device)
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for block in (self.blocks):
                    # prune_kept_number = block.prune_ddp.kept_token_number
                    merge_kept_number = block.merge_ddp.kept_token_number
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = ste_min(N, merge_kept_number)
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops
        
        def calc_etrr(self):
            mono_sched = []
            rem_tok = torch.tensor(197.0, device = torch.device('cuda:0'))
            layer = 11
            for merge_kept_number, merge_dec in zip(self._diffrate_info["merge_kept_num"], self._diffrate_info["merge_decision"]):
                r = ste_min(torch.nn.functional.relu((rem_tok - merge_kept_number)), torch.tensor(rem_tok//2, device = torch.device('cuda:0')))
                effective_r = r * merge_dec * layer
                mono_sched.append(effective_r)
                layer -=1
                rem_tok -= r

                # if merge_kept_number <= rem_tok and merge_dec == 1:
                #     mono_sched.append((rem_tok - merge_kept_number) * layer)
                #     rem_tok = merge_kept_number
                # else:
                #     mono_sched.append(0)
                # layer -= 1

            return (100 * sum(mono_sched) / (197*12))   
        

    return DiffRateVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, merge_granularity=1
):
    """
    Applies DiffRate to this transformer.
    """
    DiffRateVisionTransformer = make_diffrate_class(model.__class__)

    model.__class__ = DiffRateVisionTransformer
    model._diffrate_info = {
        "size": None,
        "mask": None,           # only for training
        "source": None,
        "class_token": model.cls_token is not None,
        "trace_source": trace_source,
    }

    block_index = 0
    # non_compressed_block_index = [0]
    non_compressed_block_index = [0, len(model.blocks)-1]
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = DiffRateBlock
            if block_index in non_compressed_block_index:
                module.introduce_diffrate(model.patch_embed.num_patches, model.patch_embed.num_patches+1)
            else:
                module.introduce_diffrate(model.patch_embed.num_patches, merge_granularity)
            block_index += 1
            module._diffrate_info = model._diffrate_info
        elif isinstance(module, Attention):
            module.__class__ = DiffRateAttention
import itertools
import numpy as np
import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.openaimodel import UNetModel

from ldm.modules.attention import SpatialTransformer
from modules.model import GatedSelfAttentionDense, FourierEmbedder

BLOCK_PREFIXES = [
    'input_blocks.0.',
    'input_blocks.1.',
    'input_blocks.2.',
    'input_blocks.3.',
    'input_blocks.4.',
    'input_blocks.5.',
    'input_blocks.6.',
    'input_blocks.7.',
    'input_blocks.8.',
    'input_blocks.9.',
    'input_blocks.10.',
    'input_blocks.11.',
    'middle_block.',
    'output_blocks.0.',
    'output_blocks.1.',
    'output_blocks.2.',
    'output_blocks.3.',
    'output_blocks.5.',
    'output_blocks.4.',
    'output_blocks.6.',
    'output_blocks.7.',
    'output_blocks.8.',
    'output_blocks.9.',
    'output_blocks.10.',
    'output_blocks.11.',
]

class ProxyBasicTransformerBlock(object):
    def __init__(self, controller, org_module: torch.nn.Module = None):
        super().__init__()
        self.org_module = org_module
        self.org_forward = None
        self.fuser = None
        self.attached = False
        self.controller = controller
        self.objs = None


    def __getattr__(self, attr):
        if attr not in ['org_module', 'org_forward', 'fuser', 'attached', 'controller', 'objs'] and self.attached:
            return getattr(self.org_module, attr)

    def initialize_fuser(self, fuser_state_dict):
        query_dim = self.org_module.attn1.to_q.in_features
        key_dim = self.org_module.attn2.to_k.in_features
        n_heads = self.org_module.attn1.heads
        d_head = int(self.org_module.attn2.to_q.out_features / n_heads)
        self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head)
        self.fuser.load_state_dict(fuser_state_dict)

    def apply_to(self):
        if self.org_forward is not None:
            return
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        self.attached = True

    def detach(self):
        if self.org_forward is None:
            return
        self.org_module.forward = self.org_forward
        self.org_forward = None
        self.attached = False

    def forward(self, x, context):
        x = self.attn1( self.norm1(x) ) + x
        x = self.fuser(x,  self.controller.batch_objs_input) # identity mapping in the beginning
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class InteractDiffusion(nn.Module):
    def __init__(self, unet:UNetModel, state_dict):
        super().__init__()

        self.unet = unet
        self.state_dict = state_dict
        self.position_net_state_dict = {key:value for key, value in state_dict.items() if 'transformer_block' not in key}

        self.attention_modules = []
        self.proxy_transformer_blocks = []

        self.create_proxies()

    def verify_transformer_block(self, block_id, state_dict):
        if len(state_dict) != 0 and \
            all(block_id in key for key in state_dict.keys()) \
            and all('transformer_block' in key for key in state_dict):

            return True
        else:
            return False
    
    def trim_keys(self, state_dict, identifier):
        pointer = list(state_dict.keys())[0].index(identifier) + len(identifier)
        state_dict = {key[pointer:]: value for key, value in state_dict.items()}
            
        return state_dict
    
    def create_proxies(self):
        for block_idx, unet_block in enumerate(itertools.chain(self.unet.input_blocks, [self.unet.middle_block], self.unet.output_blocks)):
            block_id = BLOCK_PREFIXES[block_idx]
            block_state_dict = {key: value for key, value in self.state_dict.items() if key.startswith(block_id)}

            if self.verify_transformer_block(block_id, block_state_dict):
                block_state_dict = self.trim_keys(block_state_dict, 'fuser.')
            else:
                continue

            for module in unet_block.modules():
                if type(module) is SpatialTransformer:
                    spatial_transformer = module
                    for basic_transformer_block in spatial_transformer.transformer_blocks:
                        proxy_transformer_block = ProxyBasicTransformerBlock(self, basic_transformer_block)
                        proxy_transformer_block.initialize_fuser(block_state_dict)

                        self.proxy_transformer_blocks.append(proxy_transformer_block)
                        self.attention_modules.append(proxy_transformer_block.fuser)

        
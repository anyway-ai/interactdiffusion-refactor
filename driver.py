import torch
from interactdiffusion_pluggable import PluggableInteractDiffusion
from pluggable import InteractDiffusion

from ldm.modules.diffusionmodules.openaimodel import UNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"

unet_model_path = "./models/unet_model.bin"
interactdiff_model_path = "./models/ext_interactdiff_v1.2.pth"

interactdiff_model = torch.load(str(interactdiff_model_path), map_location=device)
model = UNetModel(image_size = 64, 
                  in_channels=4, 
                  model_channels=320, 
                  out_channels=4, 
                  num_res_blocks=2,
                  channel_mult = [ 1, 2, 4, 4 ],
                  attention_resolutions = [4, 2, 1], 
                  num_heads = 8, 
                  use_spatial_transformer=True, 
                  transformer_depth = 1,
                  context_dim = 768, 
                  legacy=False, 
                  use_checkpoint=True
                  )
"""
keys = list(interactdiff_model.keys())
for net in keys:
    print(net)


print(len(keys))

"""

pluggable = InteractDiffusion(model, interactdiff_model)

#pluggable = PluggableInteractDiffusion(model, interactdiff_model)

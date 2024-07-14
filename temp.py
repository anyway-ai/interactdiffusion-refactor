import torch
from interactdiffusion_pluggable import PluggableInteractDiffusion

from ldm.modules.diffusionmodules.openaimodel import UNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"

unet_model_path = "./models/unet_model.bin"
interactdiff_model_path = "./models/ext_interactdiff_v1.2.pth"

openai_dummy_unet = UNetModel()

unet_model = torch.load(str(unet_model_path), map_location=device) 
interactdiff_model = torch.load(str(interactdiff_model_path), map_location=device)

model = PluggableInteractDiffusion(openai_dummy_unet, interactdiff_model)

print(model)
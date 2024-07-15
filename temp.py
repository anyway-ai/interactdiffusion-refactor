import torch
from interactdiffusion_pluggable import PluggableInteractDiffusion

from ldm.modules.diffusionmodules.openaimodel import UNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"

unet_model_path = "./models/unet_model.bin"
interactdiff_model_path = "./models/ext_interactdiff_v1.2.pth"
#openai_dummy_unet = UNetModel()



unet_model = torch.load(str(unet_model_path), map_location=device) 
#interactdiff_model = torch.load(str(interactdiff_model_path), map_location=device)

#model = PluggableInteractDiffusion(unet_model, interactdiff_model)
#print(model)


keys = unet_model.keys()
print([key for key in keys if key.startswith("down_blocks")] + [key for key in keys if key.startswith("mid_blocks")] + [key for key in keys if key.startswith("up_blocks")])
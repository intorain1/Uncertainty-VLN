from PIL import Image
import os
from collections import defaultdict
import json
import tqdm
import sys
sys.path.append('/home/user/intorains/Uncertainty-VLN/')
import numpy as np

import torch
from src.base.model import ProLIP, ProLIPVisionCfg, ProLIPTextCfg, TScfg
from transformers import CLIPProcessor
from src.base.tokenizer import HFTokenizer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
checkpoint_path = '/home/user/intorains/Uncertainty-VLN/log/2025_11_17-07_32_50-model_ViT-B-16-ProLIP-long-lr_1e-05-b_256-j_4-p_amp_bf16/checkpoints/epoch_278.pt'
vision_cfg=ProLIPVisionCfg(embed_unc=True)
text_cfg=ProLIPTextCfg(context_length=256, embed_unc=True)
ts_cfg=TScfg(layers=4)
model = ProLIP(embed_dim=768, vision_cfg=vision_cfg, text_cfg=text_cfg, ts_cfg=ts_cfg, init_logit_bias=True)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
with torch.serialization.safe_globals([np._core.multiarray.scalar]):
    model.load_state_dict(new_state_dict)    
model.eval()
tokenizer = HFTokenizer("timm/ViT-B-16-SigLIP", context_length=64, clean="canonicalize")

def inclusion_test(mu1, logsigma_sq1, mu2, logsigma_sq2):
    """ Test if mu1, logsigma_sq1 is included in mu2, logsigma_sq2
    The test returns a large value if 1 is included in 2, otherwise returns a small value
    """
    inv_sigma_sq1 = torch.exp(-logsigma_sq1)
    inv_sigma_sq2 = torch.exp(-logsigma_sq2)

    a = inv_sigma_sq1 + 0.5 * inv_sigma_sq2
    b = 2 * mu1 * inv_sigma_sq1 + mu2 * inv_sigma_sq2
    c = mu1 ** 2 * inv_sigma_sq1 + 0.5 * mu2 ** 2 * inv_sigma_sq2

    return -2 * logsigma_sq1 - logsigma_sq2 - 0.5 * torch.log(a) + b ** 2 / 4 / a - c

input_folder = '/home/user/intorains/Matterport3D_O'
input_sceen_list = ['zsNo4HB9uLZ', '8194nk5LbLH','EU6Fwq7SyZv', 'oLBMNvg9in8', 'pLe4wQe7qrG', 'QUCTc6BB5sX', 'TbHJrupSAjP', 'X7HyMhZNoso', 'x8F5xyUWy9e', 'Z6MFQCViBuw', '2azQ1b91cZZ']

image_place = []
image_feature = defaultdict()
test_instr = defaultdict(lambda: {'instr': [], 'path': None})
test_output = defaultdict(lambda: {'scores': []})


for item in input_sceen_list:
    print(f'processing scene {item}')
    for root, _, files in os.walk(os.path.join(input_folder, item)):
        for fname in files:
            if os.path.splitext(fname)[1].lower() == '.png' and 'rgb' in fname:
                info = os.path.splitext(fname)[0].split('_')[0]
                image_place.append(info)

    for place in image_place:
        if not os.path.exists(os.path.join(input_folder, item, f'{place}_rgb.png')):
            continue
        image = Image.open(os.path.join(input_folder, item, f'{place}_rgb.png')).convert('RGB')
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_feature[place] = inputs

    instruction_path = f'/home/user/intorains/R2R/annotations/R2R_val_unseen_instr_scan_{item}.json'
    with open(instruction_path, 'r') as f:
        instructions = json.load(f)

    for instr in instructions:
        test_instr[instr['path_id']]['instr'].append(instr['instruction'])
        test_instr[instr['path_id']]['path'] = instr['path']

# print(len(test_instr))

for instr in tqdm.tqdm(test_instr):
    texts = tokenizer(test_instr[instr]['instr'])

    total_csd_i2t = torch.zeros(1, 3)
    total_csd_t2i = torch.zeros(1, 3)
    total_text_unc = torch.zeros(1, 3)
    total_img_unc = torch.zeros(1, 3)
    total_inclu = torch.zeros(1, 3)

    input = []
    for place in test_instr[instr]['path']:
        if place not in image_feature:
            continue
        inputs = image_feature[place]
        input.append(inputs["pixel_values"])

    if len(input) == 0:
        continue
    final_input = torch.cat(input, dim=0).unsqueeze(0)

    outputs = model(image_seq=final_input, text=texts)
    l2_logit = outputs["image_features"]["mean"] @ outputs["text_features"]["mean"].T
    i_unc = torch.exp(outputs["image_features"]["std"]).sum(dim=-1)
    t_unc = torch.exp(outputs["text_features"]["std"]).sum(dim=-1)
    csd_logit = l2_logit - 0.5 * t_unc
    csd_logit2 = l2_logit.T - 0.5 * i_unc

    total_csd_i2t += csd_logit
    total_csd_t2i += csd_logit2.T
    total_text_unc += t_unc
    total_img_unc += i_unc.unsqueeze(1)

    total_inclu[0][0] = inclusion_test(
        outputs["image_features"]["mean"][0], outputs["image_features"]["std"][0],
        outputs["text_features"]["mean"][0], outputs["text_features"]["std"][0]
    ).mean()

    total_inclu[0][1] = inclusion_test(
        outputs["image_features"]["mean"][0], outputs["image_features"]["std"][0],
        outputs["text_features"]["mean"][1], outputs["text_features"]["std"][1]
    ).mean()

    total_inclu[0][2] = inclusion_test(
        outputs["image_features"]["mean"][0], outputs["image_features"]["std"][0],
        outputs["text_features"]["mean"][2], outputs["text_features"]["std"][2]
    ).mean()

    test_output[instr]['scores'] = {
        'csd_i2t': total_csd_i2t.tolist(),
        'csd_t2i': total_csd_t2i.tolist(),
        'text_unc': total_text_unc.tolist(),
        'inclusion': total_inclu.tolist(),
        'img_unc': total_img_unc.tolist()
    }

# print(test_output)
output_file_path = f'/home/user/intorains/Uncertainty-VLN/output/test_output.json'

with open(output_file_path, 'w') as outfile:
    json.dump(test_output, outfile, indent=4)

print(f'Test output written to {output_file_path}')
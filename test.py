from src.base.model import ProLIP, ProLIPHF, ProLIPVisionCfg, ProLIPTextCfg, TScfg
from huggingface_hub import hf_hub_download
import torch
# from src.base.transformer import TSTransformer

# model = ProLIPHF.from_pretrained("SanghyukChun/ProLIP-ViT-B-16-DC-1B-12_8B")

model = ProLIP(embed_dim=768, vision_cfg=ProLIPVisionCfg(embed_unc=True), text_cfg=ProLIPTextCfg(context_length=256, embed_unc=True), ts_cfg=TScfg())
# model.load_state_dict(new_model.state_dict(), strict=False)
# model.lock_image_tower()
# model.lock_text_tower()

test_input = torch.randn(2, 7, 3, 224, 224)  # Batch of 2, sequence length of 5, 3 channels, 224x224 images
text_input = torch.ones(2, 256).int()  # Batch of 2, sequence length of 77, embedding dimension of 768
print(text_input.shape)
mask = torch.ones(16, 7, 7)

# transformer = TSTransformer(input_dim=768, width=768, layers=4, heads=8)
# output = transformer(test_input)
# print(output.shape)

# for param in model.parameters():
#     param.requires_grad = False

# # Unfreeze only vision_seq_layers parameters
# for param in model.vision_seq_layers.parameters():
#     param.requires_grad = True

#     # Check all trainable parameters
# print("Trainable parameters:")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"  {name}: {param.shape}")

# exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
# include = lambda n, p: not exclude(n, p)

# named_parameters = list(model.named_parameters())
# gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
# rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

# print("\nGain or bias parameters:", sum(p.numel() for p in gain_or_bias_params))
# print("Rest of the parameters:", sum(p.numel() for p in rest_params))

# print("\nTotal trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
# print("Total parameters:", sum(p.numel() for p in model.parameters()))

output = model(image_seq=test_input, text=text_input, mask=mask)

print(output['image_features']['mean'].shape, output['image_features']['std'].shape)
from src.train.data import ImageSequenceInstructionDataset, collate_fn_image_sequence
from torch.utils.data import DataLoader
import json
import tqdm

dataset = ImageSequenceInstructionDataset(
    image_path='/home/user/intorains/Matterport3D_O',
    json_path='/home/user/intorains/annotations/R2R_train_enc.json',)

for i in tqdm.tqdm(range(len(dataset))):
    dataset.__getitem__(i)
# sampler = None

# # from torch.utils.data.distributed import DistributedSampler
# # sampler = DistributedSampler(dataset)
# is_train = True
# shuffle = True
# dataloader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=shuffle,
#         num_workers=4,
#         pin_memory=True,
#         sampler=sampler,
#         collate_fn=collate_fn_image_sequence,
#         drop_last=is_train,
#     )

# print(len(dataloader))

# with open('/home/user/intorains/annotations/R2R_train_enc.json', 'r') as f:
#     data = json.load(f)

# print(len(data))
# print(data[0])
# for item in data:
#     print(item)
#     assert 'path' in item, "JSON item must contain 'path'"
#     assert 'instructions' in item, "JSON item must contain 'instructions'"
#     assert len(item['instructions']) == 3, "Each item must have exactly 3 instructions"
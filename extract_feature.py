import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.nn import functional as F
from vcsl import VideoFramesDataset
from torch.utils.data import DataLoader
from transformers import ViTMAEModel

from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

class MAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self.fc = torch.nn.Linear(768, 512)
        ckpt = torch.load("./mae_512.ckpt")
        self.backbone.load_state_dict(ckpt['state_dict'], strict=False)
        print("backbone(MAE) checkpoint loaded")
        self.fc.load_state_dict(ckpt['state_dict'], strict=False)
        print("linear checkpoint loaded")
        
    def forward(self, x):
        x = self.backbone(x)
        x = x['last_hidden_state'][:, 0, :]
        x = self.fc(x)
        x = F.normalize(x)
        return x

# the frames_all.csv must contain columns described in Frame Sampling
# df = pd.read_csv("./data/VSC/train_reference_metadata.csv")
df = pd.read_csv("./data/VSC/train_query_metadata.csv")

# data_list = df[['uuid', 'path', 'frame_count']].values.tolist()
data_list = df[['video_id', 'duration_sec']].values.tolist()

data_transforms = [
    lambda x: x.convert('RGB'),
    transforms.Resize((224, 224)),
    lambda x: np.array(x)[:, :, ::-1]
]

dataset = VideoFramesDataset(data_list,
                             id_to_key_fn=VideoFramesDataset.build_image_key,
                             transforms=data_transforms,
                             root='/nfs_shared_/vsc2022_data_frame/train/',
                             store_type="local")

loader = DataLoader(dataset, collate_fn=lambda x: x,
                    batch_size=64,
                    num_workers=8)

model = MAE().cuda()

batch_bar = tqdm(total=len(loader), dynamic_ncols=True, leave=False, position=0, desc='feature save to numpy...')

# load data as the batch size
GREEN, RESET = '\033[32m', '\033[0m'
current_video_id, current_video_feature = '', torch.empty((0,512)).cuda()
for loop, batch_data in enumerate(loader):
    # print(f"--------------- Loop #{loop} ---------------")
    # batch data: List[Tuple[str, int, np.ndarray]]

    # video_ids:('R100001', 'R100001', ... ,  'R100001', 'R100002', 'R100002')
    # frame_ids:(80, 81, ... , 89, 0, 1)
    video_ids, frame_ids, images = zip(*batch_data)
    # print(f"video_ids:{video_ids}\nframe_ids:{frame_ids}")

    if current_video_id == '':
        current_video_id = video_ids[0]
        # print(current_video_feature)
        
    # for torch models, transform the images to a Tensor | shape(B, H, W, C)
    images = torch.Tensor(images)  
    # preprocess the images if necessary and use your model to do the inference
    images = images.permute(0, 3, 1, 2)
    images = images.cuda()

    # compute features from the model
    with torch.no_grad():
        features = model(images)
    # print(f'Features Info.\nType:{type(features)}, Size:{len(features[0])}, {len(features)}\n\n')

    # save frame features according to each video size, even for fixed batch sizes
    for i in range(len(video_ids)):
        video_id, feature = video_ids[i], features[i]
        # If next video_id : save features up to that point and initialize varaiables to the next video.
        if current_video_id != video_id:   
            # print(f"saving '{GREEN + current_video_id + RESET}' with NPY format ...\n{current_video_feature}")
            np.save(f"./feature_test/{current_video_id}.npy", current_video_feature.cpu().numpy())
            # print(f"file saved with {GREEN + str(len(current_video_feature)) + RESET} length\n")
            current_video_id = video_id
            current_video_feature = torch.empty((0,512)).cuda()
            current_video_feature = torch.vstack((current_video_feature, feature))
        else:
            current_video_feature = torch.vstack((current_video_feature, feature))
    batch_bar.update()
batch_bar.close()


    # while current_video_id != tmp_video_id:
    #     tmp_video_id = video_ids[idx]
    #     if 0 in frame_ids and (i != 0):
    #     idx = frame_ids.index(0)
    #     _feature = torch.vstack((buffer, feature[:idx]))
    #     print(_feature)
    #     np.save(f"./feature_test/{video_ids[0]}.npy", _feature.cpu().numpy())
    #     print(f"file saved with {len(_feature)} length")
    #     buffer = feature[idx:]
    # else:
    #     buffer = feature

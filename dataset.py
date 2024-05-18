# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.
import torch
from torch.utils.data import Dataset
from pathlib import Path
from skimage import color, io
import os
import zipfile
import re
import pandas as pd
import requests

label_dict = {
    'void':         (0, 0, 0), 
    'flat':         (128, 64, 128),
    'construction': (70, 70, 70), 
    'object':       (153, 153, 153), 
    'nature':       (107, 142, 35), 
    'sky':          (70, 130, 180), 
    'human':        (220, 20, 60), 
    'vehicle':      (0, 0, 142)
    }
reverse_label_dict = {v:k for k, v in label_dict.items()}
labels = dict(zip(label_dict.keys(), range(0, 8)))

def extract_seg_dataset(data_path):
    PATH = Path(data_path)
    if not PATH.exists():
        PATH.mkdir(parents=True, exist_ok=True)
    FILENAME = "./data_seg_public.zip"
    zip_file_path = Path(FILENAME)

    #assert zip_file_path.exists()
    # if not (zip_file_path).exists():

    #     print(f'downloading ... {URL + FILENAME}')
    #     content = requests.get(URL + FILENAME).content
    #     zip_file_path.open("wb").write(content)
        
    img_path = Path(os.path.join(data_path, 'data_seg_public', 'img'))
    mask_path = Path(os.path.join(data_path, 'data_seg_public', 'mask'))
    if (not img_path.exists()) or (not mask_path.exists()):
        print(f'extracting ... {zip_file_path}')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    return get_dataframe(mask_path, img_path)
    
def get_dataframe(mask_dir, img_dir):
    assert mask_dir.exists()
    assert img_dir.exists()
    samples = []
    for sub_dir in img_dir.iterdir():
        #print('Validating segmentation dataset:')
        for file in sub_dir.iterdir():
            if not '.png' in str(file):
                continue
            name = file.stem
            m = re.search('leftImg8bit', name)
            common_name = name[:m.start()]
            mask_name = common_name + 'gtFine_color' + '.png'
            img_name = common_name + 'leftImg8bit' + '.png'
            mask_path = Path(mask_dir, sub_dir.stem, mask_name)
            img_path =  Path(img_dir, sub_dir.stem, img_name)
            assert mask_path.exists(), mask_path
            assert img_path.exists(), img_path

            samples.append({'img_path':img_path, 'mask_path':mask_path})
        # print(f'removed {removed_samples} out of {n_samples} samples')
    return pd.DataFrame.from_dict(samples)
 
class SampleDataset(Dataset):

    def __init__(self, dataframe, transforms):
        self.dataframe = dataframe
        self.transform = transforms

    def mask_to_labels(self, mask):
        labels_tensor = torch.zeros(mask.size(0), mask.size(1), 8, dtype = torch.float32)
        for i, kl in enumerate(mask):
                for j, triplet in enumerate(kl):
                    data = tuple(triplet.tolist())
                    if data in reverse_label_dict:
                        labels_tensor[i][j][labels[reverse_label_dict[data]]] = 1
        return labels_tensor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        
        # read images
        img = io.imread(sample['img_path']) #.astype(np.uint8)
        mask = io.imread(sample['mask_path'])
                
        # transform img colorspaces
        if len(img.shape) == 2:
            img = gray2rgb(img)
        if img.shape[-1] == 4:
            img = rgba2rgb(img)
      
            
        # apply transformations
        transformed = self.transform(image=img, mask=mask)
        res_img = transformed['image'].to(dtype = torch.float32)
        res_mask = transformed['mask']
        res_mask = self.mask_to_labels(res_mask)
        res_mask = torch.argmax(res_mask, dim = 2)
        return res_img, res_mask  



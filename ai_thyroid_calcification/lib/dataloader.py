import glob
import numpy as np
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from lib.cfg import *


def get_calcification_data_index():
    # grep .png files in absolute path
    list_image_path = glob.glob(PATH_IMAGE+'*.png')
    list_cal_mask_path = glob.glob(PATH_CALCIFICATION_MASK+'*.png')
    list_lesion_mask_path = glob.glob(PATH_LESION_MASK+'*.png')

    # build absolute path into DataFrame
    df_image = pd.DataFrame({'image_path': list_image_path})
    df_image['id'] = list(map(lambda x: int(re.findall(r'\d+', x)[-1]), df_image['image_path']))
    df_cal_mask = pd.DataFrame({'cal_mask_path': list_cal_mask_path})
    df_cal_mask['id'] = list(map(lambda x: int(re.findall(r'\d+', x)[-1]), df_cal_mask['cal_mask_path']))
    df_lesion_mask = pd.DataFrame({'lesion_mask_path': list_lesion_mask_path})
    df_lesion_mask['id'] = list(map(lambda x: int(re.findall(r'\d+', x)[-1]), df_lesion_mask['lesion_mask_path']))

    df_image = df_image[['id', 'image_path']]
    df_data_index = df_image.merge(df_cal_mask, how='inner').merge(df_lesion_mask, how='inner')
    
    return df_data_index


class DatasetThyroid(Dataset):
    def __init__(
        self, 
        df_data, 
        image_transform=None, 
        mask_transform=None,
        random_horizontal_flip=True,
    ):
        self.df_data = df_data
        self.image_transform = image_transform 
        self.mask_transform = mask_transform
        self.random_horizontal_flip = random_horizontal_flip
        
    def __getitem__(self, index):
        row = self.df_data.iloc[index]
        img_id = row.id
        image = Image.open(row.image_path).resize(IMAGE_RESIZE_DIM)
        cal_mask = Image.open(row.cal_mask_path).resize(IMAGE_RESIZE_DIM)
        lesion_mask = Image.open(row.lesion_mask_path).resize(IMAGE_RESIZE_DIM)
        
        if self.random_horizontal_flip & (np.random.random() > 0.5):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            cal_mask = cal_mask.transpose(Image.FLIP_LEFT_RIGHT)
            lesion_mask = lesion_mask.transpose(Image.FLIP_LEFT_RIGHT)
            
        if self.image_transform is not None:
            image = self.image_transform(image)
        
        if self.mask_transform is not None:
            cal_mask = self.mask_transform(cal_mask)
            lesion_mask = self.mask_transform(lesion_mask)
            
        return image, cal_mask, lesion_mask, img_id

    def __len__(self):
        return len(self.df_data)
    

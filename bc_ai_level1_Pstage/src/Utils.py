from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import Dataset, Subset, random_split
import pandas as pd
import torch 
from torchvision import transforms
from torchvision.transforms import *
import cv2
from PIL import Image
# set imgae directory path 
class Utils:
    
    @classmethod
    def makeimgdirs(cls):
        path = Path('input/data/train/images')
        img_dirs = [str(x) for x in list(path.glob('*')) if '._' not in str(x)]
        img_dirs = np.array(img_dirs)
        return img_dirs
    
    @classmethod
    def makeKfolds(cls) : 
        path = Path('input/data/train/images')
        img_dirs = [str(x) for x in list(path.glob('*')) if '._' not in str(x)]
        img_dirs = np.array(img_dirs)
        semi_labels = []
        for img_dir in img_dirs:
            if 'female' in img_dir:
                g = 1
            else : 
                g = 0
            age = int(img_dir.split('_')[3][:2])
            if age < 30:
                a = 0
            elif age < 58:
                a = 1
            else:
                a = 2
            semi_labels.append(3*g + a)
        semi_labels = np.array(semi_labels)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = []
        for t,v in skf.split(img_dirs, semi_labels):
            folds.append({'train':t, 'valid':v})
        return folds

    
    @classmethod
    def labeling(cls, img_path):
        if 'normal' in img_path:
            m = 2
        elif 'incorrect_mask' in img_path:
            m = 1
        else:
            m = 0
            
        if 'female' in img_path:
            g = 1
        else : 
            g = 0
            
        age = int(img_path.split('_')[3][:2])
        if age < 30:
            a = 0
        elif age < 58:
            a = 1
        else:
            a = 2
        return 6 *m + 3*g + a
    
    @classmethod
    def setlabelweights(cls):
        path = Path('../input/data/train/images')
        img_paths = [str(x) for x in list(path.glob('*/*')) if '._' not in str(x)]
        img_labels = list(map(cls.labeling,img_paths))
        label_weights = pd.Series(img_labels).value_counts().sort_index()
        label_weights = torch.FloatTensor([1-(x/sum(label_weights)) for x in label_weights])
        label_weights = label_weights.to('cpu')
        
    @classmethod
    def transforms(cls, resize):
        train_transform = transforms.Compose([
            ToPILImage(),
            Resize(resize, Image.BILINEAR),
            #CenterCrop((300, 256)),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        
        valid_transform = transforms.Compose([
            ToPILImage(),
            #CenterCrop((300, 256)),
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            
            
        return train_transform, valid_transform
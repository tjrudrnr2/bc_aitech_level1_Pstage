import argparse
import os
from importlib import import_module
import glob
from tqdm import tqdm
import numpy as np 
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset, CustomDataset
from model import *

from torchvision import transforms
from torchvision.transforms import *
from PIL import Image

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def Merging_label(train_versions):
    train_version = []
    for version in train_versions.keys():
        train_version.append(version)
    
    mask_csv = pd.read_csv(f'model/{args.name}/{train_version[0]}_{args.name}_output.csv')
    age_csv = pd.read_csv(f'model/{args.name}/{train_version[1]}_{args.name}_output.csv')
    gender_csv = pd.read_csv(f'model/{args.name}/{train_version[2]}_{args.name}_output.csv')

    total_csv = pd.DataFrame()
    total_csv['ImageID'] = mask_csv['ImageID']
    total_csv['ans'] = mask_csv['ans']*6 + gender_csv['ans']*3 + age_csv['ans']
    
    total_csv.to_csv(f'model/{args.name}/total_{args.name}_output.csv', index=False)
    
    return "Merged Labels"


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args, train_version, num_classes):
    """
    """
    # 1. cuda settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 2. make dataset
    img_root = os.path.join(data_dir, 'images') # 'input/data/eval/images'
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    
    img_paths = [os.path.join(img_root,img_id) for img_id in info.ImageID]
    valid_transform = transforms.Compose([
            ToPILImage(),
            Resize(args.resize, Image.BILINEAR),
            #CenterCrop((320, 256)),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset = CustomDataset(img_paths, train_version=train_version, training=False)
    dataset.set_transform(valid_transform)
    loader = loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    model_pred_lst = []
    for best_model in glob.glob(f'model/exp/{train_version}_*.pth'):   # model/{args.name}/{train_version}_*{args.name}.pth'): # 경로 수정할 것
        #model = MyEfficientNet(num_classes=num_classes)
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.load_state_dict(torch.load(best_model))
        model.to(device)
        model.eval()
        
        small_pred_lst = []
        with tqdm(loader, total=loader.__len__(), unit='batch') as test_bar:
            for sample in test_bar:
                imgs = sample['image'].float().to(device)
                pred = model(imgs)
                pred = pred.cpu().detach().numpy()
                small_pred_lst.extend(pred)
        model_pred_lst.append(np.array(small_pred_lst)[...,np.newaxis])
    
    print("Calculating inference results..")

    info['ans'] = np.argmax(np.mean(np.concatenate(model_pred_lst, axis=2), axis=2), axis=1)
    info.to_csv(f'model/{args.name}/{train_version}_{args.name}_output.csv', index=False)
    print(f'Inference Done!')

    del model, imgs
    torch.cuda.empty_cache()
    
class InCfg:
    batch_size = 64
    resize = (224,224)
    model = "VIT"
    name = "exp"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=InCfg.batch_size, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=InCfg.resize, help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default=InCfg.model, help='model type (default: BaseModel)')
    parser.add_argument('--name', type=str, default=InCfg.name, help='folder name')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
        
    train_version = {'mask':3, 'age':3, 'gender':2}
    for version_name, num_classes in train_version.items():
        print(f'Training "{version_name}" label')
        inference(data_dir, model_dir, output_dir, args, version_name, num_classes)

    print(Merging_label(train_version))
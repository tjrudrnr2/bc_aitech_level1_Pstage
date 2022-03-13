import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from loss import create_criterion
from model import *
# from adamp import AdamP
import wandb
from Utils import Utils
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from imbalance import ImbalancedDatasetSampler

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

# ============ 위에까지는 주어진 함수들. 바꾼 것이 없습니다. 

def train(data_dir, model_dir, args, train_version, num_classes):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    # -- 1. cpu settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- 2. fold settings
    folds = Utils.makeKfolds()
    img_dirs = Utils.makeimgdirs()
    train_transform, valid_transform = Utils.transforms(args.resize)
    
    # -- 3. criterion(=loss function)
    label_weights = Utils.setlabelweights()
    criterion = torch.nn.CrossEntropyLoss(weight=label_weights).to(device)
#        기존 코드 : criterion = create_criterion(args.criterion)  # default: cross_entropy
    
    
    for fold in range(args.fold_num):
        print(f"now training fold is {fold}")
        valid_loss_min = 3
        early_stop_cnt = 0
        
        # -- 4. dataset
        train_img_paths, valid_img_paths = [], []
        for train_dir in img_dirs[folds[fold]['train']]:
            train_img_paths.extend(glob.glob(train_dir+ '/*'))
        for valid_dir in img_dirs[folds[fold]['valid']]:
            valid_img_paths.extend(glob.glob(valid_dir+ '/*'))
        train_dataset = CustomDataset(train_img_paths, train_version=train_version, training=True)
        valid_dataset = CustomDataset(valid_img_paths, train_version=train_version, training=True)
        
        # -- 5. augmentation (Transform)
        transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
        test_transform = transform_module(resize=args.resize,mean=train_dataset.mean,std=train_dataset.std,)
        train_dataset.set_transform(train_transform)
        valid_dataset.set_transform(valid_transform)
        
        # -- 6. dataloader
        train_loader = DataLoader(dataset=train_dataset, sampler=ImbalancedDatasetSampler(train_dataset) , batch_size=args.batch_size, num_workers=0)
        #train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size//4, shuffle=True, num_workers=0)

        # -- 7. model
        name = "exp"
        """
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        retrain_model = glob.glob(f'model/*/{train_version}_{fold}fold*{name}.pth')
        print(retrain_model)
        """
        best_valid_loss = np.inf
        """
        if retrain_model :
            _, ep, loss, _ = ''.join(retrain_model).split('_')
            ep = int(ep.replace('epoch', ''))
            best_valid_loss = float(loss)
            model = model_module(num_classes=num_classes)
       #     model.EFF._dropout = torch.nn.Dropout(p=0.7, inplace=False)
            model.load_state_dict(torch.load(retrain_model[0]))
            model.to(device)
        else : 
            ep = 0
            model = model_module(num_classes=num_classes)
        #    model.EFF._dropout = torch.nn.Dropout(p=0.7, inplace=False)
            model = model.to(device)
#         model = torch.nn.DataParallel(model)
        """
        ep=0
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model = model.to(device)
        
        # -- 8. optimizer & scheduler
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
        

        # -- 9. logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_val_acc = 0
        early_stop_cnt = 0
        for epoch in range(args.epochs):
            if epoch < ep : 
                continue
            
            model.train()
            # -- 10. train loop in epoch
            train_bar = tqdm(train_loader, total=train_loader.__len__(), unit='batch')
            train_f1 = []
            train_loss = []
            for sample in train_bar:
                train_bar.set_description(f'EpochT {epoch+1}/{args.epochs}')
                imgs = sample['image'].float().to(device)
                labels = sample['label'].long().to(device)

                optimizer.zero_grad()
                outs = model(imgs)
                loss = criterion(outs, labels)
                loss.backward()
                optimizer.step()

                train_f1.append(f1_score(labels.cpu().detach().float(), torch.argmax(outs.cpu().detach(), 1), average='macro'))
                train_loss.append(loss.item())
                train_bar.set_postfix(t_f1=np.mean(train_f1), loss=np.mean(train_loss))
            wandb.log({
                    f"Train F1" : train_f1[-1], 
#                         f"Train ACC":train_acc, 
                    f"Train Loss":train_loss[-1], 
#                         f"Learning Rate":current_lr,
              })
            scheduler.step()
            
            
            # -- 11. valid loop in epoch
            valid_bar = tqdm(valid_loader, total=valid_loader.__len__(), unit='batch') 
            valid_f1 = []
            valid_loss = []
            for sample in valid_bar:
                valid_bar.set_description(f"EpochV {epoch+1}/{args.epochs}")
                imgs = sample['image'].float().to(device)
                labels = sample['label'].long().to(device)

                model.eval()
                optimizer.zero_grad()
                with torch.no_grad():
                    outs = model(imgs)
                    loss = criterion(outs, labels)

                valid_f1.append(f1_score(labels.cpu().detach().float(), torch.argmax(outs.cpu().detach(), 1), average='macro'))
                valid_loss.append(loss.item())
                valid_bar.set_postfix(v_f1=np.mean(valid_f1), loss=np.mean(valid_loss))
            wandb.log({f"Valid F1":valid_f1[-1], f"Train Loss":valid_loss[-1]})
                    
            # -- 12. update by checking valid_loss 
            if np.mean(valid_loss) < best_valid_loss:
                best_valid_loss = np.mean(valid_loss)
                early_stop_cnt = 0
                for f in glob.glob(f'model/{name}/{train_version}_{fold}fold_*{name}.pth'):
                    open(f, 'w').close()
                    os.remove(f)
                torch.save(model.state_dict(), f'model/{name}/{train_version}_{fold}fold_{epoch+1}epoch_{np.mean(valid_loss):2.4f}_{name}.pth')
            else : 
                early_stop_cnt += 1
                if early_stop_cnt >= 5:
                    break 
                    
    # empty cuda at end of train()
    del model, imgs, labels
    torch.cuda.empty_cache()

class Cfg:
    augmentation='CCRAugmentation'
    batch_size=64
    criterion='cross_entropy'
    model_dir='./model'
    data_dir='../input/data/train/images'
    dataset='MaskBaseDataset'
    epochs=10
    log_interval=20
    lr=0.0001 # 수정 
    lr_decay_step=20 
    model='VIT'
    name='exp'
    optimizer='Adam'
    resize=[224, 224]
    seed=42
    val_ratio=0.2
    valid_batch_size=1000
    fold_num = 5
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=Cfg.seed, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=Cfg.epochs, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default=Cfg.dataset, help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default=Cfg.augmentation, help='data augmentation type (default:  BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=Cfg.resize, help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=Cfg.batch_size, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=Cfg.valid_batch_size, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default=Cfg.model, help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default=Cfg.optimizer, help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=Cfg.lr, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=Cfg.val_ratio, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default=Cfg.criterion, help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=Cfg.lr_decay_step, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=Cfg.log_interval, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default=Cfg.name, help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--fold_num', type=int, default=Cfg.fold_num, help='fold_num')
    parser.add_argument('--data_dir', type=str, default=Cfg.data_dir)
    parser.add_argument('--model_dir', type=str, default=Cfg.model_dir)
    
    # Container environment
#     parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
#     parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    run = wandb.init(project="Sua", name="Sua_v02", entity="8ollow")
    wandb.config.update(args)
    
    data_dir = args.data_dir
    model_dir = args.model_dir
    
    """
    train_version = {'age':3}
    for version_name, num_classes in train_version.items():
        print(f'Training "{version_name}" label')
        train(data_dir, model_dir, args, version_name, num_classes)
    """
    
    train_version = {'mask':3, 'age':3, 'gender':2}
    for version_name, num_classes in train_version.items():
        print(f'Training "{version_name}" label')
        train(data_dir, model_dir, args, version_name, num_classes)
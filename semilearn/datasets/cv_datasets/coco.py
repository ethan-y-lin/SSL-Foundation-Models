# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import math
import numpy as np
from PIL import Image

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation


# Per-dataset path conventions and category_id indexing
DATASET_CONFIGS = {
    'cub': {
        'subdir': 'cub',
        'train_ann': os.path.join('metadata', 'annotations', 'train', 'train.json'),
        'train_img_dir': os.path.join('data', 'images', 'train'),
        'val_ann': os.path.join('data', 'annotations', 'val', 'annotations.json'),
        'val_img_dir': os.path.join('data', 'images', 'val'),
        'test_ann': os.path.join('test', 'annotations', 'annotations.json'),
        'test_img_dir': os.path.join('test', 'images'),
        'cat_id_offset': -1,  # 1-indexed categories -> 0-indexed
    },
    'fungi': {
        'subdir': 'fungi',
        'train_ann': os.path.join('metadata', 'annotations', 'train', 'train.json'),
        'train_img_dir': os.path.join('data', 'images', 'train'),
        'val_ann': os.path.join('data', 'annotations', 'val', 'val.json'),
        'val_img_dir': os.path.join('data', 'images', 'val'),
        'test_ann': os.path.join('test', 'annotations', 'test.json'),
        'test_img_dir': os.path.join('test', 'images'),
        'cat_id_offset': 0,
    },
    'resisc45': {
        'subdir': 'resisc45',
        'train_ann': os.path.join('metadata', 'annotations', 'train', 'train.json'),
        'train_img_dir': os.path.join('data', 'images', 'train'),
        'val_ann': os.path.join('data', 'annotations', 'val', 'val.json'),
        'val_img_dir': os.path.join('data', 'images', 'val'),
        'test_ann': os.path.join('test', 'annotations', 'test.json'),
        'test_img_dir': os.path.join('test', 'images'),
        'cat_id_offset': 0,
    },
    'sun397': {
        'subdir': 'sun397',
        'train_ann': os.path.join('metadata', 'annotations', 'train', 'train.json'),
        'train_img_dir': os.path.join('data', 'images', 'train'),
        'val_ann': os.path.join('data', 'annotations', 'val', 'val.json'),
        'val_img_dir': os.path.join('data', 'images', 'val'),
        'test_ann': os.path.join('test', 'annotations', 'test.json'),
        'test_img_dir': os.path.join('test', 'images'),
        'cat_id_offset': -1,
    },
    
}


def _load_coco_ssl(args, alg, dataset_name, num_labels, num_classes, data_dir, include_lb_to_ulb, cfg):
    """Generic COCO-format SSL dataset loader used by CUB, Fungi, and RESISC45."""
    data_dir = os.path.join(data_dir, cfg['subdir'])
    cat_offset = cfg['cat_id_offset']

    # --- Parse train annotations (COCO format) ---
    train_json_path = os.path.join(data_dir, cfg['train_ann'])
    with open(train_json_path, 'r') as f:
        train_meta = json.load(f)

    imgid_to_catid = {}
    for ann in train_meta['annotations']:
        imgid_to_catid[ann['image_id']] = ann['category_id'] + cat_offset

    bare_to_info = {}
    all_train_paths = []
    all_train_targets = []
    for img_entry in train_meta['images']:
        img_id = img_entry['id']
        full_path = os.path.join(data_dir, cfg['train_img_dir'], img_entry['file_name'])
        cat_id = imgid_to_catid[img_id]
        bare_name = os.path.basename(img_entry['file_name'])
        bare_to_info[bare_name] = (full_path, cat_id)
        all_train_paths.append(full_path)
        all_train_targets.append(cat_id)

    # --- Parse partition file to determine labeled set (k-shot format) ---
    k = num_labels
    data_setting = getattr(args, 'data_setting', 'setA')
    data_seed = getattr(args, 'data_seed', 0)
    partition_path = os.path.join(data_dir, data_setting, f'k{k}', f'seed{data_seed}', 'annotations', 'train.json')
    with open(partition_path, 'r') as f:
        partition_meta = json.load(f)

    lb_bare_names = set()
    for img_entry in partition_meta['images']:
        lb_bare_names.add(os.path.basename(img_entry['file_name']))

    lb_data = []
    lb_targets = []
    for bare_name in lb_bare_names:
        full_path, cat_id = bare_to_info[bare_name]
        lb_data.append(full_path)
        lb_targets.append(cat_id)

    sorted_indices = np.argsort(lb_data)
    lb_data = [lb_data[i] for i in sorted_indices]
    lb_targets = [lb_targets[i] for i in sorted_indices]

    # --- Build unlabeled set ---
    if include_lb_to_ulb:
        ulb_data = list(all_train_paths)
        ulb_targets = list(all_train_targets)
    else:
        ulb_data = []
        ulb_targets = []
        for path, target in zip(all_train_paths, all_train_targets):
            bare_name = os.path.basename(path)
            if bare_name not in lb_bare_names:
                ulb_data.append(path)
                ulb_targets.append(target)

    # --- Parse val annotations (COCO format) ---
    val_json_path = os.path.join(data_dir, cfg['val_ann'])
    with open(val_json_path, 'r') as f:
        val_meta = json.load(f)

    val_imgid_to_catid = {}
    for ann in val_meta['annotations']:
        val_imgid_to_catid[ann['image_id']] = ann['category_id'] + cat_offset

    eval_data = []
    eval_targets = []
    for img_entry in val_meta['images']:
        full_path = os.path.join(data_dir, cfg['val_img_dir'], img_entry['file_name'])
        cat_id = val_imgid_to_catid[img_entry['id']]
        eval_data.append(full_path)
        eval_targets.append(cat_id)

    # Preload val images into numpy arrays for speed
    eval_data_np = []
    for p in eval_data:
        img = Image.open(p).convert("RGB")
        eval_data_np.append(np.array(img))
    eval_data = eval_data_np

    # --- Parse test annotations (COCO format) ---
    test_json_path = os.path.join(data_dir, cfg['test_ann'])
    with open(test_json_path, 'r') as f:
        test_meta = json.load(f)

    test_imgid_to_catid = {}
    for ann in test_meta['annotations']:
        test_imgid_to_catid[ann['image_id']] = ann['category_id'] + cat_offset

    test_data = []
    test_targets = []
    for img_entry in test_meta['images']:
        full_path = os.path.join(data_dir, cfg['test_img_dir'], img_entry['file_name'])
        cat_id = test_imgid_to_catid[img_entry['id']]
        test_data.append(full_path)
        test_targets.append(cat_id)

    # Preload test images into numpy arrays for speed
    test_data_np = []
    for p in test_data:
        img = Image.open(p).convert("RGB")
        test_data_np.append(np.array(img))
    test_data = test_data_np

    # --- Print label distribution summary ---
    lb_count = [0] * num_classes
    ulb_count = [0] * num_classes
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print(f"{dataset_name} labeled: {len(lb_data)}, unlabeled: {len(ulb_data)}, eval: {len(eval_data)}, test: {len(test_data)}")
    print(f"lb per-class min: {min(lb_count)}, max: {max(lb_count)}")

    # --- Transforms (ImageNet-style) ---
    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10, exclude_color_aug=getattr(args, 'exclude_color_aug', False)),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    # --- Build dataset instances ---
    lb_dset = COCOSSLDataset(alg, lb_data, lb_targets, num_classes,
                             transform=transform_weak, is_ulb=False,
                             strong_transform=None, onehot=False)
    ulb_dset = COCOSSLDataset(alg, ulb_data, ulb_targets, num_classes,
                              transform=transform_weak, is_ulb=True,
                              strong_transform=transform_strong, onehot=False)
    eval_dset = COCOSSLDataset(alg, eval_data, eval_targets, num_classes,
                               transform=transform_val, is_ulb=False,
                               strong_transform=None, onehot=False)
    test_dset = COCOSSLDataset(alg, test_data, test_targets, num_classes,
                               transform=transform_val, is_ulb=False,
                               strong_transform=None, onehot=False)

    return lb_dset, ulb_dset, eval_dset, test_dset


class COCOSSLDataset(BasicDataset):
    def __sample__(self, idx):
        data = self.data[idx]
        if isinstance(data, str):
            img = Image.open(data).convert("RGB")
        else:
            img = Image.fromarray(data)
        target = self.targets[idx]
        return img, target


# Keep backward compat alias
COCODataset = COCOSSLDataset

def get_coco(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    return _load_coco_ssl(args, alg, name, num_labels, num_classes, data_dir, include_lb_to_ulb, DATASET_CONFIGS[name])
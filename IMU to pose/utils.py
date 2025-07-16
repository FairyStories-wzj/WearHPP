import os, glob, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# ==== 工具函数 ====
def reorder_imu(df):
    id_to_name = ['right hand', 'right pocket', 'glasses', 'left pocket', 'left hand']
    device_names = {
        'right hand':    ['right hand_加速度X', 'right hand_加速度Y', 'right hand_加速度Z', 'right hand_角速度X', 'right hand_角速度Y', 'right hand_角速度Z'],
        'right pocket':  ['right pocket_加速度X', 'right pocket_加速度Y', 'right pocket_加速度Z', 'right pocket_角速度X', 'right pocket_角速度Y', 'right pocket_角速度Z'],
        'glasses':       ['glasses_加速度X', 'glasses_加速度Y', 'glasses_加速度Z', 'glasses_角速度X', 'glasses_角速度Y', 'glasses_角速度Z'],
        'left pocket':   ['left pocket_加速度X', 'left pocket_加速度Y', 'left pocket_加速度Z', 'left pocket_角速度X', 'left pocket_角速度Y', 'left pocket_角速度Z'],
        'left hand':     ['left hand_加速度X', 'left hand_加速度Y', 'left hand_加速度Z', 'left hand_角速度X', 'left hand_角速度Y', 'left hand_角速度Z'],
    }
    select_cols = []
    for device in id_to_name:
        select_cols += device_names[device]
    for col in select_cols:
        if col not in df.columns:
            df[col] = 0.0
    imu_data = df[select_cols].values.astype(np.float32)
    imu_arr = imu_data.reshape(-1, 5, 6)
    return imu_arr

import numpy as np

def generate_modality_mask(mask_probs):
    arr = np.array([mask_probs[name] for name in ['right hand', 'right pocket', 'glasses', 'left pocket', 'left hand']])
    mask = np.random.rand(5) > arr
    if mask.any():
        return mask
    else:
        return generate_modality_mask(mask_probs)




def collate_fn(batch):
    L = batch[0][0].shape[0]
    B = len(batch)
    n_valid_list = [b[0].shape[1] for b in batch]
    n_valid_max = max(n_valid_list)
    imu_batch = torch.zeros(B, L, n_valid_max, 6)
    keep_batch = torch.full((B, n_valid_max), -1, dtype=torch.long)
    pose_batch = torch.zeros(B, L, 15, 3)
    for i, (imu, pose, keep) in enumerate(batch):
        n_valid = imu.shape[1]
        imu_batch[i, :, :n_valid, :] = imu
        keep_batch[i, :n_valid] = keep
        pose_batch[i] = pose
    return imu_batch, pose_batch, keep_batch, n_valid_list
# ==== 稳定loss和评估 ====
def pose_loss(pred, gt):
    xy_gt = gt[..., :2]
    conf  = gt[..., 2]
    mse = ((pred - xy_gt) ** 2).sum(dim=-1)
    conf_sum = conf.sum()
    if conf_sum < 1e-6:  # 没有有效标签
        return torch.tensor(0.0, device=pred.device)
    weighted = (mse * conf).sum() / (conf_sum + 1e-8)
    if torch.isnan(weighted):  # 防nan
        weighted = torch.tensor(0.0, device=pred.device)
    return weighted

def mpjpe(pred, gt, conf=None):
    err = ((pred - gt) ** 2).sum(dim=-1).sqrt()
    if conf is not None:
        conf_sum = conf.sum()
        if conf_sum < 1e-6:
            return 0.0
        mpjpe = (err * conf).sum() / (conf_sum + 1e-8)
    else:
        mpjpe = err.mean()
    return mpjpe.item()


import os, glob, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import reorder_imu, generate_modality_mask,collate_fn, pose_loss, mpjpe
from imu_model import IMU2PoseNet_Fusion,DeviceFeatureTransformer,DeviceBiLSTMEncoder,IMUPoseSeqDataset
# ==== 参数区 ====
window_size = 75
stride = 5
batch_size = 8
lr = 1e-4
hidden_dim = 256
dropout = 0.1
num_epochs = 30
mask_probs = {
    'right hand': 0.1, 'right pocket': 0.1, 'glasses': 0.1, 'left pocket': 0.1, 'left hand': 0.1
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==== 数据加载 ====
train_set = IMUPoseSeqDataset(
    imu_dir="E:\\Xrf2\\imu\\train",
    pose_dir="E:\\Xrf2\\train",
    window_size=window_size, stride=stride, mask_probs=mask_probs)
test_set = IMUPoseSeqDataset(
    imu_dir="E:\\Xrf2\\imu\\test",
    pose_dir="E:\\Xrf2\\test",
    window_size=window_size, stride=stride, mask_probs=mask_probs)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

# ==== 训练 ====
model = IMU2PoseNet_Fusion(pose_dim=15, feat_dim=512, hidden_dim=hidden_dim, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
import datetime

save_dir = "E:\\Python Project\\WearHPP\\IMU to pose\\results"
os.makedirs(save_dir, exist_ok=True)

best_val_mpjpe = float('inf')
save_tag = f"ws{window_size}_st{stride}_bs{batch_size}_lr{lr}_hd{hidden_dim}_dr{dropout}_mask{'-'.join([f'{k}{v}' for k,v in mask_probs.items()])}"
start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imu_batch, pose_batch, keep_batch, n_valid_list in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        imu_batch = imu_batch.float().to(device)
        pose_batch = pose_batch.float().to(device)
        pred = model(imu_batch, keep_batch, n_valid_list)
        print(pred.shape)
        loss = pose_loss(pred, pose_batch)
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: loss is nan or inf, skip this batch!")
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"\nEpoch {epoch} Train loss: {total_loss/len(train_loader):.5f}")

    # 验证
    model.eval()
    total_mpjpe = 0
    with torch.no_grad():
        for imu_batch, pose_batch, keep_batch, n_valid_list in tqdm(test_loader, desc="Val"):
            imu_batch = imu_batch.float().to(device)
            pose_batch = pose_batch.float().to(device)
            pred = model(imu_batch, keep_batch, n_valid_list)
            xy_gt = pose_batch[..., :2]
            conf  = pose_batch[..., 2]
            err = mpjpe(pred.cpu(), xy_gt.cpu(), conf.cpu())
            total_mpjpe += err
        val_mpjpe = total_mpjpe / len(test_loader)
        print(f"Val MPJPE: {val_mpjpe:.5f}")

        # ----------- 保存最佳模型 -------------
        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            model_path = os.path.join(save_dir, f"best_{save_tag}_{start_time}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_mpjpe": best_val_mpjpe,
                "window_size": window_size,
                "stride": stride,
                "batch_size": batch_size,
                "lr": lr,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "mask_probs": mask_probs,
            }, model_path)
            print(f"*** [Model Saved] *** epoch={epoch}, mpjpe={val_mpjpe:.5f}, path={model_path}")

print("训练完成！最佳模型保存在", model_path)


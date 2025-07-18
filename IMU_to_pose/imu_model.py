import os, glob, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from IMU_to_pose.utils import reorder_imu, generate_modality_mask,collate_fn, pose_loss, mpjpe

# ==== 数据集 ====
class IMUPoseSeqDataset(Dataset):
    def __init__(self, imu_dir, pose_dir, window_size=75, stride=5, mask_probs=None):
        self.imu_files = sorted(glob.glob(os.path.join(imu_dir ,"*.csv")))
        self.pose_files = sorted(glob.glob(os.path.join(pose_dir,"*.csv")))
        self.win = window_size
        self.stride = stride
        self.mask_probs = mask_probs
        self.index = []
        for i in tqdm(range(len(self.imu_files))):
            imu = pd.read_csv(self.imu_files[i])
            pose = pd.read_csv(self.pose_files[i])
            T = min(len(imu), len(pose))
            for t in range(0, T-self.win+1, stride):
                self.index.append((i, t))

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        fi, t0 = self.index[idx]
        imu_df = pd.read_csv(self.imu_files[fi]).iloc[t0:t0+self.win]
        imu_arr = reorder_imu(imu_df)
        msk = generate_modality_mask(self.mask_probs)
        keep = np.where(msk)[0]
        valid_imu = imu_arr[:, keep, :]  # [L, n_valid, 6]
        pose_df = pd.read_csv(self.pose_files[fi]).iloc[t0:t0+self.win]
        pose_seq = [np.array(ast.literal_eval(row), np.float32) for row in pose_df['keypoints']]
        pose_arr = np.stack(pose_seq, axis=0)
        return (
            torch.from_numpy(valid_imu),
            torch.from_numpy(pose_arr),
            torch.tensor(keep)
        )
class IMU2PoseNet_Fusion(nn.Module):
    def __init__(self, pose_dim=15, feat_dim=512, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = DeviceBiLSTMEncoder(input_dim=6, hidden_dim=hidden_dim, output_dim=feat_dim, dropout=dropout)
        self.device_fusion = DeviceFeatureTransformer(feat_dim=feat_dim, num_heads=4, num_layers=2, dropout=dropout)
        self.pose_head = nn.Linear(feat_dim, 2 * pose_dim)

    def forward(self, imu_batch, keep_batch, n_valid_list):
        B, L, n_valid_max, _ = imu_batch.shape
        # === 批量并行所有modal ===
        imu_all = imu_batch.permute(0,2,1,3).reshape(B*n_valid_max, L, 6)
        feat_all = self.encoder(imu_all)  # (B*n_valid_max, L, 512)
        feats = feat_all.reshape(B, n_valid_max, L, 512).permute(0,2,1,3)  # [B, L, n_valid_max, 512]
        # [B, L, n_valid_max, 512] → [B*L, n_valid_max, 512]
        feats = feats.reshape(B*L, n_valid_max, 512)
        key_padding_mask = (keep_batch == -1).unsqueeze(1).expand(B, L, n_valid_max).reshape(B*L, n_valid_max)
        key_padding_mask = key_padding_mask.to(feats.device)
        fused = self.device_fusion(feats, key_padding_mask)
        fused = fused.reshape(B, L, n_valid_max, 512)
        # === 只聚合有效通道 ===
        feat_agg = []
        for b in range(B):
            n_valid = n_valid_list[b]
            valid_indices = (keep_batch[b] != -1).nonzero(as_tuple=True)[0]
            if n_valid == 0:
                # 极端情况，全部丢失
                agg = torch.zeros(L, 512, device=imu_batch.device)
            else:
                sample_feats = fused[b, :, valid_indices, :]  # (L, n_valid, 512)
                agg = sample_feats.mean(dim=1)
            feat_agg.append(agg)
        feat_agg = torch.stack(feat_agg, dim=0)  # (B, L, 512)
        pose_pred = self.pose_head(feat_agg).view(B, L, 15, 2)
        return pose_pred
class DeviceFeatureTransformer(nn.Module):
    def __init__(self, feat_dim=512, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=num_heads, dropout=dropout,
            batch_first=True, dim_feedforward=feat_dim*2,
            norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x, key_padding_mask=None):
        # x: [B*L, n_valid_max, feat_dim]
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return x
# ==== 模型部分（参数共享高效批量版） ====
class DeviceBiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=512, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim*2, output_dim)
    def forward(self, x):  # x: (N, L, 6)
        out, _ = self.bilstm(x)
        out = self.fc(out)
        return out  # (N, L, 512)

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json

from utils import reorder_imu, generate_modality_mask, collate_fn, pose_loss
from imu_model import IMU2PoseNet_Fusion, IMUPoseSeqDataset

def load_pose_zscore_param(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)
        mean = np.array(d["mean"], dtype=np.float32).reshape(15, 3)
        std = np.array(d["std"], dtype=np.float32).reshape(15, 3)
    return mean, std

def plot_pose(gt_pts, pr_pts, edges, title, save_path):
    plt.figure(figsize=(5, 6))
    plt.scatter(gt_pts[:, 0], gt_pts[:, 1], marker='o', label='GT')
    plt.scatter(pr_pts[:, 0], pr_pts[:, 1], marker='x', label='Pred')
    for (a, b) in edges:
        plt.plot([gt_pts[a, 0], gt_pts[b, 0]], [gt_pts[a, 1], gt_pts[b, 1]], 'b--', lw=1)
        plt.plot([pr_pts[a, 0], pr_pts[b, 0]], [pr_pts[a, 1], pr_pts[b, 1]], 'r--', lw=1)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def test_and_visualize(model, test_loader, device, mean_xy, std_xy, out_dir, n_vis_frames=50, conf_thres=0.5):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)
    ]
    all_mpjpe_px, all_mpjpe_mm = [], []

    with torch.no_grad():
        for imu_batch, pose_batch, keep_batch, n_valid_list in tqdm(test_loader, desc="Test+Visualize"):
            imu_batch = imu_batch.float().to(device)
            pose_batch = pose_batch.float().to(device)
            pred = model(imu_batch, keep_batch, n_valid_list)  # (B,L,15,2)
            xy_gt = pose_batch[..., :2]
            conf = pose_batch[..., 2]  # (B,L,15)

            pred_np = pred.cpu().numpy()
            gt_np = xy_gt.cpu().numpy()
            conf_np = conf.cpu().numpy()
            pred_denorm = pred_np * std_xy[None, None, :, :] + mean_xy[None, None, :, :]
            gt_denorm = gt_np * std_xy[None, None, :, :] + mean_xy[None, None, :, :]

            pixel_error = np.linalg.norm(pred_denorm - gt_denorm, axis=-1)  # [B, L, 15]
            neck = gt_denorm[:, :, 1, :]   # [B, L, 2]
            midhip = gt_denorm[:, :, 8, :] # [B, L, 2]
            dist_px = np.linalg.norm(neck - midhip, axis=2)  # [B, L]
            scale = 375.0 / (dist_px + 1e-6)  # [B, L]

            # ----------- CONF MASK ---------------
            conf_mask = (conf_np > conf_thres).astype(np.float32)  # (B,L,15)
            masked_pixel_error = (pixel_error * conf_mask)
            n_valid = conf_mask.sum(axis=-1)  # [B,L]

            mpjpe_px = masked_pixel_error.sum(axis=-1) / (n_valid + 1e-8)  # (B,L)
            mpjpe_mm = (masked_pixel_error * scale[:, :, None]).sum(axis=-1) / (n_valid + 1e-8)  # (B,L)
            all_mpjpe_px.append(mpjpe_px)
            all_mpjpe_mm.append(mpjpe_mm)
            # ------------------------------------

            # 可视化第一个 batch 第一条序列的前 n_vis_frames
            B, L, _, _ = pred.shape
            b = 0
            for frame in range(min(n_vis_frames, L)):
                gt_pts = gt_denorm[b, frame]
                pr_pts = pred_denorm[b, frame]
                plot_pose(
                    gt_pts, pr_pts, edges,
                    title=f"Frame {frame} | MPJPE(px): {mpjpe_px[b,frame]:.1f} | MPJPE(mm): {mpjpe_mm[b,frame]:.1f}",
                    save_path=os.path.join(out_dir, f"seq0_frame{frame:03d}.png")
                )
            break

    all_mpjpe_px = np.concatenate(all_mpjpe_px, axis=1)
    all_mpjpe_mm = np.concatenate(all_mpjpe_mm, axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(all_mpjpe_px[0], label='MPJPE (px, masked)')
    plt.plot(all_mpjpe_mm[0], label='MPJPE (mm, masked)')
    plt.xlabel('Frame')
    plt.ylabel('Error')
    plt.title('Masked MPJPE over time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mpjpe_curve.png"))
    plt.close()

    print(f"【图片已保存至】{out_dir}")
    print(f"【平均MPJPE (px, masked)】{all_mpjpe_px[0].mean():.2f}    【平均MPJPE (mm, masked)】{all_mpjpe_mm[0].mean():.2f}")

def main():
    # ==== 配置路径和参数 ====
    ckpt = "/data/wangtiantian/pose/result/best_ws75_st5_bs8_lr0.0001_hd256_dr0.1_maskright hand0.1-right pocket0.1-glasses0.1-left pocket0.1-left hand0.1_20250714_180616.pth"
    
    zscore_dir = "/data/wangtiantian/pose/processed_data/pose_norm_zscore"
    output_dir = "/data/wangtiantian/pose/result/result_picture_all"
    batch_size = 8
    window_size = 75
    stride = 5
    num_workers = 4
    n_vis_frames = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask_probs = {
        'right hand': 0.1, 'right pocket': 0.1, 'glasses': 0.1, 'left pocket': 0.1, 'left hand': 0.1
    }
    scene_list = ['kitchen', 'office', 'livingroom']
    zscore_files = {
        s: os.path.join(zscore_dir, f"pose_zscore_param_{s}.json")
        for s in scene_list
    }

    for scene in scene_list:
        print(f"\n======= 评估场景: {scene} =======")
        zscore_path = zscore_files[scene]
        mean, std = load_pose_zscore_param(zscore_path)
        mean_xy = mean[:, :2]
        std_xy = std[:, :2]

        test_set = IMUPoseSeqDataset(
    imu_dir="/data/wangtiantian/pose/processed_data/imu/test",
    pose_dir="/data/wangtiantian/pose/processed_data/pose/test",
    window_size=window_size, stride=stride, mask_probs=mask_probs)

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

        model = IMU2PoseNet_Fusion().to(device)
        state_dict = torch.load(ckpt, map_location=device)
        if 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded model from {ckpt}")

        scene_out_dir = os.path.join(output_dir, scene)
        test_and_visualize(model, test_loader, device, mean_xy, std_xy, scene_out_dir, n_vis_frames=n_vis_frames)

if __name__ == "__main__":
    main()

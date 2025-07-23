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

def test_and_visualize(model, test_loader, device,
                       mean_xy, std_xy,
                       out_dir, n_vis_frames=50, conf_thres=0.5):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (10, 11),
        (8, 12), (12, 13), (13, 14)
    ]

    # -------- 全局统计量初始化 --------
    total_px_err, total_mm_err, total_valid = 0.0, 0.0, 0.0

    # 若仍想画曲线，可存首 batch 的值
    curve_px, curve_mm = None, None

    with torch.no_grad():
        for step, (imu_batch, pose_batch, keep_batch, n_valid_list) in \
                enumerate(tqdm(test_loader, desc="Test+Visualize")):

            imu_batch  = imu_batch.float().to(device)
            pose_batch = pose_batch.float().to(device)
            pred = model(imu_batch, keep_batch, n_valid_list)  # (B,L,15,2)

            xy_gt = pose_batch[..., :2]            # (B,L,15,2)
            conf  = pose_batch[..., 2]             # (B,L,15)

            # ---- 反归一化 ----
            pred_denorm = pred.cpu().numpy() * std_xy[None, None] + mean_xy[None, None]
            gt_denorm   = xy_gt.cpu().numpy()     * std_xy[None, None] + mean_xy[None, None]
            conf_np     = conf.cpu().numpy()

            # ---- 逐关节点误差 (像素) ----
            pixel_error = np.linalg.norm(pred_denorm - gt_denorm, axis=-1)  # (B,L,15)

            # ---- 计算毫米 scale ----
            neck   = gt_denorm[:, :, 1, :]
            midhip = gt_denorm[:, :, 8, :]
            dist_px  = np.linalg.norm(neck - midhip, axis=2)                # (B,L)
            scale_mm = 375.0 / (dist_px + 1e-6)                             # (B,L)

            # ---- 用置信度遮罩 ----
            conf_mask = (conf_np > conf_thres).astype(np.float32)           # (B,L,15)
            masked_pixel_error = pixel_error * conf_mask                    # (B,L,15)
            masked_mm_error    = masked_pixel_error * scale_mm[:, :, None]  # (B,L,15)

            # ---- 累加到全局 ----
            total_px_err += masked_pixel_error.sum()
            total_mm_err += masked_mm_error.sum()
            total_valid  += conf_mask.sum()

            # ---- 若想画曲线，只保留首 batch ----
            if step == 0:
                curve_px = masked_pixel_error.mean(axis=-1)[0]   # (L,)
                curve_mm = masked_mm_error.mean(axis=-1)[0]      # (L,)

                # 同时可视化前 n_vis_frames 帧
                B, L, _, _ = pred.shape
                b = 0
                for frame in range(min(n_vis_frames, L)):
                    plot_pose(
                        gt_denorm[b, frame], pred_denorm[b, frame], edges,
                        title=f"Frame {frame} | MPJPE(px): {curve_px[frame]:.1f} | "
                              f"MPJPE(mm): {curve_mm[frame]:.1f}",
                        save_path=os.path.join(out_dir, f"seq0_frame{frame:03d}.png")
                    )

    # -------- 全局平均值 --------
    global_mpjpe_px = total_px_err / (total_valid + 1e-8)
    global_mpjpe_mm = total_mm_err / (total_valid + 1e-8)

    # -------- 曲线绘制（可选） --------
    if curve_px is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(curve_px, label='MPJPE (px, masked)')
        plt.plot(curve_mm, label='MPJPE (mm, masked)')
        plt.xlabel('Frame')
        plt.ylabel('Error')
        plt.title('Masked MPJPE over time (first sequence)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mpjpe_curve.png"))
        plt.close()

    # -------- 结果打印 --------
    print(f"【图片已保存至】{out_dir}")
    print(f"【全局平均 MPJPE (px, masked)】{global_mpjpe_px:.2f}")
    print(f"【全局平均 MPJPE (mm, masked)】{global_mpjpe_mm:.2f}")


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

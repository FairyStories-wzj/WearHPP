import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from IMU_to_pose.utils import reorder_imu, generate_modality_mask
from IMU_to_pose.imu_model import IMU2PoseNet_Fusion

def infer_pose(
    model_pth,
    imu_csv,
    start,
    end,
    mask_probs=None,
    device='cuda',
    chunk_size=200  # 每次推理多少帧（避免超显存）
):
    """
    输入：imu_csv路径、起止帧
    输出：pred_pose  shape=(L, 15, 2)
    """
    # 1. 加载模型
    model = IMU2PoseNet_Fusion().to(device)
    state_dict = torch.load(model_pth, map_location=device)
    if 'model_state' in state_dict:
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    # 2. 加载并切片IMU
    imu_df = pd.read_csv(imu_csv).iloc[start:end+1]
    imu_arr = reorder_imu(imu_df)        # (L, 5, 6)
    L = imu_arr.shape[0]

    # 随机mask（可选）
    if mask_probs is not None:
        msk = generate_modality_mask(mask_probs)
    else:
        msk = np.ones(5, dtype=bool)
    keep = np.where(msk)[0]
    valid_imu = imu_arr[:, keep, :]      # (L, n_valid, 6)

    # 3. 分块推理（加进度条）
    pred_pose_all = []
    n_chunks = (L + chunk_size - 1) // chunk_size

    for i in tqdm(range(n_chunks), desc='Infer Pose'):
        s = i * chunk_size
        e = min((i+1)*chunk_size, L)
        chunk = valid_imu[s:e]    # [chunk, n_valid, 6]
        imu_tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(device)  # [1, chunk, n_valid, 6]
        keep_tensor = torch.tensor(keep).unsqueeze(0)
        n_valid_list = [len(keep)]
        with torch.no_grad():
            pred = model(imu_tensor, keep_tensor, n_valid_list)    # [1, chunk, 15, 2]
        pred_pose = pred.squeeze(0).cpu().numpy()                  # [chunk, 15, 2]
        pred_pose_all.append(pred_pose)

    pred_pose_all = np.concatenate(pred_pose_all, axis=0)          # (L, 15, 2)
    return pred_pose_all


def infer_pose_training(  # 我自己改的，区别是①传入的不是model_path，而是一个model  ②返回torch.tensor而不是numpy以便可以更新损失
    model,
    imu_csv,
    start,
    end,
    mask_probs=None,
    device='cuda',
    chunk_size=200  # 每次推理多少帧（避免超显存）
):
    """
    输入：imu_csv路径、起止帧
    输出：pred_pose  shape=(L, 15, 2)
    """

    # 2. 加载并切片IMU
    imu_df = pd.read_csv(imu_csv).iloc[start:end+1]
    imu_arr = reorder_imu(imu_df)        # (L, 5, 6)
    L = imu_arr.shape[0]

    # 随机mask（可选）
    if mask_probs is not None:
        msk = generate_modality_mask(mask_probs)
    else:
        msk = np.ones(5, dtype=bool)
    keep = np.where(msk)[0]
    valid_imu = imu_arr[:, keep, :]      # (L, n_valid, 6)

    # 3. 分块推理（加进度条）
    pred_pose_all = []
    n_chunks = (L + chunk_size - 1) // chunk_size

    for i in range(n_chunks):  # 进度条不要了，输出太多
        s = i * chunk_size
        e = min((i+1)*chunk_size, L)
        chunk = valid_imu[s:e]    # [chunk, n_valid, 6]
        imu_tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(device)  # [1, chunk, n_valid, 6]
        keep_tensor = torch.tensor(keep).unsqueeze(0)
        n_valid_list = [len(keep)]
        # with torch.no_grad():
        #     pred = model(imu_tensor, keep_tensor, n_valid_list)    # [1, chunk, 15, 2]
        pred = model(imu_tensor, keep_tensor, n_valid_list)  # [1, chunk, 15, 2]
        # pred_pose = pred.squeeze(0).cpu().numpy()                  # [chunk, 15, 2]
        pred_pose = pred.squeeze(0)
        pred_pose_all.append(pred_pose)

    # pred_pose_all = np.concatenate(pred_pose_all, axis=0)          # (L, 15, 2)
    pred_pose_all = torch.cat(pred_pose_all, dim=0)
    return pred_pose_all

# 用法举例
if __name__ == '__main__':
    model_pth = "E:\\Python Project\\WearHPP\\IMU_to_pose\\checkpoints\\best_ws75_st5_bs8_lr0.0001_hd256_dr0.1_maskright hand0.1-right pocket0.1-glasses0.1-left pocket0.1-left hand0.1_20250714_180616.pth"
    imu_csv = "E:\\Xrf2\\imu\\test\\0_kitchen_2_imu.csv"
    start, end = 0, 75  #

    pred_pose = infer_pose(
        model_pth=model_pth,
        imu_csv=imu_csv,
        start=start,
        end=end,
        mask_probs=None,    # 不mask
        device='cuda',
        chunk_size=256      # 可调整, 显存小可以更小
    )

    print(pred_pose)  # (L, 15, 2)
    print(pred_pose.shape)


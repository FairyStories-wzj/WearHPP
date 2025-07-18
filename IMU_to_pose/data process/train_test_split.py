import os
import pandas as pd
import shutil
from tqdm import tqdm

# 映射关系
room_map = {"kitchen": "3", "livingroom": "1", "office": "2"}

def find_scene_and_name(fname):
    # fname: 0_1_6.h5
    g, room_id, seg = fname.replace('.h5', '').split('_')
    # 反查房间名
    for scene, num in room_map.items():
        if room_id == num:
            return scene, f"{g}_{scene}_{seg}"
    raise ValueError(f"房间id {room_id} 无法映射场景，文件名: {fname}")

def copy_by_csv_list(csv_path, imu_src_root, pose_src_root, imu_dst_dir, pose_dst_dir):
    os.makedirs(imu_dst_dir, exist_ok=True)
    os.makedirs(pose_dst_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    for fname in tqdm(df['imu_filename'], desc=f"复制 {csv_path}"):
        scene, shortname = find_scene_and_name(fname)
        imu_src = os.path.join(imu_src_root, scene, f"{shortname}_imu.csv")
        pose_src = os.path.join(pose_src_root, scene, f"{shortname}_pose.csv")
        imu_dst = os.path.join(imu_dst_dir, f"{shortname}_imu.csv")
        pose_dst = os.path.join(pose_dst_dir, f"{shortname}_pose.csv")
        if not os.path.exists(imu_src):
            print(f"[警告] IMU文件不存在: {imu_src}")
        else:
            shutil.copy2(imu_src, imu_dst)
        if not os.path.exists(pose_src):
            print(f"[警告] POSE文件不存在: {pose_src}")
        else:
            shutil.copy2(pose_src, pose_dst)

if __name__ == "__main__":
    imu_src_root = "/data/wangtiantian/pose/processed_data/imu_norm"
    pose_src_root = "/data/wangtiantian/pose/processed_data/pose_norm_zscore"
    # 目的地
    train_imu_dst = "/data/wangtiantian/pose/processed_data/imu/train"
    train_pose_dst = "/data/wangtiantian/pose/processed_data/pose/train"
    test_imu_dst = "/data/wangtiantian/pose/processed_data/imu/test"
    test_pose_dst = "/data/wangtiantian/pose/processed_data/pose/test"

    train_csv = "/data/wangtiantian/pose/imu/testandtrain/cur_train.csv"
    test_csv  = "/data/wangtiantian/pose/imu/testandtrain/cur_test.csv"

    # 训练集
    copy_by_csv_list(train_csv, imu_src_root, pose_src_root, train_imu_dst, train_pose_dst)
    # 测试集
    copy_by_csv_list(test_csv, imu_src_root, pose_src_root, test_imu_dst, test_pose_dst)

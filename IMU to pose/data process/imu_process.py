import os
import glob
import csv
import pandas as pd
import json

device_to_name = {
    'WT5300006649': 'right hand',
    'WT5300006678': 'right pocket',
    'WT5300006735': 'glasses',
    'WT5300006748': 'left pocket',
    'WT5300006770': 'left hand',
    'WT5300005126': 'left pocket',
    'WT5300006633': 'right pocket',
    'WT5300006676': 'right hand',
    'WT5300006727': 'glasses',
    'WT5300006751': 'left hand'
}

scene_to_roomid = {
    "kitchen": "3",
    "livingroom": "1",
    "office": "2"
}

N_KPTS = 15
KPT_DIM = 3  # (x, y, conf)

def read_imu_file(imu_path, missing_time_files):
    with open(imu_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
    if '接收时间' in header:
        df = pd.read_csv(
            imu_path,
            sep='\t',
            parse_dates=['接收时间'],
            infer_datetime_format=True
        ).rename(columns={'接收时间':'time', '设备编号':'device_id'})
    else:
        missing_time_files.append(imu_path)
        df = pd.read_csv(imu_path, sep='\t')
        time_col = df.columns[0]
        df = df.rename(columns={time_col:'time'})
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if '设备编号' in df.columns:
            df = df.rename(columns={'设备编号':'device_id'})
    return df

def align_imu_pose_all_scenes(imu_root, pose_root, imu_out_root, pose_out_root, scene_to_roomid):
    scenes = scene_to_roomid.keys()
    for scene in scenes:
        print(f"\n==== 处理场景: {scene} ====")
        imu_dir = os.path.join(imu_root, scene)
        pose_dir = os.path.join(pose_root, f"{scene}15")
        imu_out_dir = os.path.join(imu_out_root, scene)
        pose_out_dir = os.path.join(pose_out_root, scene)
        os.makedirs(imu_out_dir, exist_ok=True)
        os.makedirs(pose_out_dir, exist_ok=True)
        missing_time_files = []
        room_id = scene_to_roomid[scene]

        for imu_path in sorted(glob.glob(os.path.join(imu_dir, '*.txt'))):
            base = os.path.basename(imu_path).replace('.txt','')
            parts = base.split('_')
            if len(parts) != 3:
                print(f"跳过非标准文件名：{base}")
                continue
            g, room_id_in_file, seg = parts
            if room_id_in_file != room_id:
                print(f"⚠️ 文件 {base} 的房间号 {room_id_in_file} 与场景 {scene} 不匹配，跳过")
                continue
            start_k, end_k = map(int, seg.split('-'))

            df_imu = read_imu_file(imu_path, missing_time_files)
            imu_cols = ['加速度X','加速度Y','加速度Z','角速度X','角速度Y','角速度Z']
            df_imu = df_imu[['time','device_id'] + imu_cols].sort_values('time').reset_index(drop=True)
            device_ids = sorted(df_imu['device_id'].unique())
            used_parts = set()
            imu_pivot = []
            for dev in device_ids:
                part_name = device_to_name.get(dev, dev)
                if part_name in used_parts:
                    continue
                used_parts.add(part_name)
                sub = df_imu[df_imu['device_id']==dev][['time'] + imu_cols].copy()
                sub = sub.rename(columns={c: f'{part_name}_{c}' for c in imu_cols})
                imu_pivot.append(sub)
            if not imu_pivot:
                print(f"⚠️ 没有有效IMU数据：{imu_path}")
                continue
            df_imu_merged = imu_pivot[0]
            for sub in imu_pivot[1:]:
                df_imu_merged = pd.merge_asof(
                    df_imu_merged.sort_values('time'), 
                    sub.sort_values('time'), 
                    on='time', direction='nearest'
                )

            for k in range(start_k, end_k+1):
                json_file = os.path.join(pose_dir, f'{g}_{scene}_{k}.json')
                if not os.path.isfile(json_file):
                    print(f"⚠️ 缺少 JSON: {json_file}")
                    continue
                with open(json_file, 'r', encoding='utf-8') as f:
                    pose_list = json.load(f)
                rows = []
                for p in pose_list:
                    t = pd.to_datetime(p['frame_time'], errors='coerce')
                    raw_kpts = p.get('pose_key_points:', [])
                    if raw_kpts and isinstance(raw_kpts, list) and len(raw_kpts) > 0:
                        coords = raw_kpts[0]
                        if len(coords) < N_KPTS * KPT_DIM:
                            coords = coords + [0.0] * (N_KPTS * KPT_DIM - len(coords))
                        elif len(coords) > N_KPTS * KPT_DIM:
                            coords = coords[:N_KPTS * KPT_DIM]
                    else:
                        coords = [0.0] * (N_KPTS * KPT_DIM)
                    rows.append({'time': t, 'keypoints': coords})
                df_pose = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
                pose_csv  = os.path.join(pose_out_dir, f'{g}_{scene}_{k}_pose.csv')
                df_pose.to_csv(pose_csv, index=False)
                print(f"✅ 已保存 pose：{pose_csv}")

                start_time = df_pose['time'].min()
                end_time   = df_pose['time'].max()
                df_imu_clip = df_imu_merged[(df_imu_merged['time'] >= start_time) & (df_imu_merged['time'] <= end_time)].copy()
                imu_csv = os.path.join(imu_out_dir, f'{g}_{scene}_{k}_imu.csv')
                df_imu_clip.to_csv(imu_csv, index=False)
                print(f"✅ 已保存 imu：{imu_csv}")

        if missing_time_files:
            print("\n下列 IMU 文件缺少 '接收时间' 列，用第1列解析 (可能产生 NaT)：")
            for p in missing_time_files:
                print(" -", p)
        else:
            print("\n所有 IMU 文件均包含 '接收时间' 列。")

if __name__ == '__main__':
    imu_root     = '/data/wangtiantian/pose/raw_data/imu'
    pose_root    = '/data/wangtiantian/pose/xrfv2/pose_15'
    imu_out_root = '/data/wangtiantian/pose/processed_data/imu'
    pose_out_root= '/data/wangtiantian/pose/processed_data/pose'
    align_imu_pose_all_scenes(imu_root, pose_root, imu_out_root, pose_out_root, scene_to_roomid)

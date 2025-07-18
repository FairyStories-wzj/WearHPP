import os
import pandas as pd
import numpy as np
from glob import glob

def normalize_imu_csv_folder(folder, out_folder, method="zscore"):
    os.makedirs(out_folder, exist_ok=True)
    files = glob(os.path.join(folder, "*.csv"))
    for f in files:
        df = pd.read_csv(f)
        df_norm = df.copy()
        for col in df.columns:
            if col == "time":
                continue
            vals = df[col].values
            # 非零（不考虑0/缺失作为归一化依据）
            nonzero = vals[vals != 0]
            if len(nonzero) == 0:
                continue  # 全是0
            if method == "zscore":
                mean = nonzero.mean()
                std = nonzero.std()
                # 只对非0的数值做归一化，0保持0
                df_norm[col] = np.where(
                    vals != 0,
                    (vals - mean) / (std + 1e-8),
                    0
                )
            elif method == "minmax":
                vmin = nonzero.min()
                vmax = nonzero.max()
                df_norm[col] = np.where(
                    vals != 0,
                    (vals - vmin) / (vmax - vmin + 1e-8),
                    0
                )
            else:
                raise ValueError("method should be 'zscore' or 'minmax'")
        out_path = os.path.join(out_folder, os.path.basename(f))
        df_norm.to_csv(out_path, index=False)
        print(f"✅ 归一化后保存：{out_path}")

# 用法示例
if __name__ == "__main__":
    src_dir = "/data/wangtiantian/pose/processed_data/imu/office"    # 改成你的输入目录
    dst_dir = "/data/wangtiantian/pose/processed_data/imu_norm/office"  # 改成输出目录
    normalize_imu_csv_folder(src_dir, dst_dir, method="zscore")  # or method="minmax"

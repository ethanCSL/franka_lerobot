import os
import pandas as pd
import numpy as np

parquet_dir = '/home/csl/.cache/huggingface/lerobot/ethanCSL/0804_wipe_fix/data/chunk-001'

# 取得所有 parquet 檔案
parquet_files = sorted([
    f for f in os.listdir(parquet_dir)
    if f.endswith('.parquet')
])

print(f'📂 找到 {len(parquet_files)} 個 .parquet 檔案，開始處理...')

for fname in parquet_files:
    fpath = os.path.join(parquet_dir, fname)
    print(f'🔧 處理中: {fname}')

    # 讀取 parquet
    df = pd.read_parquet(fpath)

    def remove_index_8(x):
        if hasattr(x, '__len__') and len(x) > 8:
            x_list = list(x)  # 確保可以 slicing
            return x_list[:8] + x_list[9:]  # 移除 index 8
        return x

    df['observation.state'] = df['observation.state'].apply(remove_index_8)
    df['action'] = df['action'].apply(remove_index_8)

    # 覆寫 parquet 檔案
    df.to_parquet(fpath, index=False)

print('✅ 所有檔案處理完成！')

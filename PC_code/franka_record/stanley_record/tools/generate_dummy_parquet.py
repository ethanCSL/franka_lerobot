import pandas as pd
import numpy as np

# 假設有 100 幀的資料
N = 254

# 建立 dummy 資料
df = pd.DataFrame({
    "timestamp": np.linspace(0.0, N / 30.0, N),           # 30 FPS 時間戳
    "robot_state": [np.zeros(10).tolist()] * N,           # 10 維 robot state
    "action": [np.ones(4).tolist()] * N,                  # 4 維 action
    "episode_index": [0] * N                              # 👈 必要欄位！
})

df.to_parquet("episode_000000.parquet")
print("✅ Saved with episode_index")


import pandas as pd
from pathlib import Path

def overwrite_action_with_state(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    if "action" not in df.columns or "observation.state" not in df.columns:
        print(f"❌ {parquet_path.name} 缺少 'action' 或 'observation.state' 欄位，跳過")
        return

    df["action"] = df["observation.state"]
    df.to_parquet(parquet_path, index=False)
    print(f"✅ 已處理: {parquet_path.name}")

# 🔧 設定你的資料夾路徑
chunk_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "ethanCSL" / "0804_wipe" / "data" / "chunk-001"

# 🧾 處理所有 parquet 檔
parquet_files = sorted(chunk_path.glob("episode_*.parquet"))
print(f"共發現 {len(parquet_files)} 個 parquet 檔案")

for pfile in parquet_files:
    overwrite_action_with_state(pfile)

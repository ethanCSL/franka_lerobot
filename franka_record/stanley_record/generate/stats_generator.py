### stats_generator.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import logging

class StatsGenerator:
    """
    Reads all Parquet files and video files under a dataset root to compute:
      - action mean/std
      - observation.state mean/std
      - observation.images mean/std for each view
    Writes stats.json under meta/
    """
    def __init__(self, root: Path, fps: int, chunk_size: int):
        self.root = root
        self.fps = fps
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)

    def generate(self) -> None:
        stats = {}
        data_dir = self.root / "data"
        video_dir = self.root / "videos"

        # 1. Gather all Parquet paths
        parquet_paths = list(data_dir.rglob("*.parquet"))
        self.logger.info(f"Found {len(parquet_paths)} parquet files")

        # Load all into a single DataFrame (or empty)
        df_list = [pd.read_parquet(p) for p in parquet_paths]
        all_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

        #2. Compute action stats
        if "action" in all_df.columns:
            try:
                state_arr = np.stack(all_df["action"].to_numpy())
                stats["action"] = {
                    "mean": state_arr.mean(axis=0).tolist(),
                    "std": state_arr.std(axis=0).tolist(),
                }
            except Exception as e:
                self.logger.warning(f"Failed to compute action stats: {e}")
                stats["action"] = {"mean": [], "std": []}
        else:
            stats["action"] = {"mean": [], "std": []}

        # 3. Compute observation.state stats
        if "observation.state" in all_df.columns:
            try:
                state_arr = np.stack(all_df["observation.state"].to_numpy())
                stats["observation.state"] = {
                    "mean": state_arr.mean(axis=0).tolist(),
                    "std": state_arr.std(axis=0).tolist(),
                }
            except Exception as e:
                self.logger.warning(f"Failed to compute observation.state stats from array: {e}")
                stats["observation.state"] = {"mean": [], "std": []}
        else:
            # fallback for old format with separate columns
            state_cols = [
                "eef_position_x", "eef_position_y", "eef_position_z",
                "eef_quat_x", "eef_quat_y", "eef_quat_z", "eef_quat_w", "gripper_width", "f_ext"
            ]
            if all(c in all_df.columns for c in state_cols):
                try:
                    sarr = all_df[state_cols].to_numpy()
                    stats["observation.state"] = {
                        "mean": sarr.mean(axis=0).tolist(),
                        "std": sarr.std(axis=0).tolist(),
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to compute observation.state stats from legacy columns: {e}")
                    stats["observation.state"] = {"mean": [], "std": []}
            else:
                stats["observation.state"] = {"mean": [], "std": []}

        # 4. Compute image stats per view
        for folder, stat_key in [
            ("observation.images.image_additional_view", "observation.images.image_additional_view"),
            ("observation.images.image", "observation.images.image"),
        ]:
            # Correct glob pattern to find all videos under chunk-*/viewX
            video_paths = list(video_dir.glob(f"chunk-*/{folder}/episode_*.mp4"))
            ch_sum = np.zeros(3)
            ch_sq = np.zeros(3)
            total_pixels = 0
            for vp in video_paths:
                cap = cv2.VideoCapture(str(vp))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img = frame.astype(np.float32) / 255.0
                    h, w, _ = img.shape
                    total_pixels += h * w
                    ch_sum += img.sum(axis=(0, 1))
                    ch_sq += (img ** 2).sum(axis=(0, 1))
                cap.release()
            if total_pixels > 0:
                mean = (ch_sum / total_pixels).tolist()
                var = (ch_sq / total_pixels) - np.square(ch_sum / total_pixels)
                std = np.sqrt(np.maximum(var, 0)).tolist()
                stats[stat_key] = {"mean": mean, "std": std}
            else:
                stats[stat_key] = {"mean": [], "std": []}

        # 5. Write to meta/stats.json
        out_path = self.root / "meta" / "stats.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=4)


# import json
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import cv2
# import logging


# class StatsGenerator:
#     """
#     Reads all Parquet files and video files under a dataset root to compute:
#       - action min/max/mean/std/count
#       - observation.state min/max/mean/std/count
#       - observation.images mean/std/count for each view (normalized)
#     Writes stats.json under meta/
#     """
#     def __init__(self, root: Path, fps: int, chunk_size: int):
#         self.root = root
#         self.fps = fps
#         self.chunk_size = chunk_size
#         self.logger = logging.getLogger(__name__)

#     def generate(self) -> None:
#         stats: dict = {}
#         data_dir = self.root / "data"
#         video_dir = self.root / "videos"

#         # 1. Gather all Parquet paths
#         parquet_paths = list(data_dir.rglob("*.parquet"))
#         self.logger.info(f"Found {len(parquet_paths)} parquet files")

#         # Load all into a single DataFrame
#         df_list = [pd.read_parquet(p) for p in parquet_paths]
#         all_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

#         # Helper to compute full stats
#         def full_stats(arr: np.ndarray) -> dict:
#             return {
#                 "min": np.min(arr, axis=0).tolist(),
#                 "max": np.max(arr, axis=0).tolist(),
#                 "mean": np.mean(arr, axis=0).tolist(),
#                 "std": np.std(arr, axis=0).tolist(),
#                 "count": int(arr.shape[0]),
#             }

#         # 2. Compute action stats
#         if "action" in all_df.columns:
#             try:
#                 arr = np.stack(all_df["action"].to_numpy())
#                 stats["action"] = full_stats(arr)
#             except Exception as e:
#                 self.logger.warning(f"Failed action stats: {e}")
#                 stats["action"] = full_stats(np.zeros((0,8)))
#         else:
#             stats["action"] = full_stats(np.zeros((0,8)))

#         # 3. Compute observation.state stats
#         if "observation.state" in all_df.columns:
#             try:
#                 arr = np.stack(all_df["observation.state"].to_numpy())
#                 stats["observation.state"] = full_stats(arr)
#             except Exception as e:
#                 self.logger.warning(f"Failed state stats: {e}")
#                 stats["observation.state"] = full_stats(np.zeros((0,8)))
#         else:
#             state_cols=["eef_position_x","eef_position_y","eef_position_z",
#                          "eef_quat_x","eef_quat_y","eef_quat_z","eef_quat_w","gripper_width"]
#             if all(c in all_df.columns for c in state_cols):
#                 arr = all_df[state_cols].to_numpy()
#                 stats["observation.state"] = full_stats(arr)
#             else:
#                 stats["observation.state"] = full_stats(np.zeros((0,8)))

#         # 4. Compute image stats per view
#         for stat_key in ["observation.images.image_additional_view", "observation.images.image"]:
#             folder = stat_key
#             video_paths = list(video_dir.glob(f"chunk-*//{folder}/episode_*.mp4"))
#             ch_sum = np.zeros(3, np.float64)
#             ch_sq = np.zeros(3, np.float64)
#             total = 0
#             for vp in video_paths:
#                 cap = cv2.VideoCapture(str(vp))
#                 while True:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     img = frame.astype(np.float32)/255.0
#                     h,w,_=img.shape
#                     total += h*w
#                     ch_sum += img.sum((0,1))
#                     ch_sq += (img*img).sum((0,1))
#                 cap.release()
#             if total>0:
#                 mean=(ch_sum/total).tolist()
#                 var=(ch_sq/total)-np.square(ch_sum/total)
#                 std=np.sqrt(np.maximum(var,0)).tolist()
#                 stats[stat_key]={"mean":mean,"std":std,"count":int(total)}
#             else:
#                 stats[stat_key]={"mean":[],"std":[],"count":0}

#         # 5. Write
#         out=self.root/"meta"/"stats.json"
#         out.parent.mkdir(parents=True,exist_ok=True)
#         with open(out,'w') as f:
#             json.dump(stats,f,indent=4)

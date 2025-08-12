#franka_dataset.py
import json
import pathlib
from typing import Optional, List, Dict

import cv2
import numpy as np
import pandas as pd


class FrankaDataset:
    """
    Builds a LeRobot-compatible dataset for a Franka robot.

    Directory structure:
        ~/.cache/huggingface/lerobot/{repo_id}/
            data/
                chunk-000/
                    episode_000000.parquet
                ...
            videos/
                chunk-000/
                    view1/episode_000000.mp4
                    view2/episode_000000.mp4
                ...
            meta/
                info.json
                stats.json
                tasks.jsonl
                episodes.jsonl
    """

    CODEBASE_VERSION = "v2.0"

    def __init__(
        self,
        repo_id: str,
        root: Optional[pathlib.Path] = None,
        fps: int = 30,
        chunk_size: int = 100,
        task_name: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.task_name = task_name or ""
        self.fps = fps
        self.chunk_size = chunk_size

        repo_path = pathlib.Path(*repo_id.split("/"))
        default_root = (
            pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / repo_path
        )
        self.root = (root or default_root).expanduser()
        self._make_dirs()

        self.episodes: List[Dict] = []
        self.total_frames: int = 0

    def _make_dirs(self) -> None:
        for sub in ("data", "videos", "meta"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def add_episode(
        self,
        episode_index: int,
        eef_positions: np.ndarray,        # (N,3)
        eef_orientations: np.ndarray,     # (N,3)
        gripper_widths: np.ndarray,       # (N,)
        view1_frames: np.ndarray,         # (N,H,W,3)
        view2_frames: np.ndarray, 
    ) -> None:
        """
        Add a single episode: store sensor data (parquet), videos, and metadata.
        """
        chunk_idx = episode_index // self.chunk_size
        chunk_name = f"chunk-{chunk_idx:03d}"

        N = eef_positions.shape[0]
        self._save_parquet(
            episode_index=episode_index,
            chunk_name=chunk_name,
            eef_pos=eef_positions,
            eef_ori=eef_orientations,
            gripper=gripper_widths,
            frame_count=N,
        )

        for view_name, frames in ("view1", view1_frames), ("view2", view2_frames):
            self._save_video(view_name, episode_index, chunk_name, frames)

        self.episodes.append({
            "episode_index": episode_index,
            "tasks": [self.task_name],
            "length": N,
        })
        self.total_frames += N

    def _save_parquet(
        self,
        episode_index: int,
        chunk_name: str,
        eef_pos: np.ndarray,
        eef_ori: np.ndarray,
        gripper: np.ndarray,
        frame_count: int,
    ) -> None:
        """
        eef_pos: shape (N,3)   每幀 xyz
        eef_ori: shape (N,3)   每幀 rpy
        gripper: shape (N,)    每幀夾爪寬度
        frame_count: N
        """
        data_dir = self.root / "data" / chunk_name
        data_dir.mkdir(parents=True, exist_ok=True)

        timestamps = np.arange(frame_count) / self.fps
        base_index = episode_index * frame_count

        df = pd.DataFrame({
            "timestamp":       timestamps,
            "frame_index":     np.arange(frame_count),
            "episode_index":   episode_index,
            "eef_position_x":  eef_pos[:, 0],
            "eef_position_y":  eef_pos[:, 1],
            "eef_position_z":  eef_pos[:, 2],
            "eef_roll":        eef_ori[:, 0],
            "eef_pitch":       eef_ori[:, 1],
            "eef_yaw":         eef_ori[:, 2],
            "gripper_width":   gripper,
            "index":           np.arange(frame_count) + base_index,
            "task_index":      np.zeros(frame_count, dtype=int),
        })

        path = data_dir / f"episode_{episode_index:06d}.parquet"
        df.to_parquet(path, index=False)
    
    def _save_video(
        self,
        view_name: str,
        episode_index: int,
        chunk_name: str,
        frames: np.ndarray,
    ) -> None:
        view_dir = self.root / "videos" / chunk_name / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        height, width = frames.shape[1], frames.shape[2]
        out_path = view_dir / f"episode_{episode_index:06d}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()

    def finalize(self) -> None:
        """
        Generate info.json, stats.json, tasks.jsonl, and episodes.jsonl under meta/.
        """
        self._write_info()
        self._write_stats()
        self._write_tasks()
        self._write_episodes()

    def _write_info(self) -> None:
        total_eps = len(self.episodes)
        total_chunks = (total_eps - 1) // self.chunk_size + 1 if total_eps > 0 else 0
        info = {
            "codebase_version": self.CODEBASE_VERSION,
            "robot_type": "franka",
            "total_episodes": total_eps,
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "total_videos": total_eps * 2,
            "total_chunks": total_chunks,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": f"0:{total_eps}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "features": {
                "observation.images.image_additional_view": {
                    "dtype": "video",
                    "shape": [128, 128, 3],
                    "names": ["height", "width", "rgb"],
                    "info": {
                        "video.fps": float(self.fps),
                        "video.height": 128,
                        "video.width": 128,
                        "video.channels": 3,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.images.image": {
                    "dtype": "video",
                    "shape": [128, 128, 3],
                    "names": ["height", "width", "rgb"],
                    "info": {
                        "video.fps": float(self.fps),
                        "video.height": 128,
                        "video.width": 128,
                        "video.channels": 3,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": {
                        "motors": ["x", "y", "z", "roll", "pitch", "yaw",
                                   "width"]
                    }
                },
                "action": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": {
                        "motors": ["x", "y", "z", "roll", "pitch", "yaw",
                                   "width"]
                    }
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                }
            }
        }
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=4)

    def _write_stats(self) -> None:
        stats = {}
        parquet_paths = list((self.root / "data").rglob("*.parquet"))
        num_cols = ["timestamp","frame_index","episode_index","index","task_index",
                    "eef_position_x","eef_position_y","eef_position_z",
                    "eef_roll","eef_pitch","eef_yaw","gripper_width"]
        df_list = []
        for p in parquet_paths:
            df_list.append(pd.read_parquet(p, columns=num_cols))
        if df_list:
            all_df = pd.concat(df_list, ignore_index=True)
            for col in num_cols:
                arr = all_df[col]
                stats[col] = {
                    "mean": [float(arr.mean())],
                    "std": [float(arr.std())],
                    "max": [float(arr.max())],
                    "min": [float(arr.min())]
                }
            state_arr = all_df[["eef_position_x","eef_position_y","eef_position_z",
                                "eef_roll","eef_pitch","eef_yaw","gripper_width"]].values
            obs_state = state_arr[:,:6]
            stats["observation.state"] = {
                "mean": [float(x) for x in obs_state.mean(axis=0).tolist()],
                "std": [float(x) for x in obs_state.std(axis=0).tolist()],
                "max": [float(x) for x in obs_state.max(axis=0).tolist()],
                "min": [float(x) for x in obs_state.min(axis=0).tolist()]
            }
        else:
            for col in num_cols:
                stats[col] = {"mean":[],"std":[],"max":[],"min":[]}
            stats["observation.state"] = {"mean":[],"std":[],"max":[],"min":[]}

        for view in ["view1","view2"]:
            video_paths = list((self.root / "videos").rglob(f"{view}/episode_*.mp4"))
            channel_sums = None
            channel_sq_sums = None
            pixel_max = np.zeros(3)
            pixel_min = np.ones(3)
            total_pixels = 0
            for vp in video_paths:
                cap = cv2.VideoCapture(str(vp))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img = frame.astype(np.float32) / 255.0
                    h,w,_ = img.shape
                    pixels = h*w
                    total_pixels += pixels
                    if channel_sums is None:
                        channel_sums = img.sum(axis=(0,1))
                        channel_sq_sums = (img**2).sum(axis=(0,1))
                    else:
                        channel_sums += img.sum(axis=(0,1))
                        channel_sq_sums += (img**2).sum(axis=(0,1))
                    pixel_max = np.maximum(pixel_max, img.reshape(-1,3).max(axis=0))
                    pixel_min = np.minimum(pixel_min, img.reshape(-1,3).min(axis=0))
                cap.release()
            key = f"observation.images.{view}"
            if total_pixels > 0:
                mean = channel_sums / total_pixels
                var = channel_sq_sums / total_pixels - mean**2
                std = np.sqrt(np.maximum(var, 0))
                stats[key] = {
                    "mean": [[[float(m)]] for m in mean.tolist()],
                    "std": [[[float(s)]] for s in std.tolist()],
                    "max": [[[float(m)]] for m in pixel_max.tolist()],
                    "min": [[[float(m)]] for m in pixel_min.tolist()]
                }
            else:
                stats[key] = {"mean":[],"std":[],"max":[],"min":[]}
        with open(self.root / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

    def _write_tasks(self) -> None:
        task_file = self.root / "meta" / "tasks.jsonl"
        with open(task_file, "w") as f:
            f.write(json.dumps({"task_index": 0, "task": self.task_name}) + "\n")

    def _write_episodes(self) -> None:
        ep_file = self.root / "meta" / "episodes.jsonl"
        with open(ep_file, "w") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + "\n")
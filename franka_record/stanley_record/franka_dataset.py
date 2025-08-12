# franka_dataset.py
from pathlib import Path
from typing import List, Dict
import json
import shutil
import numpy as np

from generate.parquet_writer import ParquetWriter
from generate.video_writer import VideoWriterModule
from generate.stats_generator import StatsGenerator
from generate.meta_writer import MetaWriter

class FrankaDataset:
    """
    Public API for building a LeRobot-compatible Franka dataset.
    Supports resume mode to append to existing dataset.
    """
    def __init__(
        self,
        repo_id: str,
        root: Path = None,
        fps: int = 30,
        chunk_size: int = 100,
        task_name: str = "",
        resume: bool = False,
    ):
        self.repo_id = repo_id
        self.task_name = task_name
        self.fps = fps
        self.chunk_size = chunk_size
        # Determine root path
        repo_path = Path(*repo_id.split("/"))
        default_root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_path
        self.root = (root or default_root).expanduser()

        # If not resume, clear existing
        if not resume and self.root.exists():
            shutil.rmtree(self.root)
        # Create subdirs: data, videos, meta
        for sub in ("data", "videos", "meta"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

        # Internal state
        self.episodes: List[Dict] = []
        self.total_frames: int = 0

        # If resume, load existing metadata
        if resume:
            meta_file = self.root / "meta" / "episodes.jsonl"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        self.episodes.append(entry)
                        self.total_frames += entry.get('length', 0)
                print(f"[resume] Loaded {len(self.episodes)} episodes, total_frames={self.total_frames}")

        # Instantiate writers
        self.parquet_writer = ParquetWriter(fps=self.fps)
        self.video_writer = VideoWriterModule(fps=self.fps)
        self.stats_generator = StatsGenerator(
            root=self.root, fps=self.fps, chunk_size=self.chunk_size
        )
        self.meta_writer = None  # to be created in finalize()

    def add_episode(
        self,
        episode_index: int,
        eef_position: np.ndarray,
        eef_orientation_euler: np.ndarray,
        gripper_widths: np.ndarray,
        ext_force: np.ndarray,
        view1_frames: np.ndarray,
        view2_frames: np.ndarray,
    ) -> None:
        chunk_idx = episode_index // self.chunk_size
        chunk_name = f"chunk-{chunk_idx:03d}"
        # Ensure data and video subdirectories exist
        data_dir = self.root / "data" / chunk_name
        data_dir.mkdir(parents=True, exist_ok=True)
        vid_dir_view1 = (
            self.root / "videos" / chunk_name / "observation.images.image_additional_view"
        )
        vid_dir_view2 = (
            self.root / "videos" / chunk_name / "observation.images.image"
        )
        vid_dir_view1.mkdir(parents=True, exist_ok=True)
        vid_dir_view2.mkdir(parents=True, exist_ok=True)

        # Write Parquet
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        self.parquet_writer.write_episode(
            episode_index,
            eef_position,
            eef_orientation_euler,
            gripper_widths,
            ext_force,
            parquet_path,
        )
        # Write Videos (view1 & view2)
        video_path_view1 = vid_dir_view1 / f"episode_{episode_index:06d}.mp4"
        video_path_view2 = vid_dir_view2 / f"episode_{episode_index:06d}.mp4"
        self.video_writer.write_episode(view1_frames, video_path_view1)
        self.video_writer.write_episode(view2_frames, video_path_view2)

        # Update metadata
        frame_count = len(view1_frames)
        entry = {
            "episode_index": episode_index,
            "tasks": [self.task_name],
            "length": frame_count,
        }
        self.episodes.append(entry)
        self.total_frames += frame_count

    def finalize(self) -> None:
        # Generate meta files (tasks.jsonl, episodes.jsonl)
        self.meta_writer = MetaWriter(
            root=self.root,
            fps=self.fps,
            chunk_size=self.chunk_size,
            task_name=self.task_name,
            total_frames=self.total_frames,
            episodes=self.episodes,
        )
        # MetaWriter should overwrite meta files reflecting full list
        self.meta_writer.write_all()
        # Generate stats.json
        self.stats_generator.generate()

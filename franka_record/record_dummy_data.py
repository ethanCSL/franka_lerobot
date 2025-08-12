import argparse
import numpy as np
from pathlib import Path
from franka_dataset import FrankaDataset
import cv2
import time
from tqdm import tqdm

def generate_fake_frames(
    num_frames: int,
    height: int = 480,
    width: int = 640
) -> np.ndarray:
    """
    Generate a block of fake RGB frames.

    Args:
        num_frames: Number of frames to generate.
        height: Frame height in pixels.
        width: Frame width in pixels.

    Returns:
        An array of shape (num_frames, height, width, 3) with uint8 values.
    """
    return np.random.randint(
        0, 256,
        size=(num_frames, height, width, 3),
        dtype=np.uint8
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record synthetic episodes into a LeRobot dataset with live monitoring and controls."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository identifier, e.g. 'ethanCSL/test1'"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Total number of episodes to generate."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for both data and videos."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of episodes per chunk folder."
    )
    parser.add_argument(
        "--episode_time_s",
        type=float,
        default=7200.0,
        help="Each episode length in seconds."
    )
    parser.add_argument(
        "--reset_time_s",
        type=float,
        default=10.0,
        help="Time in seconds to wait between episodes for reset."
    )
    parser.add_argument(
        "--single_task",
        dest="task_name",
        type=str,
        required=True,
        help="Task description for tasks.jsonl and episodes.jsonl entries."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = FrankaDataset(
        repo_id=args.repo_id,
        fps=args.fps,
        chunk_size=args.chunk_size,
        task_name=args.task_name
    )
    num_frames = int(args.episode_time_s * args.fps)
    reset_time = args.reset_time_s

    episode_index = 0
    exit_all = False
    while episode_index < args.num_episodes and not exit_all:
        re_record = False

        # Randomize example sensor inputs
        position = np.random.uniform(-1.0, 1.0, size=3)
        orientation = np.random.uniform(-np.pi, np.pi, size=3)
        gripper_width = float(np.random.uniform(0.0, 0.08))

        # Pre-generate all frames
        view1_full = generate_fake_frames(num_frames)
        view2_full = generate_fake_frames(num_frames)
        record_len = num_frames

        win1, win2 = "View1 Monitor", "View2 Monitor"
        cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win2, cv2.WINDOW_NORMAL)

        # Recording loop with progress bar
        for idx, (f1, f2) in enumerate(
            tqdm(zip(view1_full, view2_full), total=num_frames,
                 desc=f"Ep {episode_index:03d} Frame", ncols=80)
        ):
            cv2.imshow(win1, f1)
            cv2.imshow(win2, f2)
            key = cv2.waitKey(int(1000 / args.fps)) & 0xFF
            if key == 27:
                exit_all = True
                record_len = idx
                break
            elif key == 83:
                record_len = idx
                break
            elif key == 81:
                re_record = True
                break
        cv2.destroyWindow(win1)
        cv2.destroyWindow(win2)

        if exit_all:
            break
        if re_record:
            continue

        # Slice and save
        view1 = view1_full[:record_len]
        view2 = view2_full[:record_len]
        dataset.add_episode(
            episode_index, position, orientation, gripper_width, view1, view2
        )

        if reset_time > 0:
            with tqdm(total=int(reset_time), desc=f"Ep {episode_index:03d} Reset", ncols=80) as pbar:
                start = time.time()
                while time.time() - start < reset_time:
                    elapsed = time.time() - start
                    pbar.n = int(elapsed)
                    pbar.refresh()
                    if cv2.waitKey(100) & 0xFF == 83:
                        break
                    time.sleep(0.1)
                pbar.n = int(reset_time)
                pbar.refresh()

        episode_index += 1

    dataset.finalize()
    print(f"Dataset successfully generated at: {dataset.root}")


if __name__ == "__main__":
    main()

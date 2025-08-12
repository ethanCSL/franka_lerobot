from generate.stats_generator import StatsGenerator
from pathlib import Path

root = Path("/home/csl/.cache/huggingface/lerobot/ethanCSL/0721_change_action")
fps = 30  # 根據你的實際設定
chunk_size = 1000  # 視你的 chunk 大小而定

stats_gen = StatsGenerator(root=root, fps=fps, chunk_size=chunk_size)
stats_gen.generate()



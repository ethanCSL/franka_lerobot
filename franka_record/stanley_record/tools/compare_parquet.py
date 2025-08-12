import pandas as pd
import numpy as np

dataset1_path = "/home/csl/.cache/huggingface/lerobot/ethanCSL/0703_ethan/data/chunk-000/episode_000000.parquet"
dataset2_path = "/home/csl/.cache/huggingface/lerobot/StanleyChueh/franka_lerobot_red_cube/data/chunk-000/episode_000000.parquet"

df1 = pd.read_parquet(dataset1_path)
df2 = pd.read_parquet(dataset2_path)

print("=== Ethan Dataset (First Frame) ===")
obs1 = df1.iloc[0]["observation.state"]
print("Full observation.state:", obs1)
print("Orientation part:", obs1[3:])
print("Orientation length:", len(obs1[3:]))
print("Orientation abs max:", np.max(np.abs(obs1[3:])))
print("Orientation norm (if 4 dims):", np.linalg.norm(obs1[3:]) if len(obs1[3:])==4 else "N/A")

print("\n=== Stanley Dataset (First Frame) ===")
obs2 = df2.iloc[0]["observation.state"]
print("Full observation.state:", obs2)
print("Orientation part:", obs2[3:])
print("Orientation length:", len(obs2[3:]))
print("Orientation abs max:", np.max(np.abs(obs2[3:])))
print("Orientation norm (if 4 dims):", np.linalg.norm(obs2[3:]) if len(obs2[3:])==4 else "N/A")

# Optionally check first 5 frames to see consistency
print("\n=== Checking first 5 frames ===")
for i in range(5):
    o1 = df1.iloc[i]["observation.state"]
    o2 = df2.iloc[i]["observation.state"]
    print(f"\nFrame {i}:")
    print(f" Ethan orientation: {o1[3:]} (len {len(o1[3:])}, norm {np.linalg.norm(o1[3:]) if len(o1[3:])==4 else 'N/A'})")
    print(f" Stanley orientation: {o2[3:]} (len {len(o2[3:])}, norm {np.linalg.norm(o2[3:]) if len(o2[3:])==4 else 'N/A'})")

from scipy.spatial.transform import Rotation as R

# Ethan dataset orientation
ethan_orient = df1.iloc[0]["observation.state"][3:6]
r_ethan = R.from_euler("xyz", ethan_orient)
print("\nEthan rotation matrix:")
print(r_ethan.as_matrix())

# Stanley dataset orientation
stanley_orient = df2.iloc[0]["observation.state"][3:6]
r_stanley = R.from_euler("xyz", stanley_orient)
print("\nStanley rotation matrix:")
print(r_stanley.as_matrix())


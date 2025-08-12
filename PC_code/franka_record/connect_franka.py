import subprocess

commands = [
    "sudo ip addr add 172.16.0.1/24 dev enxc4411e75389a",
    "sudo ip addr flush dev enxc4411e75389a",
    "sudo ip addr add 172.16.0.1/24 dev enxc4411e75389a",
    "sudo ip link set enxc4411e75389a up",
    "sudo ufw disable",
]

# Run each command
for cmd in commands:
    try:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")

# Set environment variable for current process
import os
os.environ["ROS_DOMAIN"] = "0"
print("ROS_DOMAIN set to", os.environ["ROS_DOMAIN"])


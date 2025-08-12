# Franka Lerobot

## Try it now first!
Please make sure every terminal is in ROS1 noetic environment, and check is CSL-FET@TT.
```
source /opt/ros/noetic/setup.bash
```
### Connect to Franka
There're two ways to connect to Franka
1. Command line
```
sudo ip addr add 172.16.0.1/24 dev enxc4411e75389a
sudo ip addr flush dev enxc4411e75389a
sudo ip addr add 172.16.0.1/24 dev enxc4411e75389a
sudo ip link set enxc4411e75389a up
sudo ufw disable
```
2. Run the connection code
```
cd ~/franka_record
python connect_franka.py
```
### Launch Franka_ROS 
```
cd ~/franka_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=172.16.0.2 load_gripper:=true launch_rviz:=false 
```

### Control Robot
```
cd ~/avp_teleoperate_h1/teleop/lerobot_record
source /opt/ros/noetic/setup.bash
python roboticArm_pose_remote_threshold_ros1_test.py 
```

### Vision Pro Control
```
cd ~/avp_teleoperate_h1/teleop/lerobot_record
source /opt/ros/noetic/setup.bash
python teleop_arm_pose_threshold_ros1_test.py 
```

If you have this error when running the above code:

> [Errno 98] error while attempting to bind on address ('0.0.0.0', 8012): address already in use
> Check PID
> ```
> sudo lsof -i :8012
> ```
> Kill PID
> ```
> sudo kill -9 PID
> ```

### LeRobot Record with Franka
#### Record
```
cd ~/franka_record/stanley_record
source /opt/ros/noetic/setup.bash
python record_ros1_test.py --single_task custom_task --repo_id your_huggingface_account/custom_task
```
> Please replace 'custom_task' and 'your_huggingface_account/custom_task' to the actual task and repo_id you want.

#### Visualization

Push to huggingface
```
cd ~/franka_record/
python push_to_hub.py 
```
Note:
> If you have trouble pushing to huggingface, please make sure you have the authentication set in your PC, please follow the instruction below:
> https://hackmd.io/-TIq0K1NROibtOAh4-sfSQ?both=&stext=1313%3A28%3A0%3A1751419547%3Aeqh_-6
> 
Now you can do the visualization!
```
conda activate lerobot
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id your_huggingface_user_name/repo_id
```
![image](https://hackmd.io/_uploads/Hyym1lgHlx.png)


##### If you don't have your own dataset recorded, please try the following command to see the example:
```
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id StanleyChueh/franka_lerobot_red_cube
```
-----------------------------
**Note:**
> Please make sure you have dataset in 
> ```
> ls ~/.cache/huggingface/lerobot/StanleyChueh/franka_lerobot_red_cube
> ```
> Otherwise you should re-clone it: 
> ```
> cd ~/.cache/huggingface/lerobot/StanleyChueh
> git clone https://huggingface.co/datasets/StanleyChueh/franka_lerobot_red_cube
> ```

### LeRobot Replay with Franka
1. Launch Franka ROS
```
cd ~/franka_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=172.16.0.2 load_gripper:=true launch_rviz:=false 
```
2. Set initial position and switch to impedance control
```
cd ~/avp_teleoperate_h1/teleop/lerobot_record
source /opt/ros/noetic/setup.bash
python roboticArm_pose_remote_threshold_ros1_test.py 
```
> ## Note!!!!!!!!!: 
> [INFO] [1751621729.042889]: Switched to cartesian impedance controller.
> Once this message pops out, you can shut this code down.
3. Replay Episode
```
 python ~/franka_record/stanley_record/tools/replay_ros1_v2_quat.py 
```

### LeRobot Training with Franka
```
conda activate lerobot
cd ~/CSL/lerobot_new
python -m lerobot.scripts.train --policy.type=act --dataset.repo_id=user_name/repo_name --output_dir=outputs/train/your_task_name
```
### LeRobot Evaluation with Franka

#### On Agx Orin 
Launch socket server
```
conda activate lerobot
python lerobot/lerobot/scripts/eval_franka_socket_v2.py
```

#### On control PC

##### Launch Franka_ROS
```
cd ~/franka_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch franka_example_controllers cartesian_impedance_example_controller.launch robot_ip:=172.16.0.2 load_gripper:=true launch_rviz:=false 
```

##### Control robot
```
cd ~/avp_teleoperate_h1/teleop/lerobot_record
source /opt/ros/noetic/setup.bash
python roboticArm_pose_remote_threshold_ros1_test.py 
```

> ## Note!!!!!!!!!: 
> [INFO] [1751621729.042889]: Switched to cartesian impedance controller.
> Once this message pops out, you can shut this code down.

##### Publish camera topic
```
python ~/avp_teleoperate_h1/teleop/lerobot_record/eva_data_transform.py 
```

##### Launch socket client
```
python ~/franka_record/stanley_record/tools/replay_ros1_v3_quat.py
```

### 📁 Project Structure

```plaintext
avp_teleoperate_h1/
├── act/
├── assets/
├── img/
├── scripts/
├── teleop/
│   ├── teleop_arm_pose_threshold_ros1_test.py        # 🖐️ Hand tracking via Vision Pro
│   └── roboticArm_pose_remote_threshold_ros1_test.py # 🤖 Franka Panda control

franka_ros/
└── src/
    └── franka_ros/
        └── franka_example_controllers/
            └── launch/
                └── cartesian_impedance_example_controller.launch  # Launch file for Cartesian impedance control

franka_record/
├── franka_dataset.py   # LeRobot-compatible dataset structure
└── record_ros1.py      # ROS1-based recording script
```

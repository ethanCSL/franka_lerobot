# record.py 
import os
import argparse
import rospy
import numpy as np
from pathlib import Path
from franka_dataset import FrankaDataset
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Float32, String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from threading import Thread
from sensor_msgs.msg import Image
import cv2
import time
from tqdm import tqdm
import signal
import sys
import tf

class DataRecorderNode(object):
    def __init__(self, args):
        rospy.init_node('franka_recorder_1', anonymous=False)
        self.bridge = CvBridge()

        self.target_dt = 1.0 / args.fps
        self._last_record_time = rospy.get_time()

        self.KEY_ESC = 27
        self.KEY_LEFT = 81
        self.KEY_RIGHT = 83
        self.KEY_SKIP = ord('s')

        self.pose = None
        self.gripper = None
        self.eye_image = None
        self.agent_image = None
        self.new_pose = False
        self.new_gripper = False
        self.new_eye = False
        self.new_agent = False
        self.records = []

        self.running = True
        self.reset_current_episode = False
        self.finish_current_episode = False
        self.exit_all = False

        rospy.loginfo("Setting log level to INFO")

        # === Robot States === #
        self.pose_sub = rospy.Subscriber(
            '/robot0_eef_pose',
            PoseStamped,
            self.pose_callback,
            queue_size=10
        )
        self.gripper_sub = rospy.Subscriber(
            '/robot0_gripper_width',
            Float32,
            self.gripper_callback,
            queue_size=10
        )
        self.eye_sub = rospy.Subscriber(
            '/robot0_eye_in_hand_image_128',
            Image,
            self.eye_callback,
            queue_size=10
        )
        self.agent_sub = rospy.Subscriber(
            '/robot0_agentview_image_128',
            Image,
            self.agent_callback,
            queue_size=10
        )

        # === Recording status === #
        self.done_pub = rospy.Publisher("/done", Float32, queue_size=1)
        self.done_published = False 

        self.save_sub = rospy.Subscriber(
            '/save',
            Float32,
            self.save_callback,
            queue_size=1
        )

        self.discard_sub = rospy.Subscriber(
            '/discard',
            Float32,
            self.discard_callback,
            queue_size=1
        )
        
        self.reset_sub = rospy.Subscriber(
            '/reset_done',
            Float32,
            self.reset_done_callback,
            queue_size=1
        )

        self.img_dir = os.path.expanduser('~/.local/share/Trash/')
        os.makedirs(self.img_dir, exist_ok=True)
        self.frame_count = 0

        self.win_eye = "Eye Camera Monitor"
        self.win_agent = "Agent View Monitor"

        signal.signal(signal.SIGINT, self._signal_handler)

        self.ros_thread = Thread(target=rospy.spin)
        self.ros_thread.daemon = True
        self.ros_thread.start()

        rospy.loginfo(f"Data recorder initialized. Images will be saved to {self.img_dir}")

    def reset_done_callback(self, msg):
        if msg.data == 1.0:

            self.finish_current_episode = True
            self.reset_current_episode = True
            self.done_published = False
            # if not self.done_published:
            #     #self.done_pub.publish(Float32(1.0))
            #     rospy.loginfo("Published /done = 1 (from /save)")
            #     self.done_published = True
            #     rospy.Timer(rospy.Duration(0.1), self.publish_done_zero, oneshot=True)

    def save_callback(self, msg):
        if msg.data == 1.0:
            #self.finish_current_episode = True
            if not self.done_published:
                self.done_pub.publish(Float32(1.0))
                rospy.loginfo("Published /done = 1 (from /save)")
                self.done_published = True
                rospy.Timer(rospy.Duration(0.1), self.publish_done_zero, oneshot=True)

    def discard_callback(self, msg):
        if msg.data == 1.0:
            #self.reset_current_episode = True
            if not self.done_published:
                self.done_pub.publish(Float32(1.0))
                rospy.loginfo("Published /done = 1 (from /discard)")
                self.done_published = True
                rospy.Timer(rospy.Duration(0.1), self.publish_done_zero, oneshot=True)

    def _signal_handler(self, sig, frame):
        rospy.loginfo("Received interrupt signal, shutting down...")
        self.running = False
        if self.ros_thread and self.ros_thread.is_alive():
            self.ros_thread.join(timeout=1.0)
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        cv2.destroyAllWindows()
        rospy.loginfo("Resources cleaned up")

    def pose_callback(self, msg: PoseStamped):
        try:
            position = msg.pose.position
            orientation = msg.pose.orientation

            quat = [orientation.x, orientation.y, orientation.z, orientation.w]

            self.pose = [position.x, position.y, position.z, quat[0], quat[1], quat[2], quat[3]]
            self.new_pose = True
        except Exception as e:
            rospy.logerr(f"Error in pose callback: {e}")

    def gripper_callback(self, msg: Float32):
        try:
            self.gripper = msg.data
            self.new_gripper = True
        except Exception as e:
            rospy.logerr(f"Error in gripper callback: {e}")

    def gripper_callback(self, msg: Float32):
        self.gripper = msg.data
        if not self.new_gripper:
            rospy.loginfo(f"[Recorder] First gripper reading: {self.gripper}")
        self.new_gripper = True

    def _process_image(self, msg: Image, source_name: str):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = np_arr.reshape((msg.height, msg.width, -1))
            if msg.encoding == 'rgb8':
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                return img
            else:
                return img[:, :, :3]
        except Exception as e:
            rospy.logerr(f"{source_name} image convert error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None

    def eye_callback(self, msg: Image):
        self.eye_image = self._process_image(msg, "Eye")
        if self.eye_image is not None:
            self.new_eye = True
            rospy.logdebug("Successfully processed eye image")

    def agent_callback(self, msg: Image):
        self.agent_image = self._process_image(msg, "Agent view")
        if self.agent_image is not None:
            self.new_agent = True
            rospy.logdebug("Successfully processed agent image")

    def try_record(self):
        if self.pose is None or self.gripper is None or \
           self.eye_image is None or self.agent_image is None:
            return False

        if self.pose is None or not self.new_gripper or \
        self.eye_image is None or self.agent_image is None:
            return False

        now_t = rospy.get_time()
        if now_t - self._last_record_time < self.target_dt:
            return False
        self._last_record_time = now_t

        px, py, pz, quat_x, quat_y, quat_z,quat_w = self.pose
        gripper = self.gripper
        ts = now_t

        self.records.append({
            'timestamp': ts,
            'pos_x': px, 'pos_y': py, 'pos_z': pz,
            'quat_x': quat_x, 'quat_y': quat_y, 'quat_z': quat_z,'quat_w': quat_w,
            'gripper_width': gripper,
            'eye_image': self.eye_image.copy(),
            'agent_image': self.agent_image.copy(),
        })

        self.frame_count += 1
        return True

    def convert_to_dataset_format(self):
        frame_count = len(self.records)
        if frame_count == 0:
            rospy.logwarn("No frames recorded!")
            return None, None, None, None, None

        rospy.loginfo(f"Converting {frame_count} frames to dataset format...")

        eef_position = np.zeros((frame_count, 3))
        eef_orientation = np.zeros((frame_count, 4))
        gripper_width = np.zeros(frame_count)
        view1_frames = []
        view2_frames = []

        for i, record in enumerate(self.records):
            eef_position[i] = [record['pos_x'], record['pos_y'], record['pos_z']]
            eef_orientation[i] = [record['quat_x'], record['quat_y'], record['quat_z'], record['quat_w']]
            gripper_width[i] = record['gripper_width']

            try:
                eye_img = record['eye_image']
                view1_frames.append(eye_img)

                agent_img = record['agent_image']

                if agent_img is None:
                    rospy.logwarn(f"Failed to read agent image: {record['agent_image_path']}")
                else:
                    view2_frames.append(agent_img)
            except Exception as e:
                rospy.logerr(f"Error processing image at index {i}: {e}")

        if not view1_frames or not view2_frames:
            rospy.logerr("No valid frames found!")
            return None, None, None, None, None

        rospy.loginfo(f"Successfully converted {len(view1_frames)} valid frames")
        return eef_position, eef_orientation, gripper_width, np.array(view1_frames), np.array(view2_frames)

    def process_keyboard_input(self):
        key = cv2.waitKey(1) & 0xFF

        if key in [self.KEY_ESC, self.KEY_RIGHT, self.KEY_LEFT]:
            if not self.done_published:
                #self.done_pub.publish(Float32(1.0))
                rospy.loginfo("Published /done = 1")
                self.done_published = True
                # After a short delay, publish 0
                rospy.Timer(rospy.Duration(0.1), self.publish_done_zero, oneshot=True)

            if key == self.KEY_ESC:
                self.exit_all = True
                print("用戶按下ESC，退出記錄")
            elif key == self.KEY_RIGHT:
                self.finish_current_episode = True
                print("用戶按下右鍵，完成當前集")
            elif key == self.KEY_LEFT:
                self.reset_current_episode = True
                print("用戶按下左鍵，重新開始當前集")

            return True

        return False

    def publish_done_zero(self, event):
        #self.done_pub.publish(Float32(0.0))
        rospy.loginfo("Published /done = 0")

def main() -> None:
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
            default=150,
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
            default= 5.0,
            help="Time in seconds to wait between episodes for reset."
        )
        parser.add_argument(
            "--single_task",
            dest="task_name",
            type=str,
            required=True,
            help="Task description for tasks.jsonl and episodes.jsonl entries."
        )
        parser.add_argument(
            "--output",
            type=str,
            default="franka_dataset",
            help="Base name for output directories"
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="If set, will append new episodes to existing dataset instead of overwriting"
        )
        return parser.parse_args()

    try:
        args = parse_args()

        recorder = DataRecorderNode(args)

        phase_pub = rospy.Publisher("/recording_phase", String, queue_size=1)

        episode_pub = rospy.Publisher('/episode_index', String, queue_size=1) 


        dataset = FrankaDataset(
            repo_id=args.repo_id,
            fps=args.fps,
            chunk_size=args.chunk_size,
            task_name=args.task_name,
            resume=args.resume
        )

            # —— 新增：如果要 resume，就讀出已存在的 episodes 數量，從下一集開始 —— #
        if args.resume:
            # 假設 FrankaDataset 有 episodes 屬性紀錄已加載的集數
            try:
                start_index = len(dataset.episodes)
                print(f"Resume 模式：已偵測到 {start_index} 集，將從第 {start_index} 集開始接續錄製")
            except Exception:
                # 若 FrankaDataset 沒有 episodes 屬性，可改用檔案系統直接數目錄
                data_root = Path(dataset.root)  # dataset.root 為儲存路徑
                existing = [d for d in data_root.glob("chunk_*") for f in d.glob("episode_*.npz")]
                start_index = len(existing)
                print(f"Resume 模式（備援檢測）：已偵測到 {start_index} 集，將從第 {start_index} 集開始接續錄製")
                
            args.num_episodes += start_index
            print(f"[自動補足] 目標總集數已更新為: {args.num_episodes}")
        else:
            start_index = 0

        num_frames = int(args.episode_time_s * args.fps)
        reset_time = args.reset_time_s

        episode_index = start_index

        while episode_index < args.num_episodes and not recorder.exit_all:

            episode_pub.publish(String(f"Episode: {episode_index - 1}"))

            recorder.records = []
            recorder.frame_count = 0
            recorder.reset_current_episode = False
            recorder.finish_current_episode = False

            # 清空上一集的影像快取
            recorder.eye_image = None
            recorder.agent_image = None
            recorder.new_eye = False
            recorder.new_agent = False

            # 等待新的影像到來
            timeout = 5.0
            start = time.time()
            while (recorder.eye_image is None or recorder.agent_image is None) and time.time() - start < timeout:

                rospy.loginfo_throttle(1.0, "等待新的相機影像...")

                time.sleep(0.05)

            recorder.new_eye = False
            recorder.new_agent = False

            phase_pub.publish(String("start recording"))

            print(f"\n準備記錄第 {episode_index} 集影像...")
            print("按ESC退出整個記錄過程，按左鍵(←)重新記錄當前集，按右鍵(→)完成當前集")

            cv2.namedWindow(recorder.win_eye, cv2.WINDOW_NORMAL)
            cv2.namedWindow(recorder.win_agent, cv2.WINDOW_NORMAL)

            start_time = time.time()
            with tqdm(total=num_frames, desc=f"Ep {episode_index:03d} Frame", ncols=80) as pbar:
                while len(recorder.records) < num_frames:
                    if recorder.try_record():
                        pbar.n = len(recorder.records)
                        pbar.refresh()

                    if recorder.eye_image is not None and recorder.agent_image is not None:
                        cv2.imshow(recorder.win_eye, recorder.eye_image)
                        cv2.imshow(recorder.win_agent, recorder.agent_image)

                    recorder.process_keyboard_input()

                    if recorder.exit_all:
                        break
                    if recorder.finish_current_episode:
                        recorder.finish_current_episode = False
                        print(f"recorder: {recorder.finish_current_episode}")
                        break
                    if recorder.reset_current_episode:
                        break

                    time.sleep(0.01)
                        
                    if time.time() - start_time > args.episode_time_s * 1.5:
                        print("記錄超時，跳過該集")
                        break

            cv2.destroyWindow(recorder.win_eye)
            cv2.destroyWindow(recorder.win_agent)

            if recorder.exit_all:
                break   

            # 如果是按下左鍵 (重置要求)
            if recorder.reset_current_episode:
                print(f"捨棄第 {episode_index} 集的紀錄，準備重置後重新錄製。")
                # 不做任何事，直接跳到下面的 reset_time 區塊
                # 同時 episode_index 維持不變，下一輪就會重錄本集
            
            # 如果有錄到數據 (正常完成或按右鍵)
            elif len(recorder.records) > 0:
                eef_position, eef_orientation, gripper_width, view1_frames, view2_frames = recorder.convert_to_dataset_format()

                if view1_frames is not None and len(view1_frames) > 0:
                    try:
                        dataset.add_episode(
                            episode_index, eef_position, eef_orientation, gripper_width,
                            view1_frames, view2_frames
                        )
                        print(f"成功記錄第 {episode_index} 集 ({len(recorder.records)} 幀)")
                        # 只有在成功儲存後，才將集數 +1
                        episode_index += 1
                    except Exception as e:
                        print(f"添加集到數據集失敗: {e}。準備重試本集。")
                        # 添加失敗，不增加 episode_index，稍後重試本集
                else:
                    print("警告: 轉換後無有效數據，準備重試本集。")
                    # 轉換失敗，不增加 episode_index，稍後重試本集
            else:
                # 如果因為超時等原因跳出且沒錄到任何數據
                print(f"警告: 第 {episode_index} 集無有效數據，準備重試本集。")
                # 同樣不增加 episode_index

            # 以下區塊現在會在「成功儲存」、「要求重置」或「錄製失敗」後執行
            if reset_time > 0:
                phase_pub.publish(String("reset"))
                with tqdm(total=int(reset_time), desc=f"Ep {episode_index:03d} Reset", ncols=80) as pbar:
                    start = time.time()
                    while time.time() - start < reset_time:
                        elapsed = time.time() - start
                        pbar.n = int(elapsed)
                        pbar.refresh()
                        key = cv2.waitKey(100) & 0xFF
                        if key == recorder.KEY_SKIP:
                            print("用戶按下S，跳過等待時間")
                            break
                        time.sleep(0.1)
                    pbar.n = int(reset_time)
                    pbar.refresh()

            recorder.done_published = False

        print("完成所有記錄，生成最終數據集...")
        print(f"[Debug] episodes accumulated: {len(dataset.episodes)}")
        dataset.finalize()
        print(f"數據集成功生成於: {dataset.root}")

    except KeyboardInterrupt:
        print("\n用戶中斷記錄過程")
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("清理資源...")
        if 'recorder' in locals():
            recorder.running = False
            if hasattr(recorder, 'ros_thread') and recorder.ros_thread:
                recorder.ros_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("程序已安全退出")


if __name__ == "__main__":
    main()


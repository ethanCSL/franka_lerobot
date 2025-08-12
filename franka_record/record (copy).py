# record.py (改進版本)#
import os
import argparse
import rclpy
import numpy as np
from pathlib import Path
from franka_dataset import FrankaDataset
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from cv_bridge import CvBridge
from threading import Event, Thread
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import time
from tqdm import tqdm
import signal
import sys


class DataRecorderNode(Node):
    def __init__(self, args):
        super().__init__('franka_recorder')
        self.bridge = CvBridge()

        self.target_dt = 1.0 / args.fps
        self._last_record_time = self.get_clock().now().nanoseconds * 1e-9

        # 定義按鍵常量 - 使用 OpenCV 標準鍵碼
        self.KEY_ESC = 27
        self.KEY_LEFT = 81  # OpenCV 中的左箭頭鍵
        self.KEY_RIGHT = 83  # OpenCV 中的右箭頭鍵
        self.KEY_SKIP = ord('s')  # s 鍵跳過等待

        # 緩存最新資料
        self.pose = None
        self.gripper = None
        self.eye_image = None
        self.agent_image = None
        self.new_pose = False
        self.new_gripper = False
        self.new_eye = False
        self.new_agent = False
        self.records = []  # 存儲各幀資料的清單
        
        # 控制標誌
        self.running = True
        self.reset_current_episode = False
        self.finish_current_episode = False
        self.exit_all = False
        
        # 日誌設置
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        # 創建ROS2訂閱
        self.create_subscription(
            Float64MultiArray,
            '/robot0_eef_pose',
            self.pose_callback,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/robot0_gripper_width',
            self.gripper_callback,
            10
        )
        self.create_subscription(
            Image,
            '/robot0_eye_in_hand_image',
            self.eye_callback,
            10
        )
        self.create_subscription(
            Image,
            '/robot0_agentview_image',
            self.agent_callback,
            10
        )  

        # 設置輸出目錄
        # self.img_dir = args.output + '_imgs' if hasattr(args, 'output') else 'recorded_imgs'
        self.img_dir = '.local/share/Trash/'
        os.makedirs(self.img_dir, exist_ok=True)
        self.frame_count = 0
        
        # 設置視窗標題
        self.win_eye = "Eye Camera Monitor"
        self.win_agent = "Agent View Monitor"
        
        # 註冊關閉處理函數
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # 創建 ROS2 專用線程
        self.ros_thread = None
        
        self.get_logger().info(f"Data recorder initialized. Images will be saved to {self.img_dir}")
            
    def _signal_handler(self, sig, frame):
        """處理信號中斷，確保資源釋放"""
        self.get_logger().info("Received interrupt signal, shutting down...")
        self.running = False
        if self.ros_thread and self.ros_thread.is_alive():
            self.ros_thread.join(timeout=1.0)
        self._cleanup()
        sys.exit(0)
        
    def _cleanup(self):
        """清理資源"""
        cv2.destroyAllWindows()
        self.get_logger().info("Resources cleaned up")
        
    def pose_callback(self, msg: Float64MultiArray):
        """處理機器人位姿數據"""
        try:
            if len(msg.data) >= 6:
                self.pose = list(msg.data[:6])
                self.new_pose = True
            else:
                self.get_logger().warning("Pose data incomplete")
        except Exception as e:
            self.get_logger().error(f"Error in pose callback: {e}")

    def gripper_callback(self, msg: Float32MultiArray):
        """處理夾持器數據"""
        try:
            if len(msg.data) > 0:
                self.gripper = float(msg.data[0])
                self.new_gripper = True
            else:
                self.get_logger().warning("Gripper data empty")
        except Exception as e:
            self.get_logger().error(f"Error in gripper callback: {e}")

    def _process_image(self, msg: Image, source_name: str):
        """通用圖像處理函數，避免代碼重複"""
        try:
            # 使用更直接的方式將ROS圖像轉換為OpenCV圖像
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = np_arr.reshape((msg.height, msg.width, -1))
            
            # 確保顏色通道正確
            if msg.encoding == 'rgb8':
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                return img
            else:
                # 嘗試使用簡化版本的cvtColor來處理其他編碼
                return img[:, :, :3]  # 取前三個通道
        except Exception as e:
            self.get_logger().error(f"{source_name} image convert error: {e}")
            # 記錄更多錯誤信息以幫助調試
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None

    def eye_callback(self, msg: Image):
        """處理眼部相機圖像"""
        self.eye_image = self._process_image(msg, "Eye")
        if self.eye_image is not None:
            self.new_eye = True
            self.get_logger().debug("Successfully processed eye image")

    def agent_callback(self, msg: Image):
        """處理機器人視角圖像"""
        self.agent_image = self._process_image(msg, "Agent view")
        if self.agent_image is not None:
            self.new_agent = True
            self.get_logger().debug("Successfully processed agent image")

    def try_record(self):
        if self.pose is None or self.gripper is None or \
        self.eye_image is None or self.agent_image is None:
            return False

        now_t = self.get_clock().now().nanoseconds * 1e-9
        if now_t - self._last_record_time < self.target_dt:
            return False
        self._last_record_time = now_t

        px, py, pz, roll, pitch, yaw = self.pose
        gripper = self.gripper
        ts = now_t

        eye_fn = os.path.join(self.img_dir, f"frame{self.frame_count:06d}_eye.png")
        agent_fn = os.path.join(self.img_dir, f"frame{self.frame_count:06d}_agent.png")
        cv2.imwrite(eye_fn, self.eye_image)
        cv2.imwrite(agent_fn, self.agent_image)

        self.records.append({
        'timestamp': ts,
        'pos_x': px, 'pos_y': py, 'pos_z': pz,
        'roll':roll,'pitch':pitch,'yaw':yaw,
        'gripper_width': gripper,
        'eye_image_path': eye_fn,
        'agent_image_path': agent_fn,
        })
        self.frame_count += 1
        return True

    def convert_to_dataset_format(self):
        """將記錄的數據轉換為FrankaDataset所需的格式"""
        frame_count = len(self.records)
        if frame_count == 0:
            self.get_logger().warning("No frames recorded!")
            return None, None, None, None, None
        
        self.get_logger().info(f"Converting {frame_count} frames to dataset format...")
        
        eef_position = np.zeros((frame_count, 3))
        eef_orientation = np.zeros((frame_count, 3))
        gripper_width = np.zeros(frame_count)
        view1_frames = []
        view2_frames = []
        
        for i, record in enumerate(self.records):
            eef_position[i] = [record['pos_x'], record['pos_y'], record['pos_z']]
            eef_orientation[i] = [record['roll'], record['pitch'], record['yaw']]
            gripper_width[i] = record['gripper_width']
            
            # 添加錯誤處理，確保圖像檔案存在
            try:
                eye_img = cv2.imread(record['eye_image_path'])
                if eye_img is None:
                    self.get_logger().warning(f"Failed to read eye image: {record['eye_image_path']}")
                else:
                    view1_frames.append(eye_img)
                    
                agent_img = cv2.imread(record['agent_image_path'])
                if agent_img is None:
                    self.get_logger().warning(f"Failed to read agent image: {record['agent_image_path']}")
                else:
                    view2_frames.append(agent_img)
            except Exception as e:
                self.get_logger().error(f"Error processing image at index {i}: {e}")
        
        # 驗證數據
        if not view1_frames or not view2_frames:
            self.get_logger().error("No valid frames found!")
            return None, None, None, None, None
            
        self.get_logger().info(f"Successfully converted {len(view1_frames)} valid frames")
        return eef_position, eef_orientation, gripper_width, np.array(view1_frames), np.array(view2_frames)
    
    def start_ros_thread(self):
        """啟動專用線程處理ROS回調"""
        def _ros_spin():
            while self.running:
                rclpy.spin_once(self, timeout_sec=0.01)
                time.sleep(0.001)  # 小休眠防止CPU佔用過高

        self.ros_thread = Thread(target=_ros_spin)
        self.ros_thread.daemon = True
        self.ros_thread.start()
    
    def process_keyboard_input(self):
        """處理鍵盤輸入，這個函數會在主線程中調用"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == self.KEY_ESC:  # ESC
            self.exit_all = True
            print("用戶按下ESC，退出記錄")
            return True
        elif key == self.KEY_RIGHT:  # 右鍵(→)
            self.finish_current_episode = True
            print("用戶按下右鍵，完成當前集")
            return True
        elif key == self.KEY_LEFT:  # 左鍵(←)
            self.reset_current_episode = True
            print("用戶按下左鍵，重新開始當前集")
            return True
        
        # 添加更多按鍵檢測，例如空格暫停等
        return False


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
        parser.add_argument(
            "--output",
            type=str,
            default="franka_dataset",
            help="Base name for output directories"
        )
        return parser.parse_args()

    # 確保在退出時正確清理資源
    try:
        args = parse_args()
        
        rclpy.init()
        recorder = DataRecorderNode(args)
        
        # 啟動 ROS 專用線程
        recorder.start_ros_thread()
        
        dataset = FrankaDataset(
            repo_id=args.repo_id,
            fps=args.fps,
            chunk_size=args.chunk_size,
            task_name=args.task_name
        )
        
        num_frames = int(args.episode_time_s * args.fps)
        reset_time = args.reset_time_s
        
        episode_index = 0
        
        while episode_index < args.num_episodes and not recorder.exit_all:
            # 清除上一次記錄的數據
            recorder.records = []
            recorder.frame_count = 0
            recorder.reset_current_episode = False
            recorder.finish_current_episode = False
            
            print(f"\n準備記錄第 {episode_index} 集影像...")
            print("按ESC退出整個記錄過程，按左鍵(←)重新記錄當前集，按右鍵(→)完成當前集")
            
            cv2.namedWindow(recorder.win_eye, cv2.WINDOW_NORMAL)
            cv2.namedWindow(recorder.win_agent, cv2.WINDOW_NORMAL)
            
            # 開始記錄
            start_time = time.time()
            with tqdm(total=num_frames, desc=f"Ep {episode_index:03d} Frame", ncols=80) as pbar:
                while len(recorder.records) < num_frames:
                    # 嘗試記錄一幀
                    if recorder.try_record():
                        # 更新進度條
                        pbar.n = len(recorder.records)
                        pbar.refresh()
                    
                    # 顯示最新的圖像
                    if recorder.eye_image is not None and recorder.agent_image is not None:
                        cv2.imshow(recorder.win_eye, recorder.eye_image)
                        cv2.imshow(recorder.win_agent, recorder.agent_image)
                    
                    # 處理鍵盤輸入
                    recorder.process_keyboard_input()
                    
                    # 檢查控制標誌
                    if recorder.exit_all:
                        break
                    if recorder.finish_current_episode:
                        recorder.finish_current_episode = False
                        break
                    if recorder.reset_current_episode:
                        recorder.reset_current_episode = False
                        recorder.records = []
                        recorder.frame_count = 0
                        pbar.n = 0
                        pbar.refresh()
                    
                    # 加入短暫休眠，減少CPU使用率
                    time.sleep(0.01)
                    
                    # 如果超時則退出
                    if time.time() - start_time > args.episode_time_s * 1.5:
                        print("記錄超時，跳過該集")
                        break
            
            cv2.destroyWindow(recorder.win_eye)
            cv2.destroyWindow(recorder.win_agent)
            
            if recorder.exit_all or len(recorder.records) == 0:
                break
            
            # 轉換記錄的數據為數據集格式
            eef_position, eef_orientation, gripper_width, view1_frames, view2_frames = recorder.convert_to_dataset_format()
            
            if view1_frames is not None and len(view1_frames) > 0:
                # 添加到數據集
                try:
                    dataset.add_episode(
                        episode_index, eef_position, eef_orientation, gripper_width, 
                        view1_frames, view2_frames
                    )
                    
                    print(f"成功記錄第 {episode_index} 集 ({len(recorder.records)} 幀)")
                except Exception as e:
                    print(f"添加集到數據集失敗: {e}")
                    continue
                
                # 等待重置
                if reset_time > 0:
                    with tqdm(total=int(reset_time), desc=f"Ep {episode_index:03d} Reset", ncols=80) as pbar:
                        start = time.time()
                        while time.time() - start < reset_time:
                            elapsed = time.time() - start
                            pbar.n = int(elapsed)
                            pbar.refresh()
                            # 處理鍵盤輸入
                            key = cv2.waitKey(100) & 0xFF
                            if key == recorder.KEY_SKIP:  # 按S跳過等待
                                print("用戶按下S，跳過等待時間")
                                break
                            time.sleep(0.1)
                        pbar.n = int(reset_time)
                        pbar.refresh()
                
                episode_index += 1
            else:
                print("警告: 無有效數據，跳過該集")
        
        # 結束記錄
        print("完成所有記錄，生成最終數據集...")
        dataset.finalize()
        print(f"數據集成功生成於: {dataset.root}")
        
    except KeyboardInterrupt:
        print("\n用戶中斷記錄過程")
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # 確保資源正確釋放
        print("清理資源...")
        if 'recorder' in locals():
            recorder.running = False
            if hasattr(recorder, 'ros_thread') and recorder.ros_thread:
                recorder.ros_thread.join(timeout=1.0)
            recorder.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        print("程序已安全退出")

if __name__ == "__main__":
    main()

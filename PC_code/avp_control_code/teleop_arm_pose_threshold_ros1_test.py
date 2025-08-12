#teleop_arm_pose_threshold_ros1_test.py
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String, Bool
import numpy as np
import cv2
import time
import yaml
from pathlib import Path
from multiprocessing import shared_memory, Queue, Event
import signal
from pytransform3d import rotations
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs 
import threading

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessorLegacy as VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig

class DisplacementPublisher:
    def __init__(self):
        #===Publisher===#
        self.publisher = rospy.Publisher('/displacement', Float32MultiArray, queue_size=10)
        self.publisher2 = rospy.Publisher('/finger', Float32, queue_size=10)
        self.vision_pro_status_pub = rospy.Publisher('/vision_pro_status', Bool, queue_size=10)
        self.image_pub = rospy.Publisher('/robot0_agentview_image', Image, queue_size=10)
        self.image_pub_128 = rospy.Publisher('/robot0_agentview_image_128', Image, queue_size=10)
        self.eyeinhand_pub = rospy.Publisher('/robot0_eye_in_hand_image', Image, queue_size=10)
        self.eyeinhand_pub_128 = rospy.Publisher('/robot0_eye_in_hand_image_128', Image, queue_size=10)

        self.bridge = CvBridge()
        self.active = False
        self.rs_img = None

        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.rs_pipeline.start(self.rs_config)
        except Exception as e:
            rospy.logerr(f"Failed to initialize RealSense: {e}")
            self.rs_pipeline = None

        if self.rs_pipeline:
            threading.Thread(target=self.rs_loop, daemon=True).start()
            rospy.loginfo("RealSense thread started")

    # === vision pro active status === #
    def set_active(self, is_active):
        self.active = is_active
        print(self.active)
        self.vision_pro_status_pub.publish(self.active)

    # === publish user's wrist and camera  === #
    def publish_displacement(self, displacement):
        if not self.active:
            return
        msg = Float32MultiArray()
        msg.data = displacement
        self.publisher.publish(msg)

    def publish_finger(self,distance):
        if not self.active:
            return
        msg = Float32()
        msg.data = distance
        self.publisher2.publish(msg)

    def publish_agentview_image(self, cv_image):
        if not self.active:                       #cancle here
            return
        resized = cv2.resize(cv_image, (640, 480))  # 640x480
        msg = self.bridge.cv2_to_imgmsg(resized, encoding="rgb8")
        self.image_pub.publish(msg)

    def publish_agentview_image_128(self, cv_image):
        if not self.active:                       #cancle here
            return
        resized_ = cv2.resize(cv_image, (128, 128))  # 640x480
        msg_ = self.bridge.cv2_to_imgmsg(resized_, encoding="rgb8")
        self.image_pub_128.publish(msg_)

    def rs_loop(self):
        while not rospy.is_shutdown():
            # if not self.active:
            #     return
            if not self.rs_pipeline:
                break
                
            try:
                frames = self.rs_pipeline.poll_for_frames()
                if frames and frames.get_color_frame():
                    color_frame = frames.get_color_frame()
                    rs_img = np.asanyarray(color_frame.get_data())
                    
                    # 原圖
                    msg = self.bridge.cv2_to_imgmsg(rs_img, encoding='bgr8')
                    self.eyeinhand_pub.publish(msg)

                    # 壓縮後圖
                    resized_128 = cv2.resize(rs_img, (128, 128))
                    msg_128 = self.bridge.cv2_to_imgmsg(resized_128, encoding='bgr8')
                    self.eyeinhand_pub_128.publish(msg_128)
                    
            except Exception as e:
                rospy.logwarn(f"RealSense error: {e}")
                
            time.sleep(0.001)

    def __del__(self):
        if hasattr(self, 'rs_pipeline') and self.rs_pipeline:
            self.rs_pipeline.stop()
            rospy.loginfo("Stopped RealSense pipeline")
    
class USBCameraSystem:
    def __init__(self, camera_id=0, resolution=(720, 1280)):
        self.resolution = resolution
        self.cam = cv2.VideoCapture(camera_id)

        if not self.cam.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[1])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[0])

    def get_frames(self, zoom_factor= 1.0):
        ret, frame = self.cam.read()
        if not ret:
            print("Failed to capture frame from camera")
            return None, None

        if frame.shape[:2] != self.resolution:
            frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))

        # === Digital zoom === #
        # if zoom_factor > 1.0:
        #     h, w = frame.shape[:2]
        #     new_w = int(w / zoom_factor)
        #     new_h = int(h / zoom_factor)
        #     start_x = (w - new_w) // 2
        #     start_y = (h - new_h) // 2
        #     cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]
        #     frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        if zoom_factor >= 1.0:
            h, w = frame.shape[:2]
            new_w = int(w / zoom_factor)
            new_h = int(h / zoom_factor)
            start_x = (w - new_w) // 2
            start_y = max((h // 2) - new_h, 0)  # From middle upward

            cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.copy(), frame.copy()

    def release(self):
        self.cam.release()

class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('../../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
 
        thumb_tip = left_hand_mat[tip_indices][0]  # Thumb tip (3D position)
        index_tip = left_hand_mat[tip_indices][1]  # Index finger tip (3D position)
        right_thumb_tip = right_hand_mat[tip_indices][0]
        right_index_tip = right_hand_mat[tip_indices][1] 
        right_middle_tip = right_hand_mat[tip_indices][2]
        right_ring_tip = right_hand_mat[tip_indices][3]

        right_distance = float(np.linalg.norm(right_thumb_tip - right_index_tip))
        right_middle_distance = float(np.linalg.norm(right_thumb_tip - right_middle_tip))
        right_ring_distance = float(np.linalg.norm(right_thumb_tip - right_ring_tip))
        distance = float(np.linalg.norm(thumb_tip - index_tip))
        return head_rmat, left_pose, right_pose, left_qpos, right_qpos, distance, right_distance, right_middle_distance, right_ring_distance

class Sim:
    def __init__(self, print_freq=True):
        self.publisher = DisplacementPublisher()
        self.print_freq = print_freq
        self.target_pose = np.array([-0.4, 0.08, 1.1]) 
        self.threshold = 0.1
        self.camera_system = USBCameraSystem()
        self.active_position = None
        self.active_orientation = None
        self.status = "Inactive"
        self.displacement = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.left_finger = np.zeros(12)
        self.right_finger = np.zeros(12)
        self.previous_pose = np.array([-0.45, 0.25,1.6]) #np.array([-0.45, 0.16, 1.34])
        self.right_distance = 0.0
        self.distance = 0.0
        self.phase = "reset"
        self.episode_info = "Episode: " 


        #=== Publisher ===#
        # Save and discard episode by using vision pro 
        self.save_pub = rospy.Publisher('/save', Float32, queue_size=10)
        self.discard_pub = rospy.Publisher('/discard', Float32, queue_size=10)

        rospy.Subscriber("/recording_phase", String, self.phase_callback)
        rospy.Subscriber("/episode_index", String, self.episode_callback)


    def phase_callback(self, msg):
        self.phase = msg.data    

    def episode_callback(self, msg):
        self.episode_info = msg.data

    def step(self, head_rmat, left_pose, right_qpos, left_qpos, right_pose, distance, right_distance, right_middle_distance, right_ring_distance):
        if self.print_freq:
            start = time.time()

        left_position = np.array(left_pose[:3])
        left_orientation = np.array(left_pose[3:6])
        right_position = np.array(right_pose[:3])
        right_orientation = np.array(right_pose[3:6])
        self.distance = distance
        self.right_distance = right_distance
        self.right_middle_distance = right_middle_distance
        self.ring_distance = right_ring_distance

        if self.status == "Inactive": 
            if self.ring_distance <= 0.01 and self.right_distance >= 0.07:
                self.status = "Active" 
                self.publisher.set_active(True) 
                if self.active_position is None:
                    self.active_position = left_position
                    self.active_orientation = np.array([-0.45, 0, left_orientation[2]])    
                self.previous_pose = left_position  
                
        else:
            if self.previous_pose is not None:
                displacement_posi = left_position - self.active_position 
                left_orientation = np.array([left_orientation[1], left_orientation[0], left_orientation[2]])
                displacement_orien = left_orientation - self.active_orientation
                self.displacement = np.concatenate(( displacement_posi, displacement_orien))
                self.left_finger = np.array(left_qpos)
                self.right_finger = np.array(right_qpos)# right hand
                if self.right_distance <= 0.01:
                    self.status = "Inactive"
                    self.publisher.set_active(False)
                    
                    self.save_pub.publish(Float32(1.0))
                    rospy.loginfo("Published /save = 1.0")

                if self.right_middle_distance <= 0.01:
                    self.status = "Inactive"
                    self.publisher.set_active(False)

                    self.discard_pub.publish(Float32(1.0))
                    rospy.loginfo("Published /discard = 1.0")

            else:
                self.displacement = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.right_displacement = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # right hand
                
            self.previous_pose = left_position
            self.publisher.publish_displacement(self.displacement.tolist())
            self.publisher.publish_finger(self.distance)

        left_image, right_image = self.camera_system.get_frames()
        if left_image is None or right_image is None:
            left_image = np.zeros((self.camera_system.resolution[0], self.camera_system.resolution[1], 3), dtype=np.uint8)
            right_image = np.zeros((self.camera_system.resolution[0], self.camera_system.resolution[1], 3), dtype=np.uint8)
        else:
            self.publisher.publish_agentview_image(left_image)
            self.publisher.publish_agentview_image_128(left_image)

            # displacement_text = [
            #         f"dx={self.displacement[0]:.2f}",
            #         f"dy={self.displacement[1]:.2f}",
            #         f"dz={self.displacement[2]:.2f}",
            #         f"roll={self.displacement[3]:.2f}",
            #         f"pitch={self.displacement[4]:.2f}",
            #         f"yaw={self.displacement[5]:.2f}"
            #     ]   
            # for i, line in enumerate(displacement_text):
            #     cv2.putText(right_image, line, (600, 300 + i * 50), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
            color = (0, 255, 0) if self.status == "Active" else (255, 0, 0)
            cv2.putText(left_image, f"Status: {self.status}", (400, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            phase_color = (0, 255, 0) if self.phase == "start recording" else (255, 0, 0)
            cv2.putText(left_image, self.phase, (400, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, phase_color, 2)
            
            episode_color = (0, 255, 255)  
            cv2.putText(left_image, self.episode_info, (600, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, episode_color, 2)

            left_image = cv2.cvtColor(cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
            right_image = cv2.cvtColor(cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)

        if self.print_freq:
            end = time.time()
        return left_image, right_image

    def end(self):
        self.camera_system.release()
        rospy.signal_shutdown("shutdown requested")

def signal_handler(sig, frame):
    print("You pressed Ctrl+C! Exiting gracefully...")
    simulator.end()     
    exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler) 
    try:
        rospy.init_node('teleop_sim_node') 
        teleoperator = VuerTeleop('../inspire_hand.yml')
        simulator = Sim()

        test_frame = simulator.camera_system.get_frames()[0]
        if test_frame is None:
            raise RuntimeError("Camera initialization failed")

        while not rospy.is_shutdown():
            head_rmat, left_pose, right_pose, left_qpos, right_qpos, distance, right_distance, right_middle_distance, right_ring_distance = teleoperator.step()
            left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos, distance, right_distance, right_middle_distance, right_ring_distance)
            result = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos, distance, right_distance, right_middle_distance, right_ring_distance)

            if left_img is not None and right_img is not None:
                np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))

    except KeyboardInterrupt:
        simulator.end()
        exit(0)
    except Exception as e:
        print(f"Error occurred: {e}")
        simulator.end()
        exit(1)




#teleop_arm_pose_threshold_ros1_test.py
# import rospy
# from sensor_msgs.msg import Image
# from std_msgs.msg import Float32, Float32MultiArray, String, Bool
# import numpy as np
# import cv2
# import time
# import yaml
# from pathlib import Path
# from multiprocessing import shared_memory, Queue, Event
# import signal
# from pytransform3d import rotations
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import pyrealsense2 as rs 
# import threading

# from TeleVision import OpenTeleVision
# from Preprocessor import VuerPreprocessorLegacy as VuerPreprocessor
# from constants_vuer import tip_indices
# from dex_retargeting.retargeting_config import RetargetingConfig

# class DisplacementPublisher:
#     def __init__(self):
#         #===Publisher===#
#         self.publisher = rospy.Publisher('/displacement', Float32MultiArray, queue_size=10)
#         self.publisher2 = rospy.Publisher('/finger', Float32, queue_size=10)
#         self.vision_pro_status_pub = rospy.Publisher('/vision_pro_status', Bool, queue_size=10)
#         self.image_pub = rospy.Publisher('/robot0_agentview_image', Image, queue_size=10)
#         self.image_pub_128 = rospy.Publisher('/robot0_agentview_image_128', Image, queue_size=10)
#         self.eyeinhand_pub = rospy.Publisher('/robot0_eye_in_hand_image', Image, queue_size=10)
#         self.eyeinhand_pub_128 = rospy.Publisher('/robot0_eye_in_hand_image_128', Image, queue_size=10)

#         self.bridge = CvBridge()
#         self.active = False
#         self.rs_img = None

#         try:
#             self.rs_pipeline = rs.pipeline()
#             self.rs_config = rs.config()
#             self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#             self.rs_pipeline.start(self.rs_config)
#         except Exception as e:
#             rospy.logerr(f"Failed to initialize RealSense: {e}")
#             self.rs_pipeline = None

#         if self.rs_pipeline:
#             threading.Thread(target=self.rs_loop, daemon=True).start()
#             rospy.loginfo("RealSense thread started")

#     # === vision pro active status === #
#     def set_active(self, is_active):
#         self.active = is_active
#         print(self.active)
#         self.vision_pro_status_pub.publish(self.active)

#     # === publish user's wrist and camera  === #
#     def publish_displacement(self, displacement):
#         if not self.active:
#             return
#         msg = Float32MultiArray()
#         msg.data = displacement
#         self.publisher.publish(msg)

#     def publish_finger(self,distance):
#         if not self.active:
#             return
#         msg = Float32()
#         msg.data = distance
#         self.publisher2.publish(msg)

#     def publish_agentview_image(self, cv_image):
#         # if not self.active:                       #cancle here
#         #     return
#         resized = cv2.resize(cv_image, (640, 480))  # 640x480
#         msg = self.bridge.cv2_to_imgmsg(resized, encoding="rgb8")
#         self.image_pub.publish(msg)

#     def publish_agentview_image_128(self, cv_image):
#         if not self.active:                       #cancle here
#             return
#         resized_ = cv2.resize(cv_image, (128, 128))  # 640x480
#         msg_ = self.bridge.cv2_to_imgmsg(resized_, encoding="rgb8")
#         self.image_pub_128.publish(msg_)

#     def rs_loop(self):
#         while not rospy.is_shutdown():
#             # if not self.active:
#             #     return
#             if not self.rs_pipeline:
#                 break
                
#             try:
#                 frames = self.rs_pipeline.poll_for_frames()
#                 if frames and frames.get_color_frame():
#                     color_frame = frames.get_color_frame()
#                     rs_img = np.asanyarray(color_frame.get_data())
#                     self.rs_img = rs_img
                    
#                     # 原圖
#                     msg = self.bridge.cv2_to_imgmsg(rs_img, encoding='bgr8')
#                     self.eyeinhand_pub.publish(msg)

#                     # 壓縮後圖
#                     resized_128 = cv2.resize(rs_img, (128, 128))
#                     msg_128 = self.bridge.cv2_to_imgmsg(resized_128, encoding='bgr8')
#                     self.eyeinhand_pub_128.publish(msg_128)
                    
#             except Exception as e:
#                 rospy.logwarn(f"RealSense error: {e}")
                
#             time.sleep(0.001)

#     def __del__(self):
#         if hasattr(self, 'rs_pipeline') and self.rs_pipeline:
#             self.rs_pipeline.stop()
#             rospy.loginfo("Stopped RealSense pipeline")
    
# class USBCameraSystem:
#     def __init__(self, camera_id=0, resolution=(720, 1280)):
#         self.resolution = resolution
#         self.cam = cv2.VideoCapture(camera_id)

#         if not self.cam.isOpened():
#             raise RuntimeError(f"Failed to open camera {camera_id}")

#         self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[1])
#         self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[0])

#     def get_frames(self, zoom_factor= 1.0):
#         ret, frame = self.cam.read()
#         if not ret:
#             print("Failed to capture frame from camera")
#             return None, None

#         if frame.shape[:2] != self.resolution:
#             frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))

#         # === Digital zoom === #
#         # if zoom_factor > 1.0:
#         #     h, w = frame.shape[:2]
#         #     new_w = int(w / zoom_factor)
#         #     new_h = int(h / zoom_factor)
#         #     start_x = (w - new_w) // 2
#         #     start_y = (h - new_h) // 2
#         #     cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]
#         #     frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
#         if zoom_factor >= 1.0:
#             h, w = frame.shape[:2]
#             new_w = int(w / zoom_factor)
#             new_h = int(h / zoom_factor)
#             start_x = (w - new_w) // 2
#             start_y = max((h // 2) - new_h, 0)  # From middle upward

#             cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]
#             frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         return frame.copy(), frame.copy()

#     def release(self):
#         self.cam.release()

# class VuerTeleop:
#     def __init__(self, config_file_path):
#         self.resolution = (720, 1280)
#         self.crop_size_w = 0
#         self.crop_size_h = 0
#         self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

#         self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
#         self.img_height, self.img_width = self.resolution_cropped[:2]

#         self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
#         self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
#         image_queue = Queue()
#         toggle_streaming = Event()
#         self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
#         self.processor = VuerPreprocessor()

#         RetargetingConfig.set_default_urdf_dir('../../assets')
#         with Path(config_file_path).open('r') as f:
#             cfg = yaml.safe_load(f)
#         left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
#         right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
#         self.left_retargeting = left_retargeting_config.build()
#         self.right_retargeting = right_retargeting_config.build()

#     def step(self):
#         head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
#         head_rmat = head_mat[:3, :3]

#         left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
#                                     rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
#         right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
#                                      rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
#         left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
#         right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

#         thumb_tip = left_hand_mat[tip_indices][0]  # Thumb tip (3D position)
#         index_tip = left_hand_mat[tip_indices][1]  # Index finger tip (3D position)
#         right_thumb_tip = right_hand_mat[tip_indices][0]
#         right_index_tip = right_hand_mat[tip_indices][1] 
#         right_middle_tip = right_hand_mat[tip_indices][2]
#         right_ring_tip = right_hand_mat[tip_indices][3]

#         right_distance = float(np.linalg.norm(right_thumb_tip - right_index_tip))
#         right_middle_distance = float(np.linalg.norm(right_thumb_tip - right_middle_tip))
#         right_ring_distance = float(np.linalg.norm(right_thumb_tip - right_ring_tip))
#         distance = float(np.linalg.norm(thumb_tip - index_tip))
#         return head_rmat, left_pose, right_pose, left_qpos, right_qpos, distance, right_distance, right_middle_distance, right_ring_distance

# class Sim:
#     def __init__(self, print_freq=True):
#         self.publisher = DisplacementPublisher()
#         self.print_freq = print_freq
#         self.target_pose = np.array([-0.4, 0.08, 1.1]) 
#         self.threshold = 0.1
#         self.camera_system = USBCameraSystem()
#         self.active_position = None
#         self.active_orientation = None
#         self.status = "Inactive"
#         self.displacement = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#         self.left_finger = np.zeros(12)
#         self.right_finger = np.zeros(12)
#         self.previous_pose = np.array([-0.45, 0.25,1.6]) #np.array([-0.45, 0.16, 1.34])
#         self.right_distance = 0.0
#         self.distance = 0.0
#         self.phase = "reset"
#         self.episode_info = "Episode: " 


#         #=== Publisher ===#
#         # Save and discard episode by using vision pro 
#         self.save_pub = rospy.Publisher('/save', Float32, queue_size=10)
#         self.discard_pub = rospy.Publisher('/discard', Float32, queue_size=10)

#         rospy.Subscriber("/recording_phase", String, self.phase_callback)
#         rospy.Subscriber("/episode_index", String, self.episode_callback)


#     def phase_callback(self, msg):
#         self.phase = msg.data    

#     def episode_callback(self, msg):
#         self.episode_info = msg.data

#     def step(self, head_rmat, left_pose, right_qpos, left_qpos, right_pose, distance, right_distance, right_middle_distance, right_ring_distance):
#         if self.print_freq:
#             start = time.time()

#         left_position = np.array(left_pose[:3])
#         left_orientation = np.array(left_pose[3:6])
#         right_position = np.array(right_pose[:3])
#         right_orientation = np.array(right_pose[3:6])
#         self.distance = distance
#         self.right_distance = right_distance
#         self.right_middle_distance = right_middle_distance
#         self.ring_distance = right_ring_distance

#         if self.status == "Inactive": 
#             if self.ring_distance <= 0.01 and self.right_distance >= 0.07:
#                 self.status = "Active" 
#                 self.publisher.set_active(True) 
#                 if self.active_position is None:
#                     self.active_position = left_position
#                     self.active_orientation = np.array([-0.45, 0, left_orientation[2]])    
#                 self.previous_pose = left_position  
                
#         else:
#             if self.previous_pose is not None:
#                 displacement_posi = left_position - self.active_position 
#                 left_orientation = np.array([left_orientation[1], left_orientation[0], left_orientation[2]])
#                 displacement_orien = left_orientation - self.active_orientation
#                 self.displacement = np.concatenate(( displacement_posi, displacement_orien))
#                 self.left_finger = np.array(left_qpos)
#                 self.right_finger = np.array(right_qpos)# right hand
#                 if self.right_distance <= 0.01:
#                     self.status = "Inactive"
#                     self.publisher.set_active(False)
                    
#                     self.save_pub.publish(Float32(1.0))
#                     rospy.loginfo("Published /save = 1.0")

#                 if self.right_middle_distance <= 0.01:
#                     self.status = "Inactive"
#                     self.publisher.set_active(False)

#                     self.discard_pub.publish(Float32(1.0))
#                     rospy.loginfo("Published /discard = 1.0")

#             else:
#                 self.displacement = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#                 self.right_displacement = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # right hand
                
#             self.previous_pose = left_position
#             self.publisher.publish_displacement(self.displacement.tolist())
#             self.publisher.publish_finger(self.distance)

#         left_image, right_image = self.camera_system.get_frames()
#         if left_image is None or right_image is None:
#             left_image = np.zeros((self.camera_system.resolution[0], self.camera_system.resolution[1], 3), dtype=np.uint8)
#             right_image = np.zeros((self.camera_system.resolution[0], self.camera_system.resolution[1], 3), dtype=np.uint8)
#         else:
#             self.publisher.publish_agentview_image(left_image)
#             self.publisher.publish_agentview_image_128(left_image)

#             # displacement_text = [
#             #         f"dx={self.displacement[0]:.2f}",
#             #         f"dy={self.displacement[1]:.2f}",
#             #         f"dz={self.displacement[2]:.2f}",
#             #         f"roll={self.displacement[3]:.2f}",
#             #         f"pitch={self.displacement[4]:.2f}",
#             #         f"yaw={self.displacement[5]:.2f}"
#             #     ]   
#             # for i, line in enumerate(displacement_text):
#             #     cv2.putText(right_image, line, (600, 300 + i * 50), 
#             #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
#             color = (0, 255, 0) if self.status == "Active" else (255, 0, 0)
#             cv2.putText(left_image, f"Status: {self.status}", (400, 350),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#             phase_color = (0, 255, 0) if self.phase == "start recording" else (255, 0, 0)
#             cv2.putText(left_image, self.phase, (400, 250),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, phase_color, 2)
            
#             episode_color = (0, 255, 255)  
#             cv2.putText(left_image, self.episode_info, (600, 450),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, episode_color, 2)

#             # left_image = cv2.cvtColor(cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
#             # right_image = cv2.cvtColor(cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
#             left_image = self.publisher.rs_img
#             right_image = self.publisher.rs_img

#         if self.print_freq:
#             end = time.time()
#         return left_image, right_image

#     def end(self):
#         self.camera_system.release()
#         rospy.signal_shutdown("shutdown requested")

# def signal_handler(sig, frame):
#     print("You pressed Ctrl+C! Exiting gracefully...")
#     simulator.end()     
#     exit(0)

# if __name__ == '__main__':
#     signal.signal(signal.SIGINT, signal_handler) 
#     try:
#         rospy.init_node('teleop_sim_node') 
#         teleoperator = VuerTeleop('../inspire_hand.yml')
#         simulator = Sim()

#         test_frame = simulator.camera_system.get_frames()[0]
#         if test_frame is None:
#             raise RuntimeError("Camera initialization failed")

#         while not rospy.is_shutdown():
#             head_rmat, left_pose, right_pose, left_qpos, right_qpos, distance, right_distance, right_middle_distance, right_ring_distance = teleoperator.step()
#             left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos, distance, right_distance, right_middle_distance, right_ring_distance)
#             result = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos, distance, right_distance, right_middle_distance, right_ring_distance)

#             if left_img is not None and right_img is not None:
#                 np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))

#     except KeyboardInterrupt:
#         simulator.end()
#         exit(0)
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         simulator.end()
#         exit(1)




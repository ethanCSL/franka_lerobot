# #!/usr/bin/env python3
# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import rospy
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image

# def main():
#     rospy.init_node('realsense_camera_publisher', anonymous=True)
#     rgb_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
#     depth_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=10)

#     bridge = CvBridge()

#     # Configure RealSense pipeline
#     pipeline = rs.pipeline()
#     config = rs.config()

#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

#     # Start streaming
#     pipeline.start(config)

#     try:
#         while not rospy.is_shutdown():
#             frames = pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()

#             if not color_frame or not depth_frame:
#                 continue

#             # Convert images to numpy arrays
#             color_image = np.asanyarray(color_frame.get_data())
#             depth_image = np.asanyarray(depth_frame.get_data())

#             # Convert to ROS messages
#             color_msg = bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
#             depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')

#             # Publish
#             rgb_pub.publish(color_msg)
#             depth_pub.publish(depth_msg)

#     except rospy.ROSInterruptException:
#         pass
#     finally:
#         pipeline.stop()

# if __name__ == '__main__':
#     main()
import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    # Create align object to align depth to color stream
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Combine images horizontally for display
            combined = np.hstack((color_image, depth_colormap))

            cv2.imshow('RGB + Aligned Depth', combined)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

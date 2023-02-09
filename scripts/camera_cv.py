#!/usr/bin/env python3
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage





def csi_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=10,
    flip_method=0
):
    return ('nvarguscamerasrc sensor-id=%d ! '
                'video/x-raw(memory:NVMM), '
                'width=(int)%d, height=(int)%d, '
                'format=(string)NV12, framerate=(fraction)%d/1 ! '
                'nvvidconv flip-method=%d ! '
                'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=(string)BGR ! appsink' % (sensor_id,
                                                               capture_width, capture_height, framerate, flip_method,
                                                               display_width, display_height))




def camThread():
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(csi_pipeline(), cv2.CAP_GSTREAMER)
    image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=5, tcp_nodelay=False)
    rospy.init_node('cam_stream', anonymous=False)
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
        #if video_capture.isOpened():
        ret, color_image = cap.read()
        if not ret:
            print("no image")
            continue
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', color_image)[1]).tostring()
        # Publish new image
        image_pub.publish(msg)
        
        #rate.sleep()

    cap.release() 


if __name__ == '__main__':


    try:
        camThread()

    except rospy.ROSInterruptException:
        cam.release() 
        pass


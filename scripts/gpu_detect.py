#!/usr/bin/env python3
#!/usr/bin/env python3
# Python libs
import sys, time

# numpy and scipy
import numpy as np

# OpenCV
import cv2
import io

# Ros libraries
import roslib
import rospy

import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda  

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

# Ros Messages
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from kidbright_tpu.msg import tpu_object
from kidbright_tpu.msg import tpu_objects

# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False

class image_feature:

    def __init__(self, path, model):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        #self.font_path = "/home/pi/python/cascadia_font/CascadiaCode-Regular-VTT.ttf"
        #self.font = ImageFont.truetype(self.font_path, 15)
        rospy.init_node('image_feature', anonymous=False)
        self.path = path
        self.model = model
        self.fps = 0.0
        self.cls_dict = {}
        ii = 0
        with open(self.path + '/coco.names') as file:
            lines = [line.strip() for line in file]
        for line in lines:
            self.cls_dict[ii] = line
            ii = ii + 1
        #print(lines)
        self.category_num = len(self.cls_dict)

        self.trt_yolo = TrtYOLO(self.path, self.model, self.category_num, False, cuda_ctx=pycuda.autoinit.context)
        #cls_dict = get_cls_dict(self.category_num)
        #print(cls_dict)
        self.vis = BBoxVisualization(self.cls_dict)
        
        
        self.image_pub = rospy.Publisher("/output/image_detected/compressed", CompressedImage, queue_size = 5, tcp_nodelay=False)
        # self.bridge = CvBridge()
        self.tpu_objects_pub = rospy.Publisher("/tpu_objects", tpu_objects, queue_size = 5, tcp_nodelay=False)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/output/image_raw/compressed", CompressedImage, self.callback,  queue_size = 5, tcp_nodelay=False)

        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.vel_msg = Twist()
        
        #cuda_ctx = cuda.Device(0).make_context()


    def ReadLabelFile(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    def getObjectFeatures(self, box):
        width = box[2]-box[0]
        height = box[3]-box[1]
        area = width*height
        c_x = box[0] + width/2
        c_y = box[3] + height/2
    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''

        #### direct conversion to CV2 ####
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        if image_np is None:
            print("No image")
        t1 = time.time()
        conf_th = 0.5
        boxes, confs, clss = self.trt_yolo.detect(image_np, conf_th)
        image_np = self.vis.draw_bboxes(image_np, boxes, confs, clss)
        image_np = show_fps(image_np, self.fps)
        prepimg = image_np[:, :, ::-1].copy()
        print(confs)
        tpu_objects_msg = tpu_objects()
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]     
            width = x_max - x_min
            height = y_max - y_min
            area = width*height
            c_x = x_min + width/2
            c_y = y_min + height/2       
            cls_name = self.cls_dict.get(cl, 'CLS{}'.format(cl))
            print(cls_name)
            tpu_object_m = tpu_object()
            tpu_object_m.cx = c_x
            tpu_object_m.cy = c_y
            tpu_object_m.width = width
            tpu_object_m.height = height
            tpu_object_m.label = cls_name
            tpu_objects_msg.tpu_objects.append(tpu_object_m)


        t2 = time.time()
        self.fps = 1/(t2-t1)
     
        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        #prepimg.save(fileIO,'jpeg')
        #msg.data = np.array(fileIO.getvalue()).tostring()
        open_cv_image = np.array(prepimg) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        msg.data = np.array(cv2.imencode('.jpg', open_cv_image)[1]).tostring()
        #msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)
        self.tpu_objects_pub.publish(tpu_objects_msg)
        #print(clss)

        
        #self.subscriber.unregister()

def main(path, model):
    '''Initializes and cleanup ros node'''
    ic = image_feature(path, model)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        #cuda_ctx.pop() 
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])


#!/usr/bin/env python3
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""

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

#from edgetpu.detection.engine import DetectionEngine
#from edgetpu.classification.engine import ClassificationEngine
#import classification.engine
#from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Ros Messages
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from kidbright_tpu.msg import tpu_object
from kidbright_tpu.msg import tpu_objects

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False

class image_feature:

    def __init__(self, path):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.font_path = "/home/pi/python/cascadia_font/CascadiaCode-Regular-VTT.ttf"
        self.font = ImageFont.truetype(self.font_path, 15)
        
        self.labels = read_label_file(path + '/label_map.txt') 

        self.interpreter = make_interpreter(path + '/retrained_model_edgetpu.tflite')
        self.interpreter.allocate_tensors()
        

        #self.engine = ClassificationEngine(path + '/retrained_model_edgetpu.tflite') 
        #self.labels = dataset_utils.read_label_file(path + '/label_map.txt')
     
        self.image_pub = rospy.Publisher("/output/image_detected/compressed", CompressedImage, queue_size = 5, tcp_nodelay=False)
        # self.bridge = CvBridge()
        self.tpu_objects_pub = rospy.Publisher("/tpu_objects", tpu_objects, queue_size = 5, tcp_nodelay=False)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/output/image_raw/compressed", CompressedImage, self.callback,  queue_size = 5, tcp_nodelay=False)
        self.size = 320, 240
        
        rospy.init_node('image_class', anonymous=False)
  


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        #print ('get image')
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

        prepimg = image_np[:, :, ::-1].copy()
        prepimg = Image.fromarray(prepimg)
        draw = ImageDraw.Draw(prepimg)
        t1 = time.time()
        tpu_objects_msg = tpu_objects()
        #out = self.engine.classify_with_image(prepimg, top_k=3)
        size = common.input_size(self.interpreter)
        
        image = prepimg.convert('RGB').resize(size, Image.ANTIALIAS)
        common.set_input(self.interpreter, image)
        self.interpreter.invoke()
        
        out = classify.get_classes(self.interpreter, top_k=1, score_threshold=0.90)
        #print(out)
      
     
        
        #print(out)
        ii = 1
        if out:
            
            for obj in out:
                #print ('-----------------------------------------')
                if self.labels:
                    vbal = f"{self.labels[obj[0]]} {obj[1]:0.2f}"
      
                    
                    
                    draw.text((10, 20*ii), vbal, font=self.font, fill='green')
                    
                        
                    tpu_object_m = tpu_object()
                    tpu_object_m.cx = obj[1]
                    tpu_object_m.cy = obj[1]
                    tpu_object_m.width = 0
                    tpu_object_m.height = 0
                    tpu_object_m.label = self.labels[obj[0]]
                    tpu_objects_msg.tpu_objects.append(tpu_object_m)

                

  
                    #draw.text((box[0] + (box[2]-box[0]), box[1]), self.labels[obj.label_id] , fill='green')
                ii = ii + 1
        t2 = time.time()
        fps = 1/(t2-t1)
        fps_str = 'FPS = %.2f' % fps
        draw.text((10,220), fps_str , fill='green')

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        #prepimg.save(fileIO,'jpeg')
        #msg.data = np.array(fileIO.getvalue()).tostring()
        #prepimg = prepimg.resize(self.size, Image.ANTIALIAS)
        open_cv_image = np.array(prepimg) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        msg.data = np.array(cv2.imencode('.jpg', open_cv_image)[1]).tostring()
        #msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)
        self.tpu_objects_pub.publish(tpu_objects_msg)
        
        #self.subscriber.unregister()


    

if __name__ == '__main__':
    ic = image_feature(sys.argv[1])
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    

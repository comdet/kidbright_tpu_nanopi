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
import os

# Ros libraries
import roslib
import rospy



import onnxruntime as onnxrt
import json

# Ros Messages
from sensor_msgs.msg import CompressedImage
from kidbright_tpu.msg import tpu_object
from kidbright_tpu.msg import tpu_objects

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False

class image_feature:

    def __init__(self, path):
        '''Initialize ros publisher, ros subscriber'''

        out_name = os.path.join(path, "resnet18.onnx")
        out_label = os.path.join(path, "labels.json")

        self.onnx_session= onnxrt.InferenceSession(out_name)
        with open(out_label) as f:
            self.labels = json.load(f)
        
        print(self.onnx_session.get_inputs()[0])

        self.input_size = 224
     
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
        input_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        prepimg = input_img[:, :, ::-1].copy()


           # read the image
        #input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        input_img = input_img.astype(np.float32)
        input_img = cv2.resize(input_img, (256, 256))
        # define preprocess parameters
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        scale = 1 / 255.0
        std = [0.229, 0.224, 0.225]
        # prepare input blob to fit the model input:
        #    1. subtract mean
        # 2. scale to set pixel values from 0 to 1
        input_blob = cv2.dnn.blobFromImage(
            image=input_img,
            scalefactor=scale,
            size=(self.input_size, self.input_size),  # img target size
            mean=mean,
            swapRB=True,  # BGR -> RGB
            crop=True  # center crop
        )
        # 3. divide by std
        input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        onnx_inputs= {self.onnx_session.get_inputs()[0].name:input_blob}
        onnx_output = self.onnx_session.run(None, onnx_inputs)
        detected_label = self.labels[onnx_output[0].argmax()]
        #print(detected_label)

        
        prepimg = Image.fromarray(prepimg)
        draw = ImageDraw.Draw(prepimg)
        t1 = time.time()
        
        tpu_objects_msg = tpu_objects()
        tpu_object_m = tpu_object()
        tpu_object_m.cx = 0
        tpu_object_m.cy = 0
        tpu_object_m.width = 0
        tpu_object_m.height = 0
        tpu_object_m.label = detected_label
        tpu_objects_msg.tpu_objects.append(tpu_object_m)

        self.tpu_objects_pub.publish(tpu_objects_msg)


        t2 = time.time()
        fps = 1/(t2-t1)
        fps_str = 'FPS = %.2f' % fps
        draw.text((10,220), fps_str + " " + detected_label , fill='green')

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
        
        #self.subscriber.unregister()


    

if __name__ == '__main__':
    ic = image_feature(sys.argv[1])
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    


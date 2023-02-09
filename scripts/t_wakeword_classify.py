#!/usr/bin/env python3
import rospy
import wave
from std_msgs.msg import String
from kidbright_tpu.msg import float2d, float1d
import python_speech_features
import numpy as np
from kidbright_tpu.msg import int1d
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers, models
from tensorflow.python.keras.models import load_model
import time
import pickle
import base64
import json

sampleRate = 44100 # hertz
numFramePerSec = 20
THRESHOLD = 10



output_names = ['dense_1_1/Sigmoid']
input_names = ['input_1_1']





def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def




class inference():
    def __init__(self):

        rospy.init_node('wake_class_wait')

        # Settings
        self.sampleRate = sampleRate
        self.fps = int(self.sampleRate / numFramePerSec)
        self.frame_count = 0
        self.snd_data = []
        self.num_mfcc = 16
        self.len_mfcc = 16
        self.start_index = 0
        self.window_stride = 6
        self.count = 0
        self.print_frame_count = 0
        self.nFrame = rospy.get_param('~nframe', 20)
        #self.path = rospy.get_param('~path', '/home/pi/webapp/kbapp/client/users/chitanda')
        self.model_file = rospy.get_param('~model', "/home/pi/kb_2/models/model.h5")
        self.labels_file = rospy.get_param('~label', '/home/pi/kbclientNew/server/kb_2/label_map.pkl')
        #self.model_file = self.path + '/model.pb'
        #self.labels_file = self.path + '/label.json'
        print(self.model_file)
        
        with open(self.labels_file, 'r') as f:
            self.labels = json.load(f)
     
        
    

        trt_graph = get_frozen_graph(self.model_file)

        # Create session and load graph
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        tf_config.gpu_options.allow_growth = True
        self.tf_sess = tf.Session(config=tf_config)
        tf.import_graph_def(trt_graph, name='')
        print("Session loaded")


        # Get graph input size
        for node in trt_graph.node:
            if 'input_' in node.name:
                size = node.attr['shape'].shape
                image_size = [size.dim[i].size for i in range(1, 4)]
                break
        print("image_size: {}".format(image_size))


        # input and output tensor names.
        self.input_tensor_name = input_names[0] + ":0"
        output_tensor_name = output_names[0] + ":0"

        print("input_tensor_name: {}\noutput_tensor_name: {}".format(self.input_tensor_name, output_tensor_name))

        self.output_tensor = self.tf_sess.graph.get_tensor_by_name(output_tensor_name)
        
      

        #loading sesssion run for the first time
        mfccs = np.random.randn(16,16)
        mfccs = mfccs.transpose()
        print(mfccs.shape)
        np.set_printoptions(suppress=True)

        # Reshape mfccs to have 1 more dimension
        #x = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        x = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        feed_dict = {self.input_tensor_name: x}
        self.tf_sess.run(self.output_tensor, feed_dict)
        print("Session run loaded")

          # Subscribe to audio_int
        self.a1_sub = rospy.Subscriber("/a1", String, self.callback, queue_size=4)
        self.pred_pub = rospy.Publisher('inference', String, queue_size=10)
        rospy.loginfo("Running inference ...")



     

    def callback(self, msg):

        # Extend subscribed message        
        #print("Receive data")    

        #power = np.sum(np.asarray(msg.data)**2)/2000
        xx = np.frombuffer(base64.b64decode(msg.data), dtype=np.int16).astype(np.float32)
        da_o = np.frombuffer(base64.b64decode(msg.data), dtype=np.int16)
        volume_norm = np.linalg.norm(xx/65536.0)*10
        #print(self.frame_cunt)
        #print(np.sum(np.asarray(msg.data)**2)/2000)
        self.print_frame_count += 1
        if(volume_norm > THRESHOLD and self.frame_count == 0):
            print("start frame")
            self.frame_count += 1
            self.snd_data.extend(da_o)
        else:
            if(self.print_frame_count%10 == 0):
                self.print_frame_count = 0
                self.pred_pub.publish('None')
            
            #self.pred_pub.publish('None')
        if self.frame_count > 0:
            self.frame_count += 1
            self.snd_data.extend(da_o)
            #print(np.sum(np.asarray(msg.data)**2)/2000)
            if self.frame_count == self.nFrame:
                self.frame_count = 0
                print(len(self.snd_data))
            

                mfccs = python_speech_features.base.mfcc(np.array(self.snd_data[self.start_index*self.fps:(self.start_index+self.nFrame)*self.fps]), 
                        samplerate=self.sampleRate,
                        winlen=0.256,
                        winstep=0.050,
                        numcep=self.num_mfcc,
                        nfilt=26,
                        nfft=2048,
                        preemph=0.97,
                        ceplifter=22,
                        appendEnergy=False,
                        winfunc=np.hanning)


                        
                               # Transpose MFCC, so that it is a time domain graph
                mfccs = mfccs.transpose()
                print(mfccs.shape)
                np.set_printoptions(suppress=True)

                # Reshape mfccs to have 1 more dimension
                #x = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
                x = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)

                print(x.shape)
              
                
                #prediction = self.model.predict(x)
                feed_dict = {self.input_tensor_name: x}
                preds = self.tf_sess.run(self.output_tensor, feed_dict)

                print(self.frame_count, ':', preds)
                print(np.argmax(preds) + 1)
                detected_label = self.labels[str(np.argmax(preds) + 1)]
                print(detected_label)
                self.pred_pub.publish(detected_label)
  
                self.snd_data.clear()

       
                
                
                

                



if __name__ == '__main__':
    print("hello meme")
    inference()
    try:
        rospy.spin()
    except:
        print("except")
        pass






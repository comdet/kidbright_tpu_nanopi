from numpy import genfromtxt
import numpy as np
import json
import os
import sys
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="True"
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras import layers, models
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.python.framework import graph_io
import tensorflow.contrib.tensorrt as trt


def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='fmodel.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

def create_model(data, model_param):

    # Add layer to the model
    spectrogram = Input(shape=data.shape[1:])
    #spectrogram = Input(32)
    X = spectrogram
    for layer in model_param:

        if layer['type'] == 'CNN2D':
            X = Conv2D(filters = int(layer['filters']), 
                             kernel_size = eval(layer['kernel_size']), 
                             strides = eval(layer['strides']), 
                             padding = layer['padding'], 
                             dilation_rate = eval(layer['dilation_rate']), 
                             activation = layer['activation'], 
                             use_bias = bool(layer['use_bias']))(X)
            
        if layer['type'] == 'MaxPool2D':
            X = MaxPooling2D(pool_size=eval(layer['pool_size']),
                                   strides=eval(layer['strides']),
                                   padding=layer['padding'])(X)
            
        if layer['type'] == 'Flatten':
            X = Flatten()(X)

        if layer['type'] == 'Dropout':
            X = Dropout(float(layer['remove_prob']))(X)
            
        if layer['type'] == 'Dense':
            X = Dense(units=int(layer['units']),
                            activation=layer['activation'],
                            use_bias=bool(layer['use_bias']))(X)
    
    model = Model(inputs=spectrogram, outputs=X)
    print(model.summary())
    
    return model


print("\nName of Python script:", sys.argv[1])

#path = '/home/pi/webapp/kbapp/client/users/chitanda'
path = sys.argv[1]
mfcc_path = path + '/audios/mfcc/text'
class_path = path + '/audios'




config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.compat.v1.Session(config=config))
 
# Opening JSON file
with open(class_path + '/class.json') as json_file:
    data = json.load(json_file)

print(data['annotations'])
labels = []
features = []
for x in data['annotations']:
    print(x['class'])
    f_name = os.path.splitext(x['file'])[0]
    y_data = genfromtxt(mfcc_path + '/' + f_name + '.csv', delimiter=',')
    features.append(y_data)
    labels.append(x['class'])

all_classes = set(labels)

word2index = {}
count = 1
for x in all_classes:
    word2index[x] = count
    count = count + 1

classes = []
for i, x in enumerate(labels):
    classes.append(word2index[x])

print(word2index)
print(classes)
num_classes = len(word2index)
#print(features)

arr = np.array(features)

print(arr.shape)

a = np.array(classes)
y_tf = tf.keras.utils.to_categorical(a-1, num_classes = len(word2index))

print(y_tf)

labels_res = dict((v,k) for k,v in word2index.items())

with open(path + '/label.json', 'w') as fp:
    json.dump(labels_res, fp)

#my_data = genfromtxt('my_file.csv', delimiter=',')


train_data, validation_data, train_classes, validation_classes = train_test_split(features, y_tf, test_size=0.10, random_state=42, shuffle=True)

keras.backend.clear_session() # clear previous model (if cell is executed more than once)


model_array = [{"type": "CNN2D",
                "filters": "32",
                                        "kernel_size": "(2,2)",
                                        "strides": "(1,1)",
                                        "padding": "same",
                                        "dilation_rate": "(1,1)",
                                        "activation": "relu",
                                        "use_bias": "True"},
                                        
                                        {"type": "MaxPool2D",
                                        "pool_size": "(2,2)",
                                        "strides": "None",
                                        "padding": "same"},
                                        
                                        {"type": "CNN2D",
                                        "filters": "32",
                                        "kernel_size": "(2,2)",
                                        "strides": "(1,1)",
                                        "padding": "same",
                                        "dilation_rate": "(1,1)",
                                        "activation": "relu",
                                        "use_bias": "True"},
                                        {"type": "MaxPool2D",
                                        "pool_size": "(2,2)",
                                        "strides": "None",
                                        "padding": "same"},
                                        
                                        {"type": "CNN2D",
                                        "filters": "64",
                                        "kernel_size": "(2,2)",
                                        "strides": "(1,1)",
                                        "padding": "same",
                                        "dilation_rate": "(1,1)",
                                        "activation": "relu",
                                        "use_bias": "True"},
                                        {"type": "MaxPool2D",
                                        "pool_size": "(2,2)",
                                        "strides": "None",
                                        "padding": "same"},
                                        {"type": "Flatten"},
                                        {"type": "Dense",
                                        "units": "64",
                                        "activation": "relu",
                                        "use_bias": "True"},
                                        
                                        {"type": "Dropout",
                                        "remove_prob": "0.5"}
                                        
                                        ]


last_layer = {"type": "Dense", "units": "1", "activation": "sigmoid", "use_bias": "True"}

last_layer["units"] = str(len(word2index))
model_array.append(last_layer)


train_data = np.array(train_data)
validation_data = np.array(validation_data)
train_classes = np.array(train_classes)
validation_classes = np.array(validation_classes)


train_data = train_data.reshape(train_data.shape[0], 
                          train_data.shape[1], 
                          train_data.shape[2], 
                          1)
validation_data = validation_data.reshape(validation_data.shape[0], 
                      validation_data.shape[1], 
                      validation_data.shape[2], 
                      1)
print("Shape")
print(train_data.shape)
print(validation_data.shape)
#print(x_test.shape)

model = create_model(data=train_data, model_param=model_array)


# sample_shape = train_data.shape[1:]

# model = models.Sequential()
# model.add(layers.Conv2D(32, 
#                         (2, 2), 
#                         activation='relu',
#                         input_shape=sample_shape))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# model.add(layers.Conv2D(32, (2, 2), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# model.add(layers.Conv2D(64, (2, 2), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# # Classifier
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))


print(model.summary())




sgd = keras.optimizers.SGD()
#loss_fn = keras.losses.SparseCategoricalCrossentropy() # use Sparse because classes are represented as integers not as one-hot encoding
loss_fn = keras.losses.BinaryCrossentropy()

model.compile(optimizer=sgd, loss=loss_fn, metrics=["accuracy"])

# Add training parameters to model
#model.compile(loss='binary_crossentropy',  optimizer='rmsprop', metrics=['acc'])

#early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
print(train_classes)
history = model.fit(train_data, 
                    train_classes, 
                    batch_size=16, 
                    epochs=100, 
                    validation_data=(validation_data, validation_classes))

models.save_model(model, path + '/model.h5' )


model_h5 = load_model(path + '/model.h5')

session = tf.keras.backend.get_session()

input_names = [t.op.name for t in model_h5.inputs]
output_names = [t.op.name for t in model_h5.outputs]

# Prints input and output nodes names, take notes of them.
print(input_names, output_names)

frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model_h5.outputs], save_pb_dir=path)



trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

graph_io.write_graph(trt_graph, path, "model.pb", as_text=False)

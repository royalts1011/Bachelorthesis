
# %%
import tensorflow as tf
from tf.keras import backend as K, models
from tf.keras.models import *
from tf.keras.layers import *
from tf.keras.layers.normalization import BatchNormalization
from tf.keras.applications import MobileNetV2
from tf.keras.regularizers import l2
from tf.keras.activations import relu
from tf.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from os.path import join as join_
import numpy as np
from PIL import Image
import cv2


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Num GPUs Available: ", (tf.config.experimental.list_physical_devices('GPU')))


# %%
# Setting up the dataset

SET_DIR = '../AMI/'
NUM_CLASSES = len(os.listdir('../AMI/'))

# The shape which MobilenetV2 accepts as input and thus each image is resized to
image_shape = (224, 224, 3)

# NUM_EXAMPLES is the number of (A,P,N) triplets chosen for the same class (N belongs to a different class of course)
NUM_EXAMPLES = 15

# Triplets list will contain anchor(A), positive(P) and negative(N) triplets.
triplets = []
A = P = N = []

# creating anchor, positive, negative triplets
for _ in range(NUM_EXAMPLES):
    for direc in os.listdir(SET_DIR):
        dir_path = SET_DIR + direc
        dir_contents = os.listdir(dir_path)
        length = len(dir_contents)
        
        anchor = np.asarray(Image.open(join_(dir_path, dir_contents[np.random.randint(0, length)])))/255
        anchor = cv2.resize(anchor, (180, 200)) 
        anchor = np.array([np.pad(a, ((22,22), (12,12)), 'constant') for a in anchor.T]).T
        
        
        positive = np.asarray(Image.open(join_(dir_path, dir_contents[np.random.randint(0, length)])))/255
        positive = cv2.resize(positive, (180, 200)) 
        positive = np.array([np.pad(a, ((22,22), (12,12)), 'constant') for a in positive.T]).T
        
        neg_dir = os.listdir(SET_DIR)[np.random.randint(NUM_CLASSES)]
        while neg_dir == direc: 
            neg_dir = os.listdir(SET_DIR)[np.random.randint(NUM_CLASSES)]
            
        length_negative = len(os.listdir(SET_DIR + neg_dir))
        negative = np.asarray(Image.open(
                       join_(SET_DIR + neg_dir, 
                        os.listdir(SET_DIR + neg_dir)[np.random.randint(0, length_negative)])))/255
        negative = cv2.resize(negative, (180, 200)) 
        negative = np.array([np.pad(a, ((22,22), (12,12)), 'constant') for a in negative.T]).T
        
        # append triplet
        triplets.append([anchor, positive, negative])
        A.append(anchor)
        P.append(positive)
        N.append(negative)


# %%
def triplet_function(vects, alpha=0.4):
    x, y, z = vects
    sum_square_xy = K.sum(K.square(x - y), axis=1, keepdims=True)
    sum_square_xz = K.sum(K.square(x - z), axis=1, keepdims=True)
    return K.sum(K.maximum(sum_square_xy - sum_square_xz + alpha, 0), axis=0)


# %%
def VGG():
    image_input = Input(shape=(224, 224, 3))
    model = MobileNetV2(input_tensor=image_input, weights='imagenet',include_top=True)
    model.layers[-1].activation = relu
    x_out = Dense(32)(model.layers[-1].output)
    
    new_model = Model(inputs=image_input, outputs=x_out)
    return new_model


# %%
def get_model():
    anchor = Input(shape=image_shape, name='anchor')
    positive = Input(shape=image_shape, name='positive')
    negative = Input(shape=image_shape, name='negative')
    
    # Passing each image through the VGG model
    anchor_encoding = VGG()(anchor)
    positive_encoding = VGG()(positive)
    negative_encoding = VGG()(negative)

    # Incorporating the triplet loss in the SimVecLayer
    SimVecLayer = Lambda(triplet_function, output_shape=(1,))
    
    sim_APN = SimVecLayer([anchor_encoding, positive_encoding, negative_encoding])
    
    return Model(inputs=[anchor, positive, negative], outputs=sim_APN)


# %%
model = get_model()

# Compile the model with a loss and optimizer
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae']) 
model.summary()


# %%
from IPython.display import SVG
from tf.keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))


# %%
# Train the model (done over the intel cloud) 
model.fit(x = [A, P, N], y = np.zeros((len(A),1)),
                  epochs=10, verbose=1,
                  batch_size=32, validation_split=0.3,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=2)])


# %%
model.save('model.h5')


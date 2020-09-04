import numpy as np
# import tensorflow as tf
import tensorboard as tb
import torch
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import glob


vectors = []
metadata = []
#label_img = []
class Config():
    DATABASE_FOLDER = './embeddings/radius_2.0/'
    DATASET_DIR = '../dataset/'



for label in os.listdir(Config.DATABASE_FOLDER):
    loaded_embedding = np.load(Config.DATABASE_FOLDER+label, allow_pickle=True)
    
    for idx, e in enumerate(loaded_embedding):
        vectors.append(e.detach().numpy()[0])
        metadata.append(label[:-4])
    
# for label in os.listdir(Config.DATASET_DIR):
#     test = glob.glob(Config.DATASET_DIR+label+'/*')
#     test.sort()
#     for filename in test: 
#         img = Image.open(filename)
#         label_img.append(img)


     
data = np.asarray(vectors)
#img_data = np.array(label_img)

writer = SummaryWriter()
writer.add_embedding(mat=data, metadata=metadata)
writer.close()
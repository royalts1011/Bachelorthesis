# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('../..')
import numpy as np
import matplotlib.pyplot as plt

# PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.models.mobilenet import mobilenet_v2

# DLBio and own scripts
from DLBio.pytorch_helpers import get_device
import ds_ear_siamese
import transforms_data as td
from helpers import cuda_conv
import metrics as M
from siamese_network_train import Training
from ContrastiveLossFunction import ContrastiveLoss
from NN_Siamese import SiameseNetwork


# %%
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(epochs, value1, value2, label1, label2, plt_number):
    plt.figure(plt_number)
    plt.plot(epochs,value1, label=label1)
    plt.plot(epochs,value2, label=label2)
    plt.legend()
    plt.grid()


# %%
# Set Up All Configurations here
class Config():
    #1. Boolean ändern
    #2. FC-Layer auf Bild anpassen
    #3. LR auf 0,0005
    NN_SIAMESE = False
    dataset_dir = '../dataset/'
    # training_dir = "../data/ears/training/"
    # testing_dir = "../data/ears/testing/"
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 16
    vis_batch_size = 8
    num_workers = 3
    
    EPOCHS= 30
    LEARNINGRATE = 0.001
    WEIGHT_DECAY = 0

    TRESHOLD_VER = 0.8
    a = 0


# %%
# define indicies to split Data
dset = ds_ear_siamese.get_dataset(data_path=Config.dataset_dir, transform_mode='size_only')
N = len(dset)
print(N)

# List of index where classes switch
class_switch = [0]
for c in range(len(dset.classes)):
    for i,(_, class_idx) in enumerate(dset.imgs):
        if class_idx > c:
            class_switch.append(i)
            break
# append last index
class_switch.append(len(dset.imgs)-1)

train_indices, val_indices, test_indices = [],[],[]

for i in range(len(class_switch)-1):
    rand_class = np.random.permutation(list(range(class_switch[i], class_switch[i+1])))
    n_80 = int(round(.8*len(rand_class)))
    n_70 = int(round(.7*len(rand_class)))
    n_60 = int(round(.6*len(rand_class)))
    n_20 = int(round(.2*len(rand_class)))
    n_10 = int(round(.1*len(rand_class)))
    train_indices.extend(rand_class[:n_70])
    val_indices.extend(rand_class[n_70:n_70+n_20])
    test_indices.extend(rand_class[n_70+n_20:])

# rand_indices = np.random.permutation(N)
# train_indices = rand_indices[:n_70]
# val_indices = rand_indices[n_70:n_70+n_20]
# test_indices = rand_indices[n_70+n_20:]


# definde data loader
# dl_train = ds_ear_siamese.get_dataloader(
train_dataloader = ds_ear_siamese.get_dataloader(
    data_path=Config.dataset_dir,
    indices=train_indices,
    batch_size=Config.train_batch_size,
    num_workers=Config.num_workers,
    transform_mode='siamese', # TODO switch to another transform?
    should_invert = False
)

val_dataloader = ds_ear_siamese.get_dataloader(
    data_path=Config.dataset_dir,
    indices=val_indices,
    batch_size=Config.val_batch_size,
    num_workers=Config.num_workers,
    transform_mode='siamese_valid_and_test',
    should_invert = False
)
# dl_test = ds_ear_siamese.get_dataloader(
test_dataloader = ds_ear_siamese.get_dataloader(
    data_path=Config.dataset_dir,
    indices=test_indices,
    batch_size=Config.test_batch_size,
    num_workers=Config.num_workers,
    transform_mode='siamese_valid_and_test',
    should_invert = False
)

vis_dataloader = ds_ear_siamese.get_dataloader(
    data_path=Config.dataset_dir,
    indices=train_indices,
    batch_size=Config.vis_batch_size,
    num_workers=Config.num_workers,
    transform_mode='siamese',
    should_invert = False
)


# %%
# visualize some data....
dataiter = iter(vis_dataloader)

example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0], example_batch[1]),0)
imshow(make_grid(concatenated))
print(example_batch[2].numpy())


# %%
# Definde Model and load to device
if Config.NN_SIAMESE == False:
    model = mobilenet_v2(pretrained=True)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
    
    layers = []
    
    for layer in model.features[0]:
        layers.append(layer)
    model.features[0][0] = nn.ReflectionPad2d(1)
    model.features[0][1] = layers[0]
    model.features[0][2] = layers[1]
    model.features[0].add_module('3', layers[2])
    
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])

    #model = torch.nn.Sequential(*(list(model.children()[:-1]))
    #model = model.features

else:
    model = SiameseNetwork()


device = get_device()
print(device)
model.to(device)

contrastive_loss_siamese = ContrastiveLoss(2.0)
optimizer_siamese = torch.optim.Adam(model.parameters(),lr = Config.LEARNINGRATE)


# %%
#To Define which Layers we want to train
for param in model.parameters():
    param.requires_grad = False

layers = list(model.children())[0]
sub_layer = list(layers.children())
unfreezed = [15,16,17,18]
for u in unfreezed:
    for param in sub_layer[u].parameters():
        param.requires_grad = True


# %%
# To show trainable parameters
from DLBio.pytorch_helpers import get_num_params

get_num_params(model,True)


# %%
training = Training(model=model, optimizer=optimizer_siamese,train_dataloader=train_dataloader, val_dataloader=val_dataloader, loss_contrastive=contrastive_loss_siamese, nn_Siamese=Config.NN_SIAMESE, THRESHOLD=Config.TRESHOLD_VER)

epochs, loss_history, val_loss_history, acc_history, val_acc_history = training(Config.EPOCHS)
# show_plot(epochs, loss_history, val_loss_history,'train_loss', 'val_loss',1)
# show_plot(epochs, acc_history, val_acc_history,'train_acc', 'val_acc', 2)


# %%
# tn Bilder nicht gleich, Distanz größer als THRESH
# fp Bilder nicht gleich, Distanz kleiner als THRESH
# fn Bilder gleich, Distanz größer als THRESH
# tp Bilder gleich, Distanz kleiner als THRESH

def calc_test_label(thresh=Config.TRESHOLD_VER):
    '''
    This function processes the test dataloader and returns the true labels and the predicted labels (depending on a threshold)
    Arguments
    ---------
    thresh:     Threshold for "same-different" classification
                default is the Config set threshhold
                

    Returns
    ---------
    Two lists of same length as image tuples in test loader with labels 1 or 0
    '''
    ground_truth_label, prediction_label = [], []

    for data in test_dataloader:
        # use training class for data extraction
        label, output1, output2 = training.get_label_outputs(data)
        # extend labels of the ground truth
        ground_truth_label.extend(label.flatten().tolist())
        # Extend the distance-threshold prediction
        prediction_label.extend(M.batch_predictions_bin(output1, output2, thresh))
    
    return ground_truth_label, prediction_label

ground_truth, prediction = calc_test_label(Config.TRESHOLD_VER)
# get confusion matrix
cf = M.cf_matrix(ground_truth, prediction)
print(cf)


# %%
# Set parameters for confusion_matrix plot
labels = ['True Pos','False Neg','False Pos','True Neg']
categories = ['Same', 'Different']

# plot matrix
# M.make_confusion_matrix(cf,
#                         group_names=labels,
#                         categories=categories,
#                         cbar=True
#                         )


# %%
# preparation for ROC
# define lists for the rates
tprs = []
fprs = []
# Set all Thresholds to be tested
threshholds = [x / 10 for x in list(range(1,12))]

for t in threshholds:
    ground_truth, prediction = calc_test_label(t)
    cf = M.cf_matrix(ground_truth, prediction)
    print("Threshold: ", t , "  Matrix: ", cf)
    _,_,_,sensitivity,specificity = M.get_metrics(cf)

    tprs.append(sensitivity)
    fprs.append( (1 - specificity) )

print("\n TPRS:\t", tprs)
print(" FPRS:\t", fprs)
# show_plot(fprs, tprs, tprs, "ROC", "ROC2", 3)


# %%
#model = torch.load('/Users/falcolentzsch/Develope/Bachelorthesis/Bachelorthesis/models/model.pt')


# %%
#torch.save(model,'/nfshome/lentzsch/Documents/Bachelorarbeit/Bachelorthesis/models/model_MN_1.pt')


# %%



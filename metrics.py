import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

def accuracy(img_output1, img_output2, label, THRESHOLD): 
    acc_counter = 0.0
    distances = F.pairwise_distance(img_output1, img_output2)

    for i, dis in enumerate(distances):
        if((dis <= THRESHOLD and label[i] == 0) or (dis > THRESHOLD and label[i] == 1)): acc_counter += 1
    
    acc = 100 * (acc_counter/len(distances))

    return acc

def tn_fp_fn_tp(img_output1, img_output2, label, THRESHOLD):
    a = 1

def conf_matrix_elements(img_output1, img_output2, label, THRESHOLD):
    # tn Bilder nicht gleich, Distanz größer als THRESH
    # fp Bilder nicht gleich, Distanz kleiner als THRESH
    # fn Bilder gleich, Distanz größer als THRESH
    # tp Bilder gleich, Distanz kleiner als THRESH

    distances = F.pairwise_distance(img_output1, img_output2)
    dist_thresh = [0 if d<=THRESHOLD else 1 for d in distances]

    return confusion_matrix(label, dist_thresh).ravel()

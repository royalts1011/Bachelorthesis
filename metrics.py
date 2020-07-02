import torch.nn.functional as F

def accuracy(img_output1, img_output2, label, THRESHOLD): 
    acc_counter = 0.0
    distance = F.pairwise_distance(img_output1, img_output2)

    for i, dis in enumerate(distance):
        if((dis <= THRESHOLD and label[i] == 0) or (dis > THRESHOLD and label[i] == 1)): acc_counter += 1
    
    acc = 100 * (acc_counter/len(distance))

    return acc

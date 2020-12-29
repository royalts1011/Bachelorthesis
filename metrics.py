import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def accuracy(img_output1, img_output2, label, THRESHOLD): 
    acc_counter = 0.0
    distances = F.pairwise_distance(img_output1, img_output2)

    for i, dis in enumerate(distances):
        if((dis <= THRESHOLD and label[i] == 0) or (dis > THRESHOLD and label[i] == 1)): acc_counter += 1
    
    acc = 100 * (acc_counter/len(distances))

    return acc

def batch_predictions_bin(img_output1, img_output2, THRESHOLD):
    '''
    This function  will compute distances and return a binary array. 
    0 corresponds to a value being lower or equal than the threshhold, 1 higher
    
    Arguments
    ---------
    img_output1:    output of the model (usually amount of batchsize in tensor list)
    img_output2:    output of the model (usually amount of batchsize in tensor list)
        together they represent the tuple
    THRESHOLD:      Threshhold for distances between the images

    
    Return
    ---------
    Binary list of predicted labels
    '''
    distances = F.pairwise_distance(img_output1, img_output2)
    return [0 if d<=THRESHOLD else 1 for d in distances]


def cf_matrix(ground_truth, predictions):
    return confusion_matrix(ground_truth, predictions)


def get_metrics(cf):
    '''
    This function calculates different metrics and scores of a confusion matrix
    Arguments
    ---------
    cf:  Confusion Matrix of size 2x2

    Returns
    ---------
    precision, specificity, F1-score, sensitivity, specificity
    '''
    assert len(cf)==2, "The confusion matrix is not of binary origin or has wrong size"
    #Metrics for Binary Confusion Matrices
    precision = cf[0,0] / sum(cf[:,0])
    recall    = cf[0,0] / sum(cf[0,:])
    f1_score  = 2*precision*recall / (precision + recall)
    sensitivity = recall
    specificity = cf[1,1] / sum(cf[1,:])

    return precision, recall, f1_score, sensitivity, specificity


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    ---------
    Resource see:   https://github.com/DTrimarchi10/confusion_matrix
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision, recall, f1_score, _, _ = get_metrics(cf)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                100*accuracy,100*precision,100*recall,100*f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
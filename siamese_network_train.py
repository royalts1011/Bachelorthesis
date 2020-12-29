from torch import cuda
import torch.nn.functional as NNF
from metrics import accuracy
from helpers import cuda_conv
import csv


class Training():
    def __init__(
        self,model, optimizer, train_dataloader, loss_contrastive, nn_Siamese, val_dataloader, THRESHOLD
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_contrastive = loss_contrastive
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.nn_Siamese = nn_Siamese
        self.THRESHOLD = THRESHOLD


    def __call__(self, epochs_):

        loss_history = []
        acc_history = []
        val_loss_history = []
        val_acc_history = []
        epochs = []




        for epoch in range(0,epochs_):
            acc = 0.0
            loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            self.model.train()
            # initiate threshold log file
            with open('log_dist_label.csv', 'a') as f:
                writer = csv.writer(f, delimiter=',', lineterminator='\n')
                for i, data in enumerate(self.train_dataloader):
                    # clear gradients from last step
                    self.optimizer.zero_grad()

                    label, output1, output2 = self.get_label_outputs(batch=data)

                    # compute distance and save in file
                    dist = NNF.pairwise_distance(output1, output2, keepdim = True)
                    dist = [x.cpu().detach().numpy()[0] for x in dist]
                    save_label = [x.cpu().detach().numpy()[0] for x in label]
                    for d,l in zip(dist, save_label):
                        writer.writerow([d, l])

                    loss_contrastive = self.loss_contrastive(output1,output2,label)
                    # backpropagation
                    loss_contrastive.backward()
                    self.optimizer.step()

                    acc += accuracy(output1, output2, label, self.THRESHOLD)
                    loss += loss_contrastive.item()
            f.close()
            
            acc = acc/len(self.train_dataloader)
            loss = loss/len(self.train_dataloader)
            acc_history.append(acc)
            loss_history.append(loss)
            #print("Epoch number {}\n Current loss {:.4f}\n Current acc {:.2f}\n".format(epoch,loss, acc))

            self.model.eval()
            for i, data in enumerate(self.val_dataloader):

                label, output1, output2 = self.get_label_outputs(batch=data)

                loss_contrastive = self.loss_contrastive(output1,output2,label)

                val_acc += accuracy(output1, output2, label, self.THRESHOLD)
                val_loss += loss_contrastive.item()
            
            val_acc = val_acc/len(self.val_dataloader)
            val_loss = val_loss/len(self.val_dataloader)
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            #print("Epoch number {}\n Current val_loss {:.4f}\n Current val_acc {:.2f}\n".format(epoch,val_loss, val_acc))
            print("Epoch number {}\n Current loss {:.4f}\n Current acc {:.2f}\n Current val_loss {:.4f}\n Current val_acc {:.2f}\n".format(epoch, loss, acc, val_loss, val_acc))
            epochs.append(epoch)
        return epochs, loss_history, val_loss_history, acc_history, val_acc_history
    


 
    def get_label_outputs(self, batch):
        """Extract batch images and process through model

        Parameters
        ----------
        batch : current batch

        Returns
        -------
        label, output1, output2
            binary label being same (0) or different (1)
            outputs processed through model
        """
        # get all images and labels
        img0, img1 , label = batch
        # Type conversion
        img0, img1 , label = cuda_conv(img0), cuda_conv(img1), cuda_conv(label)
        # if cuda.is_available():
        #     img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        # else: 
        #     img0, img1 , label = img0, img1 , label

        # Throw in correct network
        if self.nn_Siamese == True:
            output1, output2 = self.model(img0,img1)
        else:                 
            output1 = self.model(img0)
            output2 = self.model(img1)
        
        return label, output1, output2

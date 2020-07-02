from torch import cuda
import torch.nn.functional as NNF
from metrics import accuracy
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
        counter = []
        loss_history = []
        iteration_number = 0
        acc_history = []


        val_counter = []
        val_loss_history = []
        val_iteration_number = 0
        val_acc_history = []


        for epoch in range(0,epochs_):
            for i, data in enumerate(self.train_dataloader):
                img0, img1 , label = data
                if cuda.is_available():
                    img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                else: 
                    img0, img1 , label = img0, img1 , label
                self.optimizer.zero_grad()
                if self.nn_Siamese == False:
                    output1 = self.model(img0)
                    output2 = self.model(img1)
                else:                 
                    output1,output2 = self.model(img0,img1)

                loss_contrastive = self.loss_contrastive(output1,output2,label)
                loss_contrastive.backward()
                self.optimizer.step()
                
                if i %10 == 0:
                    acc = accuracy(output1, output2, label, self.THRESHOLD)
                    print("Epoch number {}\n Current loss {:.4f}\n Current acc {:.2f}\n".format(epoch,loss_contrastive.item(), acc))
                    iteration_number +=10
                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())
                    acc_history.append(acc)

            for i, data in enumerate(self.val_dataloader):
                img0, img1 , label = data
                if cuda.is_available():
                    img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                    output1 = self.model(img0)
                    output2 = self.model(img1)
                else: 
                    img0, img1 , label = img0, img1 , label
                    output1,output2 = self.model(img0,img1)

                loss_contrastive = self.loss_contrastive(output1,output2,label)

                if i %20 == 0 :
                    val_acc = accuracy(output1, output2, label, self.THRESHOLD)
                    print("Epoch number {}\n Current val_loss {:.4f}\n Current val_acc {:.2f}\n".format(epoch,loss_contrastive.item(), val_acc))
                    val_iteration_number +=20
                    val_counter.append(val_iteration_number)
                    val_loss_history.append(loss_contrastive.item())
                    val_acc_history.append(val_acc)


        return counter, loss_history, val_counter, val_loss_history, acc_history, val_acc_history

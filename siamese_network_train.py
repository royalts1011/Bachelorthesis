from torch import cuda
class Training():
    def __init__(
        self,model, optimizer, train_dataloader, loss_contrastive, nn_Siamese, val_dataloader
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_contrastive = loss_contrastive
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.nn_Siamese = nn_Siamese


    def __call__(self, epochs_):
        counter = []
        loss_history = []
        val_counter = []
        val_loss_history = []
        iteration_number= 0
        val_iteration_number= 0
        for epoch in range(0,epochs_):
            for i, data in enumerate(self.train_dataloader,0):
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
                if i %10 == 0 :
                    print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                    iteration_number +=10
                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())

            for i, data in enumerate(self.val_dataloader,0):
                img0, img1 , label = data
                if cuda.is_available():
                    img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                    output1 = self.model(img0)
                    output2 = self.model(img1)
                else: 
                    img0, img1 , label = img0, img1 , label
                    output1,output2 = self.model(img0,img1)

                loss_contrastive = self.loss_contrastive(output1,output2,label)

                if i %10 == 0 :
                    print("Epoch number {}\n Current Validation loss {}\n".format(epoch,loss_contrastive.item()))
                    val_iteration_number +=10
                    val_counter.append(val_iteration_number)
                    val_loss_history.append(loss_contrastive.item())


        return counter, loss_history, val_counter, val_loss_history
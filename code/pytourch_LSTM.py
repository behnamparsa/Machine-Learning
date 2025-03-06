import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import Adam
import lightning as L
from torch.utils.data import TensorDataset, DataLoader

class LSTMbyHand(L.LightningModule):

    def __init__(self):
        # Create and initialze Weight and Bias tensors.
        super().__init__()
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        self.wlr1 = nn.Parameter(torch.normal(mean= mean, std= std), requires_grad= True)
        self.wlr2 = nn.Parameter(torch.normal(mean= mean, std= std), requires_grad= True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean= mean, std= std), requires_grad= True)
        self.wpr2 = nn.Parameter(torch.normal(mean= mean, std= std), requires_grad= True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean= mean, std= std), requires_grad= True)
        self.wp2 = nn.Parameter(torch.normal(mean= mean, std= std), requires_grad= True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean= mean, std= std), requires_grad= True)
        self.wo2 = nn.Parameter(torch.normal(mean= mean, std= std), requires_grad= True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)




    def lstm_unit(self, input_value, long_memory, short_memory):

        # Do the LSTM math.
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + 
                                              (input_value *  self.wlr2) + 
                                                self.blr1)
        
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) +
                                                   (input_value * self.wpr1) + 
                                                   self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                      (input_value * self.wp2) +
                                      self.bp1)
        updated_long_memory = ((long_memory * long_remember_percent) +
                               (potential_remember_percent * potential_memory))
        
        output_percent = torch.sigmoid((short_memory * self.wo1) +
                                       (input_value * self.wo2) +
                                       self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) *output_percent

        return([updated_long_memory, updated_short_memory])
        

    def forward(self, input):
        
        #Make a forward pass through unrolled LSTM
        long_memory = 0
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory = self.lstm_unit(day1, long_memory,short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory,short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory,short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory,short_memory)

        return short_memory


   
    def configure_optimizers(self):
        # configure Adam optimizer.
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        # Calculate loss and log training progress.
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)

        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss


model = LSTMbyHand()

print("\n Now let's compare the observed and predicted values ...")

print("Company A: Observed = 0, Predicted = ",
      model(torch.tensor([0., 0.5, 0.25, 0.])).detach())

print("Company B: Observed = 1, Predicted = ",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

inputs = torch.tensor([[0., 0.5, 0.25, 0.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)

dataloader = DataLoader(dataset)

trainer = L.Trainer(max_epochs = 6000)
trainer.fit(model, train_dataloaders = dataloader)

print("\n Now let's compare the observed and predicted values after training ...")

print("Company A: Observed = 0, Predicted = ",
      model(torch.tensor([0., 0.5, 0.25, 0.])).detach())

print("Company B: Observed = 1, Predicted = ",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

#path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path
#trainer = L.Trainer(max_epochs= 3000)
#trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)

#print("\n Now let's compare the observed and predicted values after second training ...")

#print("Company A: Observed = 0, Predicted = ",
#      model(torch.tensor([0., 0.5, 0.25, 0.])).detach())

#print("Company B: Observed = 1, Predicted = ",
#      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())




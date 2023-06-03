import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor

from src.data.dataset import HotDogDataset

from pathlib import Path
import click
from tqdm import tqdm
import numpy as np

# Set the device as a global variable.
# For torch convention, it is lowercase.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training is running on: {device}')



def train(model, optimizer, loss_fun, num_epochs=10):
   
    '''
    Function that trains a model for a given number of epochs.
    Returns a dictionary with the tracked metrics.
    '''

    out_dict = {'train_acc': [],
                'test_acc': [],
                'train_loss': [],
                'test_loss': []}
  
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave=None):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
    return out_dict

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.option('--batch_size', default=32, help='batch size for training and testing')
def main(input_filepath: str, batch_size: int)-> None:
    
    # Create the datasets and dataloaders
    training_data = HotDogDataset(input_filepath, train=True)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    test_data = HotDogDataset(input_filepath, train=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    #
    


if __name__=='__main__':
    main()
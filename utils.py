# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:17:53 2024
modified from https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/engine.py
@author: lshaw
"""
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import optimisers
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pickle

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimiser: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimiser step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimiser: A PyTorch optimiser to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
   
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimiser.zero_grad()

        # 4. Loss backward
        loss.backward()
        
        def closure():
            optimiser.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            return loss
    
        # 5. Optimizer step
        if type(optimiser)==optimisers.SGHMC:
            optimiser.step(closure)
        else:
            optimiser.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimiser: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          burnin: int) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimiser: A PyTorch optimiser to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)
    
    for _ in tqdm(range(burnin),desc='burnin'):
        _,_=train_step(model=model,
                       dataloader=train_dataloader,
                       loss_fn=loss_fn,
                       optimiser=optimiser,
                       device=device)
    
    #Now start collecting mean
    optimiser.reset() #numits back to 0, wipe history

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs),desc='training'):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimiser=optimiser,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    
    results["history"]=optimiser.history
    optimiser.reset() #clear history
    # Return the filled results at the end of the epochs
    temp=optimiser.defaults
    temp['epochs']=epochs
    temp['burnin']=burnin
    d={'optimiser':str(type(optimiser)),'hyperparams':temp,'results':results}
    return d

class RandomProjection(nn.Module):
    '''
    Implements Alg. 1 from "Random Projections for k-means Clustering", 
    Boutsidis et. al.
    '''
    def __init__(self,projdim,targetdim):
        super().__init__()
        t=projdim*targetdim
        transformation_matrix=torch.randint(low=0,high=2,size=(t,),dtype=float)
        transformation_matrix[transformation_matrix==0]=-1
        transformation_matrix/=torch.sqrt(torch.tensor(t))
        self.transformation_matrix=transformation_matrix.reshape((targetdim,projdim))
    
    def forward(self, tensor):
        shape = tensor.shape
        n = shape.numel() #size of image vector
        if n != self.transformation_matrix.shape[0]:
            raise ValueError("Input tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(shape[-3], shape[-2], shape[-1]) +
                             "{}".format(self.transformation_matrix.shape[1]))
        if tensor.dtype != self.transformation_matrix.dtype:
            self.transformation_matrix=self.transformation_matrix.to(tensor.dtype)

        flat_tensor = tensor.view(-1,n)
        transformed_tensor = torch.mm(flat_tensor,self.transformation_matrix)
        return transformed_tensor
    
def plot_loss_acc(dictionary):
    fig,ax=plt.subplots(1,2,figsize=(2,1))
    results=dictionary['results']
    epochs=dictionary['hyperparams']['epochs']
    a=ax[0]
    a.plot(torch.arange(epochs),results['train_loss'],label='train')
    a.plot(torch.arange(epochs),results['test_loss'],label='test')
    a.title('Loss')
    a.legend()
    a=ax[1]
    a.plot(torch.arange(epochs),results['train_acc'],label='train')
    a.plot(torch.arange(epochs),results['test_acc'],label='test')
    a.title('Accuracy')
    for a in ax:
        a.set_xlabel('epoch')

def plot_mean_err(directory):
    fig=plt.plt(figsize=(2,1))
    for root, dirs, files in os.walk(directory):

    results=dictionary['results']
    epochs=dictionary['hyperparams']['epochs']
    a=ax[0]
    a.plot(torch.arange(epochs),results['train_loss'],label='train')
    a.plot(torch.arange(epochs),results['test_loss'],label='test')
    a.title('Loss')
    a.legend()
    a=ax[1]
    a.plot(torch.arange(epochs),results['train_acc'],label='train')
    a.plot(torch.arange(epochs),results['test_acc'],label='test')
    a.title('Accuracy')
    for a in ax:
        a.set_xlabel('epoch')

    

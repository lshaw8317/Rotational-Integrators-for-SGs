# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:30:57 2024

@author: lshaw
"""
import torchvision
import os
import torch
import torchvision.transforms as transforms
import models, utils
import data_setup as data
import optimisers
import pickle
from emcee.autocorr import integrated_time
import time

#%% MNIST LogReg
# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
directory='MNISTLogReg'
with open(os.path.join(directory,'truemean'),'r') as f:
    truemean=pickle.load(f)

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data.create_experiment(
    'MNIST_binary',batch_size=BATCH_SIZE)

# Create model with help from model_builder.py
model = models.LogisticRegression(
    input_dim=50).to(device)

# Set loss and optimiser
Ndata=train_dataloader.sampler.num_samples
loss_fn = torch.nn.CrossEntropyLoss()
kwargs={}
optimiser = optimisers.SGFS(model.parameters(),Ifmode='diag',**kwargs,
                            Ndata=Ndata,batch_size=train_dataloader.batch_size)

start = time.time()

results=utils.train(model=model,
        burnin=5,
         train_dataloader=train_dataloader,
         test_dataloader=test_dataloader,
         loss_fn=loss_fn,
         optimiser=optimiser,
         epochs=NUM_EPOCHS,
         device=device)
exec_time=time.time()-start
results['execution_time']=exec_time
ifmode=optimiser.defaults['Ifmode']
history=results['history']

temp=[]
for it in len(history):
    temp.append(torch.cat([p.view((-1)) for p in history[it]]))
temp=torch.as_tensor(temp)
ATUC=torch.as_tensor(integrated_time(temp,has_walkers=False)).mean()*exec_time
results['ATUC']=ATUC
filename=os.path.join(directory,str(type(optimiser)).split('.')[-1],ifmode,f'ATUC{ATUC}')
results['meanerr']=(torch.abs(temp.mean(dim=0)-truemean)/torch.abs(truemean)).mean()
with open(filename,'w') as f:
    pickle.dump(results,f)



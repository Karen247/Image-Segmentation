# Description:
# This file should be used for performing training of a network
# Usage: python training.py <path_2_dataset>

##############  IMPORTS  ##############
import zipfile
import sys
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchview import draw_graph
#from network import ModelExample
from pathlib import Path
from dataset import SampleDataset
import re
import pandas as pd
import os
from skimage import color, io
from sklearn.model_selection import train_test_split
from network import UNet
from dataset import extract_seg_dataset, get_dataframe
import dagshub
import mlflow
import mlflow.pytorch
import time
from torch.autograd import Variable
import numpy as np


######################################

torch.manual_seed(42)

img_size = 128
transforms_train = A.Compose([
                         # augmentation 
                         #A.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.1),
                         #A.GaussNoise(),
                         #A.HorizontalFlip(p=0.5),
                         #A.ShiftScaleRotate(rotate_limit=30, p=1, border_mode=0, value=0),
                         #A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                            
                         # preprocessing
                         A.SmallestMaxSize (img_size),
                         A.CenterCrop(img_size, img_size),
                         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                         ToTensorV2(),
                        ]   
                    )
transforms = A.Compose([
                         # preprocessing
                         A.SmallestMaxSize (img_size),
                         A.CenterCrop(img_size, img_size),
                         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                         ToTensorV2(),
                        ]   
                    )

class BestModel_Callback:

    def __init__(self, model_path):
        
        self.best_valid_loss = float('inf')
        self.model_path = model_path
        
    def __call__(self, model, valid_loss):
        
        if valid_loss < self.best_valid_loss:
            
            print(f"Saving model {self.model_path} with validation {valid_loss:.3f}")
            self.best_valid_loss = valid_loss
            torch.save(model, self.model_path)

class EarlyStopping_Callback:

    def __init__(self,
                 patience=5, delta = 1e-5,
                 #best_valid_loss=float('inf')
                 ):
        #self.best_valid_loss = best_valid_loss
        self.best_valid_loss = None
        self.delta = delta
        self.patience = patience
        self.increasing_steps = 0
        
    def __call__(self, valid_loss):

        if self.best_valid_loss is None:
            self.best_valid_loss = valid_loss

        elif abs(valid_loss - self.best_valid_loss ) < self.delta:
            
            self.increasing_steps += 1
            print(f'{self.increasing_steps} without improvement')

        elif self.best_valid_loss > valid_loss:
            self.best_valid_loss = valid_loss
            
        if self.increasing_steps == self.patience:
            return True
        
        return False
    
# sample function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(network, input_sample):
    # saves visualization of model architecture to the model_architecture.png
    model_graph = draw_graph(network, input_sample, graph_dir='LR', save_graph=True, filename="model_architecture")


# sample function for losses visualization
def plot_learning_curves(train_losses, validation_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.01
    factor = 0.01
    return base_lr/(1+factor*epoch)


def custom_accuracy_score(y_true, y_pred):
    '''
    y_true and y_pred are torch tensors
    '''
    y_true = np.array(y_true.cpu().detach())
    y_pred = np.array(y_pred.cpu().detach())
    #print(y_true.shape, y_pred.shape)
    y_pred = np.argmax(np.array(y_pred), axis=1)
    #print(y_true.shape, y_pred.shape)
    return np.sum(y_pred == y_true) / len(y_pred)

    
def loss_batch(model, loss_func, xb, yb, dev, opt=None):
    xb, yb = xb.to(dev), yb.to(dev)
    pred = model(xb).to(dtype = torch.float32)
    loss = loss_func(pred, yb)
    acc = custom_accuracy_score(yb, pred)
    #print(f"Loss: {loss}")
    #print(f"Acc: {acc}")
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), len(xb), acc
    
def train(model, train_dl, loss_func, dev, opt):
    model.train()
    current_loss = 0.0
    loss, size = 0, 0
    total_acc = 0.0
    for b_idx, (xb, yb) in enumerate(train_dl):
        b_loss, b_size, acc = loss_batch(model, loss_func, xb, yb, dev, opt)
        current_loss += b_loss
        total_acc += acc
        if b_idx % 100 == 99:
           loss_minibatch = current_loss / 100
           acc_minibatch = total_acc / 100
           print('Loss after 100 batches %5d: %.3f' % (b_idx + 1, loss_minibatch))
           print('Accuracy after 100 batches %5d: %.3f' % (b_idx + 1, acc_minibatch))
           #mlflow.log_metric("100 batches_loss", loss_minibatch, step=b_idx+1)
           #mlflow.log_metric("100 batches_accuracy", acc_minibatch, step=b_idx+1)
           current_loss = 0.0
           total_acc = 0.0
        loss += b_loss * b_size
        size += b_size
        
    return loss / size
    
def validate(model, valid_dl, loss_func, dev, opt=None):
        model.eval()
        with torch.no_grad():
            losses, nums, accs = zip(
                *[loss_batch(model, loss_func, xb, yb, dev) for xb, yb in valid_dl]
            )
            
        return np.sum(np.multiply(losses, nums)) / np.sum(nums)

# sample function for training
def fit(net, batch_size, epochs, trainloader, validloader, loss_fn, optimizer, device, scheduler = None):
    train_losses = []
    validation_losses = []
    best_model = BestModel_Callback(model_path='best_model.pt')
    early_stopping = EarlyStopping_Callback(patience=5)
    for epoch in range(epochs):
        loss = train(net, trainloader, loss_fn, device, optimizer)
        val_loss = validate(net, validloader, loss_fn, device) 
        if scheduler is not None:
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning rate", before_lr, step=epoch)
            print("Epoch %d: Optimizer lr %.7f -> %.7f" % (epoch, before_lr, after_lr))
        mlflow.log_metric("training loss", loss)
        mlflow.log_metric("validation loss", val_loss)
        print('Epoch {}, train loss: {:.5f}, val loss: {:.5f}'.format(epoch, loss, val_loss))
        train_losses.append(loss)
        validation_losses.append(val_loss)
        best_model(net, val_loss)
        if early_stopping(val_loss):
            print(f'Early Stopping: epoch {epoch}, val_loss {val_loss:.3f}')
            break 

    print('Training finished!')
    return train_losses, validation_losses

# declaration for this function should not be changed
def training(dataset_path):
    """
    training(dataset_path) performs training on the given dataset;
    saves:
    - model.pt (trained model)
    - learning_curves.png (learning curves generated during training)
    - model_architecture.png (a scheme of model's architecture)

    Parameters:
    - dataset_path (string): path to a dataset

    Returns:
    - None
    """
    start_time = time.time()
    print("********************************** STARTING **********************************")
    params = {
        "info":"Image segmentation",
        "learning_rate": 0.0001,
        "batch_size": 4,
        "epochs" : 5,
        "random_state": 22,
        "weight_decay" : 0.1,
        "epsilon":1e-08,
        "Softmax": True,
        "Adam" : True
    }
    print(params)
    mlflow.log_params(params)
    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print('Computing with {}!'.format(device))
    df = extract_seg_dataset(dataset_path)
    
    train_df, valid_df = train_test_split(df, test_size=.1, random_state=params['random_state'])
    print(train_df.shape, valid_df.shape)
    train_ds = SampleDataset(train_df, transforms)
    valid_ds = SampleDataset(valid_df, transforms)
    
    trainloader = torch.utils.data.DataLoader(train_ds,
                      batch_size=params['batch_size'],
                      shuffle=True,
                      num_workers=1)

    valloader = torch.utils.data.DataLoader(valid_ds,
                      batch_size=params['batch_size'],
                      shuffle=True,
                      num_workers=1)
    
    net = UNet(num_class = 8).to(device)
    input_sample = torch.zeros((1, 3, 128, 128))
    draw_network_architecture(net, input_sample)
    # define optimizer and learning rate
    if params["Adam"]:
        optimizer = torch.optim.Adam(net.parameters(), lr=params['learning_rate'], eps = params['epsilon'])
        #optimizer = torch.optim.AdamW(net.parameters(), lr=params['learning_rate'], weight_decay = params['weight_decay'], betas=(0.9, 0.999), eps=1e-08)
        print("Adam optimizer is used")
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=params['learning_rate'], momentum=0.9, nesterov = True)
        print("SGD optimizer is used")
    # optimizer = torch.optim.AdamW(net.parameters(), lr=params['learning_rate'], weight_decay = params['weight_decay'], betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) 
    print("Learning rate init: ", optimizer.param_groups[0]['lr'])
    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # train the network for three epochs
    tr_losses, val_losses = fit(net, params['batch_size'], params['epochs'], trainloader, valloader, loss_fn, optimizer, device, scheduler)
    # save the trained model and plot the losses, feel free to create your own functions
    print("Saving")
    torch.save(net, 'model.pt')
    plot_learning_curves(tr_losses, val_losses)
    print("********************************** ENDING **********************************")
    print("--- %s seconds ---" % (time.time() - start_time))
    return


def get_arguments():
    if len(sys.argv) != 2:
        print("Usage: python training.py <path_2_dataset> ")
        sys.exit(1)

    try:
        path = sys.argv[1]
    except Exception as e:
        print(e)
        sys.exit(1)
    return path


if __name__ == "__main__":
    path_2_dataset = get_arguments()
    training(path_2_dataset)

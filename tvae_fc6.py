
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from models.tvae_fc7 import TVAE_fc7
from models.tvae_fc6 import TVAE_fc6
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as sets
import torchvision.transforms as transforms
import sys
from scipy.stats import pearsonr
import warnings
from data.data import *

if not sys.warnoptions:
    warnings.simplefilter("ignore")


"""
Create the FC6 model and use a pre-trained model to extract features
"""

def create_model(s_dim, kernel_size):
    return TVAE_fc6(9216, s_dim, kernel_size)

Alex = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

class Feature(nn.Module):

    def __init__(self, ):
        super(Feature, self).__init__()
        # Block 0: input to maxpool1
        self.block0 = nn.Sequential(
            Alex.features,
            Alex.avgpool,
        )
        self.fc6 = nn.Sequential(
            nn.BatchNorm1d(9216)
        )


    def forward(self, inp):
        outp = self.block0(inp)
        outp = torch.flatten(outp, 1)
        outp = self.fc6(outp)

        return outp



"""
Training the model
"""
def train_epoch(feature, model, optimizer, train_loader,e):
    total_loss = 0
    total_kl = 0
    total_recon = 0
    num_batches = 0

    for x, label in train_loader:
        x = x.to('cuda')
        ## Extract pre-trained feature
        out_features = feature(x)
        out_features = out_features.detach()
        optimizer.zero_grad()

        x_recon, recon_loss, KLD, s = model(out_features)

        loss = recon_loss + KLD # + qo_loss #+ sparsity_loss * 100
        loss.backward()

        optimizer.step()

        model.normalize_weights()

        total_loss += loss
        total_recon += recon_loss
        total_kl += KLD
        num_batches += 1
        print('[%d] Loss: %.4f, Recon: %.4f, KL : %.4f' % (e+1, loss,   recon_loss,  KLD))

        if num_batches==1:
            s = s.cpu().detach()
            Plot_Covariance_Matrix(s, s,e)

    return total_loss, total_recon, total_kl, num_batches

"""
Evaluate the Selectivity of neurons:
First, save all the preferences of test images for both face or object
Second, calculate the d' according to the equation
Finally, save the d'
"""

def neurons_d(feature, classifier, train):
    face = torch.zeros([1, 4096])
    for inputs, classes in train:
        inputs = inputs.to('cuda')
        inputs = feature(inputs)
        _,_,_,_,_,x = classifier.sample(inputs)

        x = x.detach().cpu()
        face = torch.cat((x, face), 0)
    print('finished')
    return face[:-1, :].numpy()

def calculate(face, obj):
    mean_face = np.mean(face, axis=0)
    mean_object = np.mean(obj, axis=0)
    std_face = np.mean(face, axis=0)
    std_object = np.mean(obj, axis=0)
    d = (mean_face - mean_object) / np.sqrt((std_face ** 2 + std_object ** 2) / 2)
    k = d[d > 0.85]
    z = d[d < -0.85]
    print(k.shape, z.shape)
    return d

def calculate_main(feature, classifier,dirc):
    face_train = loadsampleface(dirc)
    object_train = loadsampleobject()

    face = neurons_d(feature, classifier, face_train)
    obj = neurons_d(feature, classifier, object_train)
    d = calculate(face, obj)
    return d

"""
Evaluate the Selectivity of neurons:
First, save all the preferences of test images for both face or object
Second, calculate the d' according to the equation
Finally, save the d'
"""

def correlation(s):
    s = s.numpy().reshape(-1,64,64)
    possible ={}
    for x1 in range(64):
        for y1 in range(64):
            for x2 in range(64):
                for y2 in range(64):
                    x1mm = x1 / 6.4
                    x2mm = x2 / 6.4
                    y1mm = y1 / 6.4
                    y2mm = y2 / 6.4
                    distance =  np.sqrt((x1mm - x2mm)**2 + (y1mm - y2mm)**2)
                    possible[round(distance, 1) ] =[]
    for x1 in range(63,-1,-1):
        for y1 in range(63,-1,-1):
            for x2 in range(x1,-1,-1):
                for y2 in range(y1,-1,-1): 
                    x1mm = x1 / 6.4
                    x2mm = x2 / 6.4
                    y1mm = y1 / 6.4
                    y2mm = y2 / 6.4
                    distance =  np.sqrt((x1mm - x2mm)**2 + (y1mm - y2mm)**2)
                    possible[round(distance, 1)].append(pearsonr(s[:,x1,y1], s[:,x2,y2])[0])
    np.save('./result/correlation_fc6_2.npy', possible)
    
    return possible

def eval_feature(feature, model, train_loader,plot= False,corr=False):
    all_x = []
    all_s = []
    model.eval()
    for x, label in train_loader:
        x = x.to('cuda')
        ## Extract pre-trained feature
        out_features = feature(x)
        out_features = out_features.detach()

        x_recon, recon_loss, KLD, s = model(out_features)
        all_s.append(s.cpu().detach())
        all_x.append(x.cpu().detach())
    all_s = torch.cat(all_s, 0)
    all_x = torch.cat(all_x, 0)
    if plot:
        Plot_MaxActImg(all_s, all_x)
    if corr:
        correlation(all_s)
    return

def main():
    config = {
        'savedir': './result/fc6_big_lr5_std10_k15_b40_exp4',
        'lr': 0.00001,
        'momentum': 0.9,
        'batchsize': 100,
        'max_epochs': 5,
    }
    ##load training dataset
    train_loader = loadtraning(config['batchsize'])

    feature = Feature().to('cuda')
    model = create_model(s_dim=4096, kernel_size=(15,15)).to('cuda')
    #model = torch.load('./result/tvae_lr5_e5_batch_mean_conv2d')

    optimizer = optim.SGD(model.parameters(),
                           lr=config['lr'],
                           momentum=config['momentum'])
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
    for e in range(config['max_epochs']):
        print('Epoch', e+1)

        total_loss, total_recon, total_kl, num_batches = train_epoch(feature, model, optimizer,
                                                                     train_loader,e)

        print('[%d] Epoch Avg Loss: %.4f, Epoch Avg Recon: %.4f, Epoch Avg KL : %.4f' % (e + 1, total_loss / num_batches,
                                                                     total_recon / num_batches, total_kl / num_batches))

        scheduler.step()
    torch.save(model,config['savedir'])

    feature.eval()
    model.eval()
    d = calculate_main(feature, model)
    np.save(config['savedir'], d)

    return model


def eval():
    config = {
        'savedir': './result/sub_lr5_std10_k7',
        'lr': 1e-5,
        'momentum': 0.9,
        'batchsize': 100,
        'max_epochs': 50,
    }

    train_loader = loaddata(config['batchsize'],'/project/qgao/subcombine/subsubtrain')

    feature = Feature().to('cuda')
    model = torch.load('./result/fc6_big_lr5_std10_k15_b15')

    eval_feature(feature, model, train_loader,plot=False, corr=True)

    return

def test():
    config = {
        'savedir': './result/birds_',
    }
    modellist=['fc6_big_lr5_std10_k15_b40','fc6_big_lr5_std10_k15_b40_exp2','fc6_big_lr5_std10_k15_b40_exp3','fc6_big_lr5_std10_k15_b40_exp4']
    feature = Feature().to('cuda')
    for i in modellist:
        feature = Feature().to('cuda')
        model = torch.load('./result/'+i)

        feature.eval()
        model.eval()
        d = calculate_main(feature, model,dirc='/project/qgao/birds')
        np.save(config['savedir']+i, d)

#main()
test()
#eval()


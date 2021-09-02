import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import torchvision.datasets as sets
import torchvision.transforms as transforms
from numpy import random
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


Alex = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

class Feature(nn.Module):

    def __init__(self, ):
        super(Feature, self).__init__()
        # Block 0: input to maxpool1
        self.block0 = nn.Sequential(
            Alex.features,
            Alex.avgpool,
        )
    def forward(self, inp):
        outp = self.block0(inp)
        outp = torch.flatten(outp, 1)
        return outp

class Classifier(nn.Module):
    def __init__(self, ):
        super(Classifier, self).__init__()
        self.fc6 = nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.final = nn.Linear(in_features=4096, out_features=1001, bias=True)

    def forward(self, inp):
        classifier6 = self.fc6(inp)
        classifier7 = self.fc7(classifier6)
        outp = self.final(classifier7)
        return classifier6,classifier7,outp


def Spatial_loss(matrix,D):
    x1 = matrix.view(-1, 4096, 1)
    Sv = torch.std(x1, dim=0)
    Xm = x1 - x1.mean(dim=0)
    C = Xm @ Xm.view(-1, 1, 4096)
    C = C.mean(dim=0)
    S = Sv @ Sv.view(1, -1)
    C = C / S
    loss = torch.abs(C - (1.0 / (D + 1)))
    kloss = loss.sum()-loss.trace()
    return kloss



def alex_training(classifier,Alex, trainloader,
                  criterion, optimizer, scheduler,device, epochsize=10):
    D = np.load('./result/D.npy')
    D = torch.from_numpy(D).to(device)
    for epoch in range(epochsize):
        print('epoch:', epoch+1)
        for inputs, classes in trainloader:
            face_inputs, face_labels = inputs.to(device), classes.to(device)
            out_features = Alex(face_inputs)
            out_features = out_features.detach()

            ##drop feature
            optimizer.zero_grad()
            classifier6,classifier7,outputs = classifier(out_features)
            loss = criterion(outputs, face_labels)
            f6loss = Spatial_loss(classifier6,D)
            f7loss = Spatial_loss(classifier7,D)

            #totalloss = 0.5*f6loss + 0.5*f7loss
            totalloss = loss + 0.0000001 * f6loss + 0.0000001*f7loss
            totalloss.backward()
            #f6loss.backward()
            optimizer.step()
            print('[%d] face loss: %.4f, fc6 loss: %.4f' %(epoch + 1,totalloss.item(),(0.0000001 * f6loss).item()))

        scheduler.step()

        with torch.no_grad():
            testcorrect = 0
            num = 0

            for inputs, classes in trainloader:
                inputs, classes = inputs.to(device), classes.to(device)

                out_features = Alex(inputs)
                ##drop feature
                _,_,outputs = classifier(out_features)
                _, predicted = torch.max(outputs.data, 1)
                num += classes.size(0)
                testcorrect += (predicted == classes).sum().item()
            print('Val face accuracy epoch: ', testcorrect/num)
    print('Finished Training')

    return  classifier

def get_mean_std( dir ='./object/training_images',ratio=0.01):
    """Get mean and std by sample ratio
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])
    trainset = sets.ImageFolder(dir, transform=transform )
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=int(len(trainset)*ratio),
                                             shuffle=True, num_workers=2)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0,2,3))
    std = np.std(train.numpy(), axis=(0,2,3))
    print(mean,std)
    return mean, std

def loaddata(batchsize,dirc):
    # transforms.RandomHorizontalFlip(),
    #mean, std = get_mean_std(dirc, ratio=0.01)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        ###change mean
        transforms.Normalize(mean=[0.49419224,0.46072632,0.4112667 ] ,
                             std=[0.27843302,0.25876182,0.2657966 ]),
    ])

    trainset = sets.ImageFolder(dirc, transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=2)
    return trainloader

from scipy.stats import pearsonr
def correlation(s,name):
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
    np.save('./result/'+name+'.npy', possible)
    
    return possible

def eval_feature(feature, model, train_loader,corr=False):
    all_6 = []
    all_7 = []
    model.eval()
    for x, label in train_loader:
        x = x.to('cuda')
        ## Extract pre-trained feature
        out_features = feature(x)
        out_features = out_features.detach()

        fc6,fc7,_ = model(out_features)
        all_6.append(fc6.cpu().detach())
        all_7.append(fc7.cpu().detach())
    all_6 = torch.cat(all_6, 0)
    all_7 = torch.cat(all_7, 0)
    if corr:
        correlation(all_6,'alex_fc6')
        correlation(all_7,'alex_fc7')
    return



def main():
    trainloader = loaddata(100, dirc = '/project/qgao/imagenet1000')

    Alex = Feature()

    classifier = Classifier()
    #classifier = torch.load('./Alex/result/alexspa_try')


    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    Alex.to(device)
    classifier.to(device)

    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma = 0.1,last_epoch=-1)
    net = alex_training(classifier, Alex,trainloader, criterion, optimizer,scheduler , device, epochsize=5)
    torch.save(net,'./result/alexspa_try2')


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
    model = torch.load('./result/alexspa_try2')

    eval_feature(feature, model, train_loader,corr=True)

    return

#eval()
main()

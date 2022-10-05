import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
import torchvision
from torchvision import transforms,datasets
from Models import Mnist_2NN,Mnist_CNN, LeNet, resnet20, HAR_LR
from getData import getLocalData, getTestData
import cppimport
import cppimport.import_hook
import random
from torch.utils.data import DataLoader
from connect import connecter
from myMPC import calculator
import copy

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="SSFL")
parser.add_argument('-nc', '--num_of_clients', type=int, default=40)
parser.add_argument('-E', '--epoch', type=int, default=1000)
parser.add_argument('-B', '--batchsize', type=int, default=256)
parser.add_argument('-mn', '--model_name', type=str, default='mnist_lenet')
parser.add_argument('-lr', "--learning_rate", type=float, default=1)
parser.add_argument('-ad', '--num_of_adversary', default=8, type=int)

m = cppimport.imp("secagg")
ROLE_SP0 = 0
ROLE_SP1 = 1
ROLE_CP = 2
ROLE_CLIENTS = 3
prec = 1000000

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    num_clients = args['num_of_clients']
    conner = {}
    mpcer = {}
    for i in range(num_clients):
        conner[i] = connecter(ROLE_CLIENTS, i, num_clients)
        mpcer[i] = calculator(conner[i], m)

    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev =  torch.device("cpu")
    net = None
    if args['model_name'] == 'mnist_lenet':
        net = LeNet()
    elif args['model_name'] == 'cifar10_resnet':
        net = resnet20()
    elif args['model_name'] == 'har_lr':
        net = HAR_LR()

    net = net.to(dev)

    local_datatset = None
    if args['model_name'] == 'mnist_lenet':
        local_datatset = getLocalData('./MNIST', 'mnist')
    elif args['model_name'] == 'cifar10_resnet':
        local_datatset = getLocalData('./CIFAR10', 'cifar')
    elif args['model_name'] == 'har_lr':
        local_datatset = getLocalData('./HAR', 'har')

    testDataLoader = None
    if args['model_name'] == 'mnist_lenet':
        testset = getTestData('./MNIST', 'mnist')       
        testDataLoader = DataLoader(testset, batch_size=256, shuffle=True)
    elif args['model_name'] == 'cifar10_resnet':
        testset = getTestData('./CIFAR10', 'cifar')  
        testDataLoader = DataLoader(testset, batch_size=256, shuffle=True)
    elif args['model_name'] == 'har_lr':
        testset = getTestData('./HAR', 'har')  
        testDataLoader = DataLoader(testset, batch_size=256, shuffle=True)

    local_parameters = {}
    parameters_name = []
    for key, var in net.state_dict().items():
        local_parameters[key] = var.clone()
        parameters_name.append(key)

    parm_lenth = {}
    for name in parameters_name:
        parm_lenth[name] = len(local_parameters[name].view(-1).tolist())
    
    for i in range(num_clients):
        parmlist_i = mpcer[i].restruct_recv(ROLE_SP0, ROLE_SP1)

    parmlist = []
    for j in range(len(parmlist_i)):
        parmlist.append(parmlist_i[j] / prec)
    
    global_parameters = {}
    start = 0
    end = parm_lenth[parameters_name[0]]
    for j in range(len(parameters_name)):
        global_parameters[parameters_name[j]] = torch.Tensor(parmlist[start:end]).view_as(
            local_parameters[parameters_name[j]]
        )
        if j < len(parameters_name)-1:
            start += parm_lenth[parameters_name[j]]
            end += parm_lenth[parameters_name[j+1]]

    uslr = 0.1
    history_updata = None
    loss = None
    i = 0
    skip_num = 0
    useful_grad = None
    all_range = list(range(len(local_datatset)))
    random.shuffle(all_range)
    data_len = int(len(local_datatset) / num_clients)
    m_d = None
    m_l = None
    source_class = 1
    target_class = 3
    net.train()
    for epoch in range(args['epoch']):
        for i in range(num_clients):

            indices = all_range[i * data_len: (i + 1) * data_len]
            net.load_state_dict(global_parameters, strict=False)
            if i >= args['num_of_adversary']:
                random.shuffle(indices)
            train_dl = torch.utils.data.DataLoader(
                local_datatset,
                batch_size=args['batchsize'],
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

            optimizer = torch.optim.SGD(net.parameters(), lr=uslr, momentum=0.9)
            for data, label in train_dl:
                label = label.long()
                data, label = data.to(dev), label.to(dev)
                optimizer.zero_grad()
                output = net(data)

                loss = torch.nn.functional.cross_entropy(output, label)
                loss.backward()
                optimizer.step()

            local_grad = []
            for name, parms in net.state_dict().items():
                local_grad.extend((global_parameters[name] - parms).view(-1).tolist())
                
            Grad = []
            for j in range(len(local_grad)):
                Grad.append(int(local_grad[j] * prec))
            mpcer[i].share_send(ROLE_SP0, ROLE_SP1, Grad)


            conner[i].send(conner[i].conn['id{}'.format(ROLE_CP)], np.std(local_grad))
        for i in range(num_clients):
            updata_i = mpcer[i].restruct_recv(ROLE_SP0, ROLE_SP1)
        updatalist = []
        for j in range(len(updata_i)):
            updatalist.append(updata_i[j] / (prec * prec))

        for j in range(len(parmlist)):
            parmlist[j] -= updatalist[j] * args['learning_rate']
        
        start = 0
        end = parm_lenth[parameters_name[0]]
        for j in range(len(parameters_name)):
            global_parameters[parameters_name[j]] = torch.Tensor(parmlist[start:end]).view_as(
                local_parameters[parameters_name[j]]
            )
            if j < len(parameters_name)-1:
                start += parm_lenth[parameters_name[j]]
                end += parm_lenth[parameters_name[j+1]]

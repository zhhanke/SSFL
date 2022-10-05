import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN, LeNet, HAR_LR, resnet20
import cppimport
import cppimport.import_hook
from myMPC import calculator
from myThread import myThread
from connect import connecter
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="SSFL")
parser.add_argument('-nc', '--num_of_clients', type=int, default=40)
parser.add_argument('-B', '--batchsize', type=int, default=10)
parser.add_argument('-mn', '--model_name', type=str, default='mnist_lenet')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000)

m = cppimport.imp("secagg")
ROLE_SP0 = 0
ROLE_SP1 = 1
ROLE_CP = 2
ROLE_CLIENTS = 3
prec = 1000000

if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    conner = connecter(ROLE_CP, 0, int(args['num_of_clients']))
    mpcer = calculator(conner, m)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_lenet':
        net = LeNet()
    elif args['model_name'] == 'cifar10_resnet':
        net = resnet20()
    elif args['model_name'] == 'har_lr':
        net = HAR_LR()

    net = net.to(dev)

    num_clients = int(args['num_of_clients'])
    clients_in_comm = ['client{}'.format(i) for i in range(num_clients)]

    global_parameters = {}
    parameters_name = []
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
        parameters_name.append(key)

    parmlist = []
    for name in parameters_name:
        parmlist.extend(global_parameters[name].view(-1).tolist())
    for j in range(len(parmlist)):
        parmlist[j] = int(parmlist[j] * prec)
    
    mpcer.share_send(ROLE_SP0, ROLE_SP1, parmlist)

    mpcer.getmask_test(ROLE_CP, num_clients, len(parmlist))

    conner.StartRecord()
    total_send = 0
    total_recv = 0

    for i in tqdm(range(args['num_comm'])):
        sigmaG = []
        for j in range(num_clients):
            ss = conner.recv(conner.conn['id{}'.format(ROLE_CLIENTS+j)])
            sigmaG.append(ss)

        R = {}
        Rmed = []
        for client in clients_in_comm:
            R[client] = mpcer.restruct_recv(ROLE_SP0,ROLE_SP1)

        for j in range(len(R["client1"])):
            temp = []
            for client in clients_in_comm:
                temp.append(R[client][j])
            Rmed.append(int(np.median(temp)))

        mpcer.share_send(ROLE_SP0, ROLE_SP1, Rmed)

        # compute E
        E = mpcer.restruct_recv(ROLE_SP0, ROLE_SP1)
        #print(E)
        for j in range(len(E)):
            E[j] = E[j]/ len(parmlist)
            E[j] = (E[j]/ prec)/prec
        # compute sigmaM
        sigmaM = mpcer.restruct_recv(ROLE_SP0, ROLE_SP1)
        sigmaM = sigmaM[0] / len(parmlist)
        sigmaM = np.sqrt(sigmaM)
        sigmaM = sigmaM / prec

        beta = []
        myu = []
        total = 0
        for j in range(num_clients):
            rho = (E[j]) / (sigmaM * sigmaG[j])
            with open('./result/rho.txt', 'a') as file_object:
                file_object.write('{},'.format(rho))
            if rho >= 1:
                rho = 0.9
            
            myu.append(max(0, np.log(((1 + rho) / (1 - rho))).item()-0.5))
            if j < 0:
                myu[j] = 0

            #myu[j] = 1
            total += myu[j]
        with open('./result/rho.txt', 'a') as file_object:
                file_object.write('\n')

        for j in range(num_clients):
            if total != 0:
                beta.append(myu[j]/total)
            else:
                beta.append(1/num_clients)
        
        beta_int = []
        for j in range(len(beta)):
            beta_int.append(int(beta[j]* prec))
        
        mpcer.share_send(ROLE_SP0, ROLE_SP1, beta_int)
    
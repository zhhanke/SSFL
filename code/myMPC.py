import pickle
import numpy as np
import random
import time

ROLE_SP0 = 0
ROLE_SP1 = 1
ROLE_CP = 2
ROLE_CLIENTS = 3
prec = 100000
alpha = 233
phi = 233
class calculator(object):
    def __init__(self, conn, math):
        self.conn = conn
        self.m = math

    def share_send(self, target0, target1, data):
        grab = self.m.get_garble(len(data))
        share0 = list(grab)
        share1 = list(self.m.vector_sub(self.m.VectorInt(data),grab))
        self.conn.send(self.conn.conn['id{}'.format(target0)], share0)
        self.conn.send(self.conn.conn['id{}'.format(target1)], share1)

    def share_send_sec(self, target0, target1, data, role, ver):
        grab = self.m.get_garble(len(data))
        share0 = list(grab)
        share1 = list(self.m.vector_sub(self.m.VectorInt(data),grab))
        self.conn.send(self.conn.conn['id{}'.format(target0)], share0)
        self.conn.send(self.conn.conn['id{}'.format(target1)], share1)

        if role == ROLE_CP:
            gamma = []
            delta = []
            for i in range(len(data)):
                temp = random.randint(0,prec)
                delta.append(temp)
                gamma.append(np.int64(ver * (data[i] + delta[i]) ))
            grab = self.m.get_garble(len(gamma))
            share0 = list(grab)
            share1 = list(self.m.vector_sub(self.m.VectorInt(gamma),grab))
            self.conn.send(self.conn.conn['id{}'.format(target0)], share0)
            self.conn.send(self.conn.conn['id{}'.format(target1)], share1)

            self.conn.send(self.conn.conn['id{}'.format(target0)], delta)
            self.conn.send(self.conn.conn['id{}'.format(target1)], delta)

        if role >= ROLE_CLIENTS:
            self.conn.send(self.conn.conn['id{}'.format(target0)], ver['gamma_index'])
            self.conn.send(self.conn.conn['id{}'.format(target1)], ver['gamma_index'])

            self.conn.send(self.conn.conn['id{}'.format(target0)], ver['delta'])
            self.conn.send(self.conn.conn['id{}'.format(target1)], ver['delta'])

    def share_recv(self, source):
        return self.conn.recv(self.conn.conn['id{}'.format(source)])

    def share_recv_sec(self, source):
        data_share = self.conn.recv(self.conn.conn['id{}'.format(source)])

        gamma_share = self.conn.recv(self.conn.conn['id{}'.format(source)])
        delta = self.conn.recv(self.conn.conn['id{}'.format(source)])
        
        res = {}
        res['data_share'] = data_share
        res['gamma_share'] = gamma_share
        res['delta'] = delta
        return res


    def restruct_send(self, target, data):
        self.conn.send(self.conn.conn['id{}'.format(target)], data)

    def restruct_recv(self, source0, source1):
        share0 = self.conn.recv(self.conn.conn['id{}'.format(source1)])
        share1 = self.conn.recv(self.conn.conn['id{}'.format(source0)])
        return list(self.m.vector_add(self.m.VectorInt(share0), self.m.VectorInt(share1)))

    def vector_mul_helper(self, target0, target1, lenth):
        A = list(self.m.get_garble(lenth))
        B = list(self.m.get_garble(lenth))
        C = []
        C.append(self.m.vector_mul(self.m.VectorInt(A), self.m.VectorInt(B)))
        
        self.share_send(target0, target1, A)
        self.share_send(target0, target1, B)
        self.share_send(target0, target1, C)

    def vector_mul_partner(self, another, helper, X_share, Y_share):
        A_share = (self.share_recv(helper))
        B_share = (self.share_recv(helper))
        C_share = (self.share_recv(helper))

        E_share = list(self.m.vector_sub(self.m.VectorInt(X_share), self.m.VectorInt(A_share)))
        F_share = list(self.m.vector_sub(self.m.VectorInt(Y_share), self.m.VectorInt(B_share)))

        if another == 0:
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            self.conn.send(self.conn.conn['id{}'.format(another)], F_share)
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            F_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
        else:
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            F_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            self.conn.send(self.conn.conn['id{}'.format(another)], F_share)

        E = (self.m.vector_add(self.m.VectorInt(E_share), self.m.VectorInt(E_share_)))
        F = (self.m.vector_add(self.m.VectorInt(F_share), self.m.VectorInt(F_share_)))

        res = []
        res.append(int(self.m.vector_mul(self.m.VectorInt(X_share), F)))
        temp = []
        temp.append(int(self.m.vector_mul(self.m.VectorInt(Y_share), E)))
        res = self.m.vector_add(self.m.VectorInt(res), self.m.VectorInt(temp))
        res = self.m.vector_add(self.m.VectorInt(res), self.m.VectorInt(C_share))
        if another == 0:
            temp = []
            temp.append(int(self.m.vector_mul(E, F)))
            res = self.m.vector_sub(self.m.VectorInt(res), self.m.VectorInt(temp))
        return list(res)[0]

    def vector_squ_partner(self, another, X_share, X_mask, res_mask, i):
        E_share = list(self.m.vector_sub(self.m.VectorInt(X_share), self.m.VectorInt(X_mask)))
        
        if another == 0:
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
        else:
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)

        E = self.m.vector_add(self.m.VectorInt(E_share), self.m.VectorInt(E_share_))

        temp = []
        temp.append(int(self.m.vector_mul(self.m.VectorInt(X_share), E)))
        res = []
        res.append(temp[0])
        res = self.m.vector_add(self.m.VectorInt(res), self.m.VectorInt(temp))
        res = self.m.vector_add(res, self.m.VectorInt(res_mask))
        if another == 0:
            temp = []
            temp.append(int(self.m.vector_mul(E, E)))
            res = self.m.vector_sub(res, self.m.VectorInt(temp))

        return list(res)[0]
    

    def vecmat_mul_helper(self, target0, target1, len0, len1):
        A = list(self.m.get_garble(len0))
        B = list(self.m.get_garble(len1))
        C = []
        C.extend(list(self.m.vecmat_mul(self.m.VectorInt(A), self.m.VectorInt(B))))
        
        self.share_send(target0, target1, A)
        self.share_send(target0, target1, B)
        self.share_send(target0, target1, C)

    def vecmat_mul_partner(self, another, X_share, Y_share, X_mask, Y_mask, res_mask):
        A_share = X_mask
        B_share = Y_mask
        C_share = res_mask

        E_share = list(self.m.vector_sub(self.m.VectorInt(X_share), self.m.VectorInt(A_share)))
        F_share = list(self.m.vector_sub(self.m.VectorInt(Y_share), self.m.VectorInt(B_share)))
        if another == 0:
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            self.conn.send(self.conn.conn['id{}'.format(another)], F_share)
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            F_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
        else:
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            F_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            self.conn.send(self.conn.conn['id{}'.format(another)], F_share)

        E = (self.m.vector_add(self.m.VectorInt(E_share), self.m.VectorInt(E_share_)))
        F = (self.m.vector_add(self.m.VectorInt(F_share), self.m.VectorInt(F_share_)))

        res = self.m.vecmat_mul(self.m.VectorInt(X_share), F)
        temp = self.m.vecmat_mul(E, self.m.VectorInt(Y_share))
        res = self.m.vector_add(res, temp)
        res = self.m.vector_add(res, self.m.VectorInt(C_share))
        if another == 0:
            temp = self.m.vecmat_mul(E, F)
            res = self.m.vector_sub(res, temp)
        return list(res)

    def vecmat_mul_partner_sec(self, another, X_share, Y_share, X_mask, Y_mask, res_mask, X_ver, Y_ver, C_delta, C_gamma):
        A_share = X_mask
        B_share = Y_mask
        C_share = res_mask

        E_share = list(self.m.vector_sub(self.m.VectorInt(X_share), self.m.VectorInt(A_share)))
        F_share = list(self.m.vector_sub(self.m.VectorInt(Y_share), self.m.VectorInt(B_share)))
        if another == 0:
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            self.conn.send(self.conn.conn['id{}'.format(another)], F_share)
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            F_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
        else:
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            F_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            self.conn.send(self.conn.conn['id{}'.format(another)], F_share)

        E = (self.m.vector_add(self.m.VectorInt(E_share), self.m.VectorInt(E_share_)))
        F = (self.m.vector_add(self.m.VectorInt(F_share), self.m.VectorInt(F_share_)))

        res = self.m.vecmat_mul(self.m.VectorInt(X_share), F)
        temp = self.m.vecmat_mul(E, self.m.VectorInt(Y_share))
        res = self.m.vector_add(res, temp)
        res = self.m.vector_add(res, self.m.VectorInt(C_share))
        if another == 0:
            temp = self.m.vecmat_mul(E, F)
            res = self.m.vector_sub(res, temp)

        # delta
        delta = self.m.vecmat_mul(self.m.VectorInt(X_ver['delta']), F)
        temp =  self.m.vecmat_mul(E, self.m.VectorInt(Y_ver['delta']))
        delta = self.m.vector_add(delta, temp)
        delta = self.m.vector_sub(delta, self.m.VectorInt(C_delta))
        temp = self.m.vecmat_mul(E, F)
        delta = self.m.vector_add(delta, temp)

        # gamma
        gamma = self.m.vecmat_mul(self.m.VectorInt(X_ver['gamma']), F)
        temp =  self.m.vecmat_mul(E, self.m.VectorInt(Y_ver['gamma']))
        gamma = self.m.vector_add(gamma, temp)
        gamma = self.m.vector_add(gamma, self.m.VectorInt(C_gamma))

        result = {}
        result['data'] = list(res)
        result['delta'] = list(delta)
        result['gamma'] = list(gamma)

        return result

    def getmask_test(self, role, num_clients, grad_len):
        res = []
        if role == ROLE_CP:
            Med_mask = list(self.m.get_garble(grad_len))
            G_mask = list(self.m.get_garble(grad_len * num_clients))
            M_G_mask = []
            M_G_mask.extend(list(self.m.vecmat_mul(self.m.VectorInt(Med_mask), self.m.VectorInt(G_mask))))
            M_M_mask = []
            M_M_mask.append(self.m.vector_mul(self.m.VectorInt(Med_mask), self.m.VectorInt(Med_mask)))
            G_G_mask = []
            for i in range(num_clients):
                G_single = G_mask[i*grad_len:(i+1)*grad_len]
                G_G_mask.append(self.m.vector_mul(self.m.VectorInt(G_single), self.m.VectorInt(G_single)))

            B_mask = list(self.m.get_garble(num_clients))
            B_G_mask = []
            B_G_mask.extend(list(self.m.vecmat_mul(self.m.VectorInt(B_mask), self.m.VectorInt(G_mask))))

            self.share_send(ROLE_SP0, ROLE_SP1, Med_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, G_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, M_G_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, M_M_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, G_G_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, B_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, B_G_mask)

        else:
            Med_mask = self.share_recv(ROLE_CP)
            print('get Med_mask')
            G_mask = self.share_recv(ROLE_CP)
            print('get G_mask')
            M_G_mask = self.share_recv(ROLE_CP)
            print('get M_G_mask')
            M_M_mask = self.share_recv(ROLE_CP)
            print('get M_M_mask')
            G_G_mask = self.share_recv(ROLE_CP)
            print('get G_G_mask')
            B_mask = self.share_recv(ROLE_CP)
            print('get B_mask')
            B_G_mask = self.share_recv(ROLE_CP)
            print('get B_G_mask')

            temp = {}
            temp['Med_mask'] = Med_mask
            temp['G_mask'] = G_mask
            temp['M_G_mask'] = M_G_mask
            temp['M_M_mask'] = M_M_mask
            temp['G_G_mask'] = G_G_mask
            temp['B_mask'] = B_mask
            temp['B_G_mask'] = B_G_mask

            res.append(temp)
        return res

    def getmask_ver(self, role, num_clients, grad_len):
        res = []
        if role == ROLE_CP:
            Med_mask = list(self.m.get_garble(grad_len))
            G_mask = list(self.m.get_garble(grad_len * num_clients))
            M_G_mask = []
            M_G_mask.extend(list(self.m.vecmat_mul(self.m.VectorInt(Med_mask), self.m.VectorInt(G_mask))))
            M_M_mask = []
            M_M_mask.append(self.m.vector_mul(self.m.VectorInt(Med_mask), self.m.VectorInt(Med_mask)))
            G_G_mask = []
            for i in range(num_clients):
                G_single = G_mask[i*grad_len:(i+1)*grad_len]
                G_G_mask.append(self.m.vector_mul(self.m.VectorInt(G_single), self.m.VectorInt(G_single)))

            B_mask = list(self.m.get_garble(num_clients))
            B_G_mask = []
            B_G_mask.extend(list(self.m.vecmat_mul(self.m.VectorInt(B_mask), self.m.VectorInt(G_mask))))

            self.share_send(ROLE_SP0, ROLE_SP1, Med_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, G_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, M_G_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, M_M_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, G_G_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, B_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, B_G_mask)
            
            B_G_delta = []
            B_G_gamma = []
            for i in range(len(B_G_mask)):
                B_G_delta.append(random.randint(0,prec))
            B_G_gamma = list(self.m.vector_add(self.m.VectorInt(B_G_delta),self.m.VectorInt(B_G_mask)))
            for i in range(len(B_G_gamma)):
                B_G_gamma[i] = np.int64(B_G_gamma[i]) * alpha
            self.share_send(ROLE_SP0, ROLE_SP1, B_G_gamma)
            
            self.conn.send(self.conn.conn['id{}'.format(ROLE_SP0)], B_G_delta)
            self.conn.send(self.conn.conn['id{}'.format(ROLE_SP1)], B_G_delta)

        else:
            Med_mask = self.share_recv(ROLE_CP)
            print('get Med_mask')
            G_mask = self.share_recv(ROLE_CP)
            print('get G_mask')
            M_G_mask = self.share_recv(ROLE_CP)
            print('get M_G_mask')
            M_M_mask = self.share_recv(ROLE_CP)
            print('get M_M_mask')
            G_G_mask = self.share_recv(ROLE_CP)
            print('get G_G_mask')
            B_mask = self.share_recv(ROLE_CP)
            print('get B_mask')
            B_G_mask = self.share_recv(ROLE_CP)
            print('get B_G_mask')
            B_G_gamma = self.share_recv(ROLE_CP)
            B_G_delta = self.share_recv(ROLE_CP)

            temp = {}
            temp['Med_mask'] = Med_mask
            temp['G_mask'] = G_mask
            temp['M_G_mask'] = M_G_mask
            temp['M_M_mask'] = M_M_mask
            temp['G_G_mask'] = G_G_mask
            temp['B_mask'] = B_mask
            temp['B_G_mask'] = B_G_mask
            temp['B_G_gamma'] = B_G_gamma
            temp['B_G_delta'] = B_G_delta
            res.append(temp)
        return res

    def verify(self, role, ver):
        if role == ROLE_SP0 or role == ROLE_SP1:
            delta = []
            gamma = []
            temp = []
            for i in range(10):
                temp.append(phi)
                delta.append(ver['delta'][i])
                gamma.append(ver['gamma'][i])
            delta = list(self.m.vector_add(self.m.VectorInt(delta), self.m.VectorInt(temp)))
            self.conn.send(self.conn.conn['id{}'.format(ROLE_CP)], delta)
            self.conn.send(self.conn.conn['id{}'.format(ROLE_CP)], gamma)
        if role == ROLE_CLIENTS:
            data = []
            temp = []
            for i in range(10):
                temp.append(phi)
                data.append(ver['data'][i])
                #data.append(pow(phi,ver['data'][i],2**64))
            data = list(self.m.vector_sub(self.m.VectorInt(data), self.m.VectorInt(temp)))
            self.conn.send(self.conn.conn['id{}'.format(ROLE_CP)], data)
        if role == ROLE_CP:
            delta0 = self.conn.recv(self.conn.conn['id{}'.format(ROLE_SP0)])
            gamma0 = self.conn.recv(self.conn.conn['id{}'.format(ROLE_SP0)])
            delta1 = self.conn.recv(self.conn.conn['id{}'.format(ROLE_SP1)])
            gamma1 = self.conn.recv(self.conn.conn['id{}'.format(ROLE_SP1)])
            data=self.conn.recv(self.conn.conn['id{}'.format(ROLE_CLIENTS)])
            
            tic1 = time.perf_counter()
            temp1 = list(self.m.vector_add(self.m.VectorInt(gamma0), self.m.VectorInt(gamma1)))
            delta0 = self.m.VectorInt(delta0)
            data = self.m.VectorInt(data)
            temp0 = list(self.m.vector_add(delta0, data))
            
            for i in range(10):
                x = abs((np.int64(temp0[i])*np.int64(alpha) - temp1[i])/temp1[i])
                if x > 0.1:
                    print(np.int64(temp0[i])*np.int64(alpha), temp1[i])
            tic2 = time.perf_counter()
            print(tic2-tic1)
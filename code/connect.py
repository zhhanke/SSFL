import socket
import pickle
import struct

ROLE_SP0 = 0
ROLE_SP1 = 1
ROLE_CP  = 2
ROLE_CLIENTS = 3

CP_PORT  = 8000
SP0_PORT = 8001
SP1_PORT = 8002
class connecter(object):
    def __init__(self, identity, sernum, clientnum):
        self.record_flag = False
        self.record_send = 0
        self.record_recv = 0
        self.role = identity
        if self.role != ROLE_CLIENTS:
            print('connect...')
        if identity == ROLE_CP:
            self.conn = {}
            sock = socket.socket()
            sock.bind(('127.0.0.1', CP_PORT))
            sock.listen()
            for i in range(clientnum + 2):
                sour, addr = sock.accept()
                id = self.recv(sour)
                self.conn[id] = sour

        if identity == ROLE_SP0:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', CP_PORT))
            self.send(sock, 'id{}'.format(identity))
            self.conn = {}
            self.conn['id2'] = sock

            sock = socket.socket()
            sock.bind(('127.0.0.1', SP0_PORT))
            sock.listen()
            for i in range(clientnum + 1):
                sour, addr = sock.accept()
                id = self.recv(sour)
                self.conn[id] = sour

        if identity == ROLE_SP1:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', CP_PORT))
            self.send(sock, 'id{}'.format(identity))
            self.conn = {}
            self.conn['id2'] = sock

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', SP0_PORT))
            self.send(sock, 'id{}'.format(identity))
            self.conn['id0'] = sock

            sock = socket.socket()
            sock.bind(('127.0.0.1', SP1_PORT))
            sock.listen()
            for i in range(clientnum):
                sour, addr = sock.accept()
                id = self.recv(sour)
                self.conn[id] = sour

        if identity == ROLE_CLIENTS:
            self.conn = {}
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', CP_PORT))
            self.send(sock, 'id{}'.format(sernum+identity))
            self.conn['id2'] = sock

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', SP0_PORT))
            self.send(sock, 'id{}'.format(sernum+identity))
            self.conn['id0'] = sock

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', SP1_PORT))
            self.send(sock, 'id{}'.format(sernum+identity))
            self.conn['id1'] = sock

    def send(self, target, data):
        data_pickle = pickle.dumps(data)

        headers = {'data_size': len(data_pickle)}
        head_pickle = pickle.dumps(headers)

        target.send(struct.pack('i', len(head_pickle)))
        target.send(head_pickle)
        target.sendall(data_pickle)
        #print('send:{}'.format(len(data_pickle)))
        if self.record_flag == True:
            self.record_send += len(data_pickle)

    def recv(self, source):
        head = source.recv(4)
        head_pickle_len = struct.unpack('i', head)[0]

        head_pickle = pickle.loads(source.recv(head_pickle_len))
        data_len = head_pickle['data_size']

        # 开始接收数据
        '''
        recv_size = 0
        remain_size = data_len
        recv_data = b''
        while recv_size < data_len:
            if remain_size >= 65536:
                recv_data += source.recv(65536)
                remain_size -= 65536
            else:
                recv_data += source.recv(remain_size)
            recv_size = len(recv_data)
        '''
        recv_data = source.recv(data_len, socket.MSG_WAITALL)
        data = pickle.loads(recv_data)
        if self.record_flag == True:
            self.record_recv += data_len

        #print('receive:{}'.format(len(recv_data)))
        return data

    def StartRecord(self):
        self.record_flag = True

    def CleanRecord(self):
        self.record_send = 0
        self.record_recv = 0

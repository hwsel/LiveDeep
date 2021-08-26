import numpy as np
import cv2 as cv

import csv
import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time
from matplotlib.lines import Line2D
from six import iteritems
import math


import VideoPip as data


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from UserTrace import LocationCalculate,userLocal_One, LocationCalculateB

class UserTraceEst(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    def output_y_hc(self, x, hc):
        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc


def estUserTrace(User,Cur):
    DATA_IN_X=[]
    DATA_IN_Y=[]

    for i in range (8):
        DATA_IN_X.append(i)
        DATA_IN_Y.append(i)
    
    for i in range (8):
        DATA_IN_X[i]=User[Cur+i-8][0]
        DATA_IN_Y[i]=User[Cur+i-8][1]

    inp_dim = 1
    out_dim = 1
    mid_dim = 8
    mid_layers = 2
    batch_size = 8
    mod_dir = '.'

    '''load data'''
    #data = load_data()
    data_X=[1,2,3,4,5,6,7,8]
    data_TX=[9,10,11,12,13,14,15,16]
    data_X=np.array(data_X)
    data_TX=np.array(data_TX)
    data_Yx = DATA_IN_X#[4,5,4,5,4,4,4,4]#[4,5,4,5,6,7,3,4]
    data_Yx=np.array(data_Yx)
    data_Yy = DATA_IN_Y#[4,5,4,5,6,7,3,4]
    #assert data_X.shape[0] == inp_dim

    #print("original x:")
    #print(data_x)

    train_size = int(len(data_X) )
    print(train_size,len(data_X),data_X)
    data_X = data_X[:train_size]
    data_TX=data_TX[:train_size]
    data_Yx = data_Yx[:train_size]
    data_Yy = data_Yy[:train_size]
    data_X = data_X.reshape((train_size, inp_dim))
    data_TX=data_TX.reshape((train_size, inp_dim))
    data_Yx = data_Yx.reshape((train_size, out_dim))
    data_Yy = data_Yy.reshape((train_size, out_dim))

    

    #print(inp_dim, out_dim, mid_dim, mid_layers)
    #return 0

    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''train'''
    var_x = torch.tensor(data_X, dtype=torch.float32, device=device)
    var_y = torch.tensor(data_Yx, dtype=torch.float32, device=device)
    var_yy = torch.tensor(data_Yy, dtype=torch.float32, device=device)

    data_TX
    var_Tx = torch.tensor(data_TX, dtype=torch.float32, device=device)
    var_y = torch.tensor(data_Yx, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_Tx = list()
    batch_var_y = list()
    batch_var_yy = list()

    for i in range(batch_size):
        j = train_size - i
        batch_var_x.append(var_x[j:])
        batch_var_Tx.append(var_Tx[j:])
        batch_var_y.append(var_y[j:])
        batch_var_yy.append(var_yy[j:])
    
    print(len(batch_var_y[0]))
    #print(len(train_y))
    #return 0

    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_Tx = pad_sequence(batch_var_Tx)
    batch_var_y = pad_sequence(batch_var_y)
    batch_var_yy = pad_sequence(batch_var_yy)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(data_Yx)) * (np.e / len(data_Yx)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Training Start")
    for e in range(384): #384
        out = net(batch_var_x)
    
        # loss = criterion(out, batch_var_y)
        loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if e % 64 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
    #torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))
    #print("Save in:", '{}/net.pth'.format(mod_dir))


    print("finish training========")

    outA = net(batch_var_Tx)
    #print(out)

    print("Training B Start")
    for e in range(384): #384
        out = net(batch_var_x)
    
        # loss = criterion(out, batch_var_y)
        loss = (out - batch_var_yy) ** 2 * weights
        loss = loss.mean()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if e % 64 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
    #torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))
    #print("Save in:", '{}/net.pth'.format(mod_dir))

    outB = net(batch_var_Tx)
    return int(outA[0][7].item()),int(outB[0][7].item())


def get_data():
    
    UserOverall = []
    str = TimeStamp[0].split(':')
    PreTime = math.ceil(float(str[2]))
    CurTime = math.floor(float(str[2]))
    i = 0
    j = 0
    FrameWidthT = 1280
    FrameHeightT = 720
    Frame30 = int(cap.get(5))
    FrameCount = Frame30 + 1

    user_list = []
    frame_list = []

    AnalyzeUser = []
    count = 0
    frame_number = 0
    while (cap.isOpened()):

        ret, frame = cap.read()



        
        if FrameCount >= Frame30:
            UserIn1s = []
            UserIn1sR = []
            Ind = 0
            while PreTime > CurTime:
                str = TimeStamp[j].split(':')
                # print(str[2])
                # Ind=Ind+1
                # print([Ind,j])
                CurTime = math.floor(float(str[2]))
                if CurTime == 0 and PreTime == 60:
                    break
                UserAll = []
                for k in range(0, A):
                    # print(k,j)
                    x = ProceList[k][j][1]
                    y = ProceList[k][j][2]
                    z = ProceList[k][j][3]
                    w = ProceList[k][j][4]
                    H, W = LocationCalculate(x, y, z, w)
                    IH = math.floor(H * FrameHeightT)
                    IW = math.floor(W * FrameWidthT)
                    UserAll.append([IW, IH])
                    # print(IW,IH)
                j = j + 1
                UserIn1s.append(UserAll)
            FrameCount = 0
            PreTime = CurTime + 1

            
            if len(UserIn1s) < 30:
                for k in range(0, len(UserIn1s)):
                    UserIn1sR.append(UserIn1s[k])
                for k in range(len(UserIn1s) - 1, 30):
                    UserIn1sR.append(UserIn1s[len(UserIn1s) - 1])
            else:
                for k in range(0, 30):
                    UserIn1sR.append(UserIn1s[k])
                # UserIn1s.append(UserAllR)
        else:

            for k in range(0, A):
                x = UserIn1sR[FrameCount][k][0]
                y = UserIn1sR[FrameCount][k][1]
                cv.circle(frame, (x, y), 10, color=(0, 255, 255))


            FrameCount = FrameCount + 1
            if(FrameCount>=60):
                return user_list, frame_list

        # cv.line(frame, (426, 0), (426, 720), thickness=5, color=(0, 0, 255))
        # cv.line(frame, (852, 0), (852, 720), thickness=5, color=(0, 0, 255))
        # cv.line(frame, (0, 240), (1280, 240), thickness=5, color=(0, 0, 255))
        # cv.line(frame, (0, 480), (1280, 480), thickness=5, color=(0, 0, 255))
        # cv.imshow('frame', frame)
        user_list.append((x, y))
        if x<0 or y<0:
            x=0
            y=0
            print("Error ：",frame_number)
        frame_list.append(frame)
        frame_number+=1
        if frame_number>=60:
            return user_list, frame_list
        print((x , y))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        print(count)
        if count > 12600:
            break

    # cap.release()
    # cv.destroyAllWindows()


def processed_dataB(grid_size=3, indexB = 1,u=[],f=[]):
    #u , f = get_data()
    p_u , p_f = [], []
    print("Data",f[0].shape,"Data",len(f),"Data",len(u))
    print("Data",u)
    for indexC in range(grid_size*grid_size):
        index=indexC+1
        for i in range(len(u)):
            frame = f[i]
            row = math.ceil(index/grid_size)
            col = (index-1)%grid_size
            frame_part = frame[int((720/grid_size)*(row-1)):int((720/grid_size)*(row)), int((1280/grid_size)*(col)):int((1280/grid_size)*(col+1)), :]
            #print("index",index,len(frame_part))
            frame_part = cv.resize(frame_part, (120, 120))
            p_f.append(frame_part)
            # print(int((720/grid_size)*(row-1)), int((720/grid_size)*(row)), int((1280/grid_size)*(col)), int((1280/grid_size)*(col+1)))

            pos = u[i]
            if pos[0]<=(1280/grid_size)*(row) and  pos[0]>=(1280/grid_size)*(row-1) and (pos[1]<=(720/grid_size)*(col+1) and  pos[0]>=(720/grid_size)*(col)):
                p_u.append(1)
                #print("===================================there is one======================")
            else:
                p_u.append(0)

    p_u = np.asarray(p_u)
    p_f = np.asarray(p_f)
    print("Data",  p_f.shape, "Data", len(p_f), "Data", len(p_u),"Data",p_u.shape)

    return p_u, p_f

def processed_dataC(grid_size=3, u=[] , f=[]):
    p_u, p_f = [], []
    for j in range(len(f)):
        pos = u[j]
        for i in range(grid_size*grid_size):
            row = math.floor(i / grid_size)
            col = (i) % grid_size
            frame_part = f[j][int((720 / grid_size) * (row )):int((720 / grid_size) * (row+1)),
                         int((1280 / grid_size) * (col)):int((1280 / grid_size) * (col + 1)), :]
            frame_part = cv.resize(frame_part, (32, 32))  # 120*120
            p_f.append(frame_part)
'''
            
            # print("pose",pos)
            # if pos[0]<=(1280/grid_size)*(row) and  pos[0]>=(1280/grid_size)*(row-1) and (pos[1]<=(720/grid_size)*(col+1) and  pos[0]>=(720/grid_size)*(col)):
            if pos[1] < (720 / grid_size) * (row) and pos[1] >= (720 / grid_size) * (row - 1) and (
                    pos[0] < (1280 / grid_size) * (col + 1) and pos[0] >= (1280 / grid_size) * (col)):
                p_u.append(1)
                print(index - 1, '===============================', pos, row - 1, col)
                # print("===================================there is one======================")
            else:
                p_u.append(0)
'''

def CalculateUserView(x, y):
    #1280 720
    #x=700
    #y=360
    W=150
    H=150
    UWL=x-W
    UHL=y-H
    UWH=x+W
    UHH=y+H

    if UWL<0:
        UWL=0
    if UHL<0:
        UHL=0
    if UWH>1280:
        UWH=1280
    if UHH>720:
        UHH=720
    return UWL, UHL ,UWH, UHH

def IfPInUserView(x, y,xp,yp):
    #1280 720
    #x=700
    #y=360
    W=200
    H=200
    UWL=x-W
    UHL=y-H
    UWH=x+W
    UHH=y+H

    if UWL<0:
        UWL=0
    if UHL<0:
        UHL=0
    if UWH>1280:
        UWH=1280
    if UHH>720:
        UHH=720
    flag=0
    if UWL<xp and xp<UWH and UHL<yp and UHH>yp:
        flag=1
    return flag



def processed_dataC(grid_size=3, index = 1,u=[] , f=[]):
    #u , f = get_data()
    p_u , p_f = [], []
    Uview=[]

    #print("Data",f[0].shape,"Data",len(f),"Data",len(u))
    #print("Data",u)
    for i in range(len(u)):
        frame = f[i]
        row = math.ceil(index/grid_size)
        col = (index-1)%grid_size
        frame_part = frame[int((720/grid_size)*(row-1)):int((720/grid_size)*(row)), int((1280/grid_size)*(col)):int((1280/grid_size)*(col+1)), :]
        frame_part = cv.resize(frame_part, (32, 32))      #120*120
        p_f.append(frame_part)
        # print(int((720/grid_size)*(row-1)), int((720/grid_size)*(row)), int((1280/grid_size)*(col)), int((1280/grid_size)*(col+1)))

        pos = u[i]
        #print("pose",pos)
        #if pos[0]<=(1280/grid_size)*(row) and  pos[0]>=(1280/grid_size)*(row-1) and (pos[1]<=(720/grid_size)*(col+1) and  pos[0]>=(720/grid_size)*(col)):
        if pos[1] < (720 / grid_size) * (row) and pos[1] >= (720 / grid_size) * (row - 1) and (
                pos[0] < (1280 / grid_size) * (col + 1) and pos[0] >= (1280 / grid_size) * (col)):
            p_u.append(1)
            Uview.append(frame[int((720/grid_size)*(row-1)):int((720/grid_size)*(row)), int((1280/grid_size)*(col)):int((1280/grid_size)*(col+1)), :])
            #print(index-1,'===============================',pos,row-1,col)
            #print("===================================there is one======================")
        else:
            p_u.append(0)

    #p_u = np.asarray(p_u)
    #p_f = np.asarray(p_f)
    #print("Data",  p_f.shape, "Data", len(p_f), "Data", len(p_u),"Data",p_u.shape)

    return p_u, p_f,Uview




def processed_data200IOUC(grid_size=3, index = 1,u=[] , f=[]):
    #u , f = get_data()
    p_u , p_f = [], []
    Uview=[]

    #print("Data",f[0].shape,"Data",len(f),"Data",len(u))
    #print("Data",u)

    for i in range(len(u)):
        frame = f[i]
        row = math.ceil(index / grid_size)
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",row)
        col = (index - 1) % grid_size

        yp = (720 / grid_size) * (row - 1)
        xp = (1280 / grid_size) * (col)
        yph = (720 / grid_size) * (row)
        xph = (1280 / grid_size) * (col + 1)
        frame_part = frame[int((720/grid_size)*(row-1)):int((720/grid_size)*(row)), int((1280/grid_size)*(col)):int((1280/grid_size)*(col+1)), :]
        frame_part = cv.resize(frame_part, (32, 32))      #120*120
        p_f.append(frame_part)
        # print(int((720/grid_size)*(row-1)), int((720/grid_size)*(row)), int((1280/grid_size)*(col)), int((1280/grid_size)*(col+1)))

        pos = u[i]
        #print("pose",pos)
        #if pos[0]<=(1280/grid_size)*(row) and  pos[0]>=(1280/grid_size)*(row-1) and (pos[1]<=(720/grid_size)*(col+1) and  pos[0]>=(720/grid_size)*(col)):
        x = pos[0]
        y = pos[1]
        A = IfPInUserView(x, y,xp,yp)
        B = IfPInUserView(x, y, xp, yph)
        C = IfPInUserView(x, y, xph, yp)
        D = IfPInUserView(x, y, xph, yph)

        #if pos[1] < (720 / grid_size) * (row) and pos[1] >= (720 / grid_size) * (row - 1) and (
         #       pos[0] < (1280 / grid_size) * (col + 1) and pos[0] >= (1280 / grid_size) * (col)):
        if A==1 or B==1 or C==1 or D==1:
            p_u.append(1)
            Uview.append(frame[int((720/grid_size)*(row-1)):int((720/grid_size)*(row)), int((1280/grid_size)*(col)):int((1280/grid_size)*(col+1)), :])
            #print(index-1,'===============================',pos,row-1,col)
            #print("===================================there is one======================")
        else:
            p_u.append(0)

    #p_u = np.asarray(p_u)
    #p_f = np.asarray(p_f)
    #print("Data",  p_f.shape, "Data", len(p_f), "Data", len(p_u),"Data",p_u.shape)

    return p_u, p_f,Uview

def processed_data328C(grid_size=3, index = 1,u=[] , f=[]):
    #u , f = get_data()
    p_u , p_f = [], []
    Uview=[]

    #print("Data",f[0].shape,"Data",len(f),"Data",len(u))
    #print("Data",u)

    for i in range(len(u)):
        frame = f[i]
        row = math.ceil(index / grid_size)
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",row)
        col = (index - 1) % grid_size

        yp = (720 / grid_size) * (row - 1)+72
        xp = (1280 / grid_size) * (col)+128
        yph = (720 / grid_size) * (row)+72
        xph = (1280 / grid_size) * (col + 1)+128
        frame_part = frame[int((720/grid_size)*(row-1)+72):int((720/grid_size)*(row)+72), int((1280/grid_size)*(col)+128):int((1280/grid_size)*(col+1)+128), :]
        frame_part = cv.resize(frame_part, (32, 32))      #120*120
        p_f.append(frame_part)
        # print(int((720/grid_size)*(row-1)), int((720/grid_size)*(row)), int((1280/grid_size)*(col)), int((1280/grid_size)*(col+1)))

        pos = u[i]
        #print("pose",pos)
        #if pos[0]<=(1280/grid_size)*(row) and  pos[0]>=(1280/grid_size)*(row-1) and (pos[1]<=(720/grid_size)*(col+1) and  pos[0]>=(720/grid_size)*(col)):
        x = pos[0]
        y = pos[1]
        A = IfPInUserView(x, y,xp,yp)
        B = IfPInUserView(x, y, xp, yph)
        C = IfPInUserView(x, y, xph, yp)
        D = IfPInUserView(x, y, xph, yph)

        #if pos[1] < (720 / grid_size) * (row) and pos[1] >= (720 / grid_size) * (row - 1) and (
         #       pos[0] < (1280 / grid_size) * (col + 1) and pos[0] >= (1280 / grid_size) * (col)):
        if A==1 or B==1 or C==1 or D==1:
            p_u.append(1)
            #Uview.append(frame[int((720/grid_size)*(row-1)):int((720/grid_size)*(row)), int((1280/grid_size)*(col)):int((1280/grid_size)*(col+1)), :])
            #print(index-1,'===============================',pos,row-1,col)
            #print("===================================there is one======================")
        else:
            p_u.append(0)


    return p_u, p_f,Uview

def processed_data(grid_size=3, index = 1):
    u , f = get_data()
    p_u , p_f = [], []
    print("Data",f[0].shape,"Data",len(f),"Data",len(u))
    print("Data",u)
    for i in range(len(u)):
        frame = f[i]
        row = math.ceil(index/grid_size)
        col = (index-1)%grid_size
        frame_part = frame[int((720/grid_size)*(row-1)):int((720/grid_size)*(row)), int((1280/grid_size)*(col)):int((1280/grid_size)*(col+1)), :]
        frame_part = cv.resize(frame_part, (120, 120))
        p_f.append(frame_part)
        # print(int((720/grid_size)*(row-1)), int((720/grid_size)*(row)), int((1280/grid_size)*(col)), int((1280/grid_size)*(col+1)))

        pos = u[i]
        if pos[0]<=(1280/grid_size)*(row) and  pos[0]>=(1280/grid_size)*(row-1) and (pos[1]<=(720/grid_size)*(col+1) and  pos[0]>=(720/grid_size)*(col)):
            p_u.append(1)
            print("===================================there is one======================")
        else:
            p_u.append(0)

    p_u = np.asarray(p_u)
    p_f = np.asarray(p_f)
    print("Data",  p_f.shape, "Data", len(p_f), "Data", len(p_u),"Data",p_u.shape)

    return p_u, p_f


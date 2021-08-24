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
    def __init__(self, output_size, SamplIn_U, hidden_dim, n_layers):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(SamplIn_U, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

def estUserTrace(User,Cur):
    DATA_IN_X=[]
    DATA_IN_Y=[]

    for i in range (8):
        DATA_IN_X.append(i)
        DATA_IN_Y.append(i)
    
    for i in range (8):
        DATA_IN_X[i]=User[Cur+i][0]
        DATA_IN_Y[i]=User[Cur+i][1]

    X=[DATA_IN_X,DATA_IN_Y]
    model = SentimentNet( 1, 8, 8, 2)
    lr=0.005
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 2
    counter = 0
    h = model.init_hidden(1)
    for i in range(epochs):
        model.zero_grad()
        output, h = model(X, h)


    loss = criterion(output.squeeze(), labels.float())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()


    output, h = model(X, h)

    return output


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
            #print("index为",index,len(frame_part))
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


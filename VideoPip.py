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
#import VideoPip as data


def LocationCalculate(x, y, z, w):
    X = 2 * x * z + 2 * y * w
    Y = 2 * y * z - 2 * x * w
    Z = 1 - 2 * x * x - 2 * y * y

    a = np.arccos(np.sqrt(X ** 2 + Z ** 2) / np.sqrt(X ** 2 + Y ** 2 + Z ** 2))
    if Y > 0:
        ver = a / np.pi * 180
    else:
        ver = -a / np.pi * 180

    b = np.arccos(X / np.sqrt(X ** 2 + Z ** 2))
    if Z < 0:
        hor = b / np.pi * 180
    else:
        hor = (2. - b / np.pi) * 180

    return (90 - ver) / 180, hor / 360


def userLocal_One(FrameRate, FileNameStart, i, TotalSeconds, FH, FW):
    """
    :param FrameRate: FPS
    :param FileNameStart: The filename will be generated with FileNameStart and the "i"
    :param i: the number of the user
    :return: Userdata
    """

    '''
    # test and collect video info 
    video_name = '/home/kora/Downloads/ECO-efficient-video-understanding-master/scripts/online_recognition/UserData/Video/1-1-Conan Gore FlyB.mp4'
    # video_name = '/home/kora/Downloads/1-7-Cooking Battle.mp4'
    cap = cv2.VideoCapture(video_name)
    FrameRate = int(round(cap.get(5)))      # 29.9 fps changed to 30
    TotalFrames = cap.get(7)
    print ("frame rate is: ",FrameRate,"  Total frames is: ",TotalFrames)
    '''

    '''
        Read user data file and collect all records
        Save them in two lists:
        Userdata and TimeStamp
        Userdata is where store the user location (convert to fload)
        TimeStamp is for syncronization

    '''
    # /home/kora/Downloads/ECO-efficient-video-understanding-master/scripts/online_recognition/UserData/Location
    VideoName = "video_6_D1"
    DirectorName = "/home/kora/Downloads/ECO-efficient-video-understanding-master/scripts/online_recognition/UserData/Location/"
    i = 1
    # UserFile = FileNameStart + str(i) + ".csv"
    # UserFile = DirectorName + VideoName + '_' + str(i) + ".csv"
    UserFile = DirectorName + FileNameStart + '_' + str(i) + ".csv"
    UserFile=FileNameStart
    Userdata = []
    flagTime = 1
    TimeStamp = []
    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到Userdata中
            Userdata.append(row[1:])
            if flagTime == 1:
                TimeStamp.append(row[0])
    flagTime = 0
    Userdata = [[float(x) for x in row] for row in Userdata]  # 将数据从string形式转换为float
    Userdata = np.array(Userdata)  # 将list数组转化成array数组便于查看数据结构
    TOT_Len = len(Userdata)
    print(TOT_Len)
    print(Userdata.shape)
    strItem = TimeStamp[0].split(':')
    PreTime = math.ceil(float(strItem[2]))
    CurTime = math.floor(float(strItem[2]))
    NUmberCount = 0
    UserLocationPerFrame = []
    j = 0  # j for all items in one user
    while NUmberCount < TotalSeconds:
        NUmberCount += 1
        UserIn1s = []
        UserIn1sR = []
        Ind = 0
        UserAll = []
        '''
        通过用户数据中第一列，时间戳信息，获取每一秒内的用户location。该秒内用户数据记录可能大于帧率也可能小于帧率
        '''
        while PreTime > CurTime:
            # print j,NUmberCount
            strA = TimeStamp[j].split(':')
            # print(strA[2])
            # Ind=Ind+1
            # print([Ind,j])
            CurTime = math.floor(float(strA[2]))
            if CurTime == 0 and PreTime == 60:
                break
            x = Userdata[j][1]
            y = Userdata[j][2]
            z = Userdata[j][3]
            w = Userdata[j][4]
            H, W = LocationCalculate(x, y, z, w)
            IH = math.floor(H * FH)
            IW = math.floor(W * FW)
            UserAll.append([IW, IH])
            # print(">>>>>>>>>>>>>>>>     ", H,W, "   <<<<<<<<<<<<<<<<<<<<<<<")
            # print(IW,IH)
            j = j + 1
            UserIn1s.append(UserAll)
        FrameCount = 0
        PreTime = CurTime + 1
        '''
            获得每一秒内用户视角记录后，整理每一帧用户视角位置
        '''
        LengthInOneSec = len(UserAll)
        if LengthInOneSec >= FrameRate:
            IntervalIndex = LengthInOneSec / FrameRate
            for IU in range(FrameRate):
                ModiIndex = int(round(IU * IntervalIndex))
                if ModiIndex>=len(UserAll):
                    print(ModiIndex,len(UserAll),"1")
                UserLocationPerFrame.append(UserAll[ModiIndex])
        else:
            IntervalIndex = LengthInOneSec / FrameRate
            for IU in range(FrameRate):
                ModiIndex = int(round(IU * IntervalIndex))
                if ModiIndex>=len(UserAll):
                    print(ModiIndex,len(UserAll),"2")
                    ModiIndex = len(UserAll)-1
                UserLocationPerFrame.append(UserAll[ModiIndex])
    return UserLocationPerFrame


def LocationCalculateB(x, y, z, w):
    X = 2 * x * z + 2 * y * w
    Y = 2 * y * z - 2 * x * w
    Z = 1 - 2 * x * x - 2 * y * y

    a = np.arccos(np.sqrt(X ** 2 + Z ** 2) / np.sqrt(X ** 2 + Y ** 2 + Z ** 2))
    if Y > 0:
        ver = a / np.pi * 180
    else:
        ver = -a / np.pi * 180

    b = np.arccos(X / np.sqrt(X ** 2 + Z ** 2))
    if Z < 0:
        hor = b / np.pi * 180
    else:
        hor = (2. - b / np.pi) * 180

    return (90 - ver) / 180, hor / 360


# "1-7-Cooking BattleB","1-2-FrontB","1-6-FallujaB","1-8-FootballB","1-9-RhinosB",
    #"1-7-Cooking BattleB","1-2-FrontB","1-6-FallujaB","1-8-FootballB","1-9-RhinosB",
    #
# 1-2-FrontB 1-5-Tahiti SurfB 1-9-RhinosB
FileList = [ "1-7-Cooking BattleB","1-2-FrontB","1-6-FallujaB","1-8-FootballB","1-9-RhinosB","2-1-KoreanB", "2-3-RioVRB","2-4-FemaleBasketballB", "2-5-FightingB", "2-6-AnittaB"]
K_V=7   #视频号码
FL=FileList[K_V]


VideoName = FL  # 1-2-FrontB  1-1-Conan Gore FlyB  1-9-RhinosB
videofile = '../../DataProcess/Anittia/'+VideoName + ".mp4"  # 2-4-FemaleBasketballB 2-6-AnittaB 1-6-FallujaB 2-3-RioVRB  2-5-FightingB  2-8-reloadedB
tmp1=FL[0]
tmp2=int(FL[2])-1
if tmp1 =='1':
    VideoUserName = "video_"+str(tmp2)+"_D1_"
else:
    VideoUserName = "video_" + str(tmp2) + "_"

cap = cv.VideoCapture(videofile)
'''
cap = cv.VideoCapture('1-7-Cooking BattleB.mp4')  # 2-5-FightingB  Asave2-6-AnittaB.avi 2-8-reloadedB

VideoUserName = "video_6_D1_"
'''
ProceList = []
TimeStamp = []
flagTime = 1



'''
读取用户历史数据 48 名用户  A设定用户数量 A=49来加入48 用户
'''
K=2             #用户NO
A = 2  # 47                    #49
for i in range(1, A + 1):  # 49
    #UserFile = VideoUserName + str(i) + ".csv"
    UserFile = '../../DataProcess/Anittia/'+VideoUserName + str(K) + ".csv"
    Userdata = []
    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到Userdata中
            Userdata.append(row[1:])
            if flagTime == 1:
                TimeStamp.append(row[0])
    flagTime = 0
    Userdata = [[float(x) for x in row] for row in Userdata]  # 将数据从string形式转换为float
    Userdata = np.array(Userdata)  # 将list数组转化成array数组便于查看数据结构
    ProceList.append(Userdata)

# print(len(ProceList))
# print(len(ProceList[0]))
# print(len(ProceList[0][0]))
# k=0
# for i in TimeStamp:
#    k=k+1
#    print(i,k)

def get_data():
    '''
    分析用户数据
    '''
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



        #读取一秒内用户数据，一秒内用户数据多于每秒帧数，将用户视角映射到每一帧
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

            # 加入1s内用户抽样，原用户并不是一秒30个
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
            print("我次奥，出现负值","帧编号为：",frame_number)
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
    print("数据格式为",f[0].shape,"视频图片长度为：",len(f),"用户数据长度为：",len(u))
    print("用户数据为：",u)
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
    print("整理后数据格式",  p_f.shape, "整理后图片数目为：", len(p_f), "用户数据长度为：", len(p_u),"用户数据格式为：",p_u.shape)

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

    #print("数据格式为",f[0].shape,"视频图片长度为：",len(f),"用户数据长度为：",len(u))
    #print("用户数据为：",u)
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
    #print("整理后数据格式",  p_f.shape, "整理后图片数目为：", len(p_f), "用户数据长度为：", len(p_u),"用户数据格式为：",p_u.shape)

    return p_u, p_f,Uview




def processed_data200IOUC(grid_size=3, index = 1,u=[] , f=[]):
    #u , f = get_data()
    p_u , p_f = [], []
    Uview=[]

    #print("数据格式为",f[0].shape,"视频图片长度为：",len(f),"用户数据长度为：",len(u))
    #print("用户数据为：",u)

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
    #print("整理后数据格式",  p_f.shape, "整理后图片数目为：", len(p_f), "用户数据长度为：", len(p_u),"用户数据格式为：",p_u.shape)

    return p_u, p_f,Uview

def processed_data328C(grid_size=3, index = 1,u=[] , f=[]):
    #u , f = get_data()
    p_u , p_f = [], []
    Uview=[]

    #print("数据格式为",f[0].shape,"视频图片长度为：",len(f),"用户数据长度为：",len(u))
    #print("用户数据为：",u)

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
    print("数据格式为",f[0].shape,"视频图片长度为：",len(f),"用户数据长度为：",len(u))
    print("用户数据为：",u)
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
    print("整理后数据格式",  p_f.shape, "整理后图片数目为：", len(p_f), "用户数据长度为：", len(p_u),"用户数据格式为：",p_u.shape)

    return p_u, p_f


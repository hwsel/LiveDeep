import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import cv2 as cv
import numpy as np
import logging
import argparse
import math
from time import time
import csv

# import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np

import argparse
import cv2
from models import *
from misc import progress_bar

from MyOwnModel import LeNet
import VideoPip as data

# learning_rate = 1e-3
learning_rate = 0.001
epoches = 1  # 50

transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)


def CalculateUserView(x, y):
    #1280 720
    #x=700
    #y=360
    W=180   #140
    H=180   #130
    UWL=x-W
    UHL=y-H-10
    UWH=x+W
    UHH=y+H-20

    if UWL<0:
        UWL=0
    if UHL<0:
        UHL=0
    if UWH>1280:
        UWH=1280
    if UHH>720:
        UHH=720


    return UWL, UHL ,UWH, UHH

def CheckPredictionResult(TileStaForCheck,UWL, UHL ,UWH, UHH):
    iL=int(math.floor(UWL/256))
    jL=int(math.floor(UHL/144))
    iH = int(math.floor(UWH / 256))
    jH = int(math.floor(UHH / 144))
    Flag=0
    if iH>=5:
        iH=4
    if jH>=5:
        jH=4
    #print(jH*5+iH,jH,iH,len(TileStaForCheck))
    A=TileStaForCheck[jL*5+iL]

    B = TileStaForCheck[jL * 5 + iH]
    C = TileStaForCheck[jH * 5 + iL]
    D = TileStaForCheck[jH * 5 + iH]
    T=A+B+C+D
    if A*B*C*D ==1:
        Flag=1
    return Flag, T

def CheckPredictionResultUF(TileStaForCheck,TileStaByUF,UWL, UHL ,UWH, UHH):
    iL = int(math.floor(UWL / 256))
    jL = int(math.floor(UHL / 144))
    iH = int(math.floor(UWH / 256))
    jH = int(math.floor(UHH / 144))
    Flag = 0
    if iH >= 5:
        iH = 4
    if jH >= 5:
        jH = 4
    # print(jH*5+iH,jH,iH,len(TileStaForCheck))
    A = TileStaByUF[jL * 5 + iL]

    B = TileStaByUF[jL * 5 + iH]
    C = TileStaByUF[jH * 5 + iL]
    D = TileStaByUF[jH * 5 + iH]
    T = A + B + C + D
    if A * B * C * D == 1:
        Flag = 1
    CountModify=0
    for i in range(len(TileStaForCheck)):
        if TileStaForCheck[i] == 0 and TileStaByUF[i]==1:
            TileStaForCheck[i]=1
            CountModify+=1
    return Flag, CountModify

def UpdateTileStatUF(TileStaByUF,UWL, UHL ,UWH, UHH):
    for i in range(len(TileStaByUF)):
        TileStaByUF[i]=0
    iL = int(math.floor(UWL / 256))
    jL = int(math.floor(UHL / 144))
    iH = int(math.floor(UWH / 256))
    jH = int(math.floor(UHH / 144))
    Flag = 0
    if iH >= 5:
        iH = 4
    if jH >= 5:
        jH = 4
    for i in range(iL,iH+1):
        for j in range(jL,jH+1):
            TileStaByUF[j * 5 + i]=1

def TensorImageToPILandCV(output,data):
    to_pil_image=transforms.ToPILImage()
    print(len(output), output.shape)
    img = data[0].cpu()
    #CVView=img.numpy()
    #cv2.imshow('TorchImgB', CVView)
    imge = to_pil_image(img)
    imge.show()
    imgeCV = img.numpy() * 255
    img_1 = imgeCV.astype('uint8')
    img_1 = np.transpose(img_1, (1, 2, 0))
    cv2.imshow('TorchImg', img_1)
    cv2.waitKey()
    print(img.shape, len(img))
    exit()
    img = data[0].mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    cv2.imshow('TorchImg', img)
    # imge=
    exit()

def ImageofCVToTensor(imgdata):
    img1=imgdata[0]
    imgT1=transform1(img1)
    img2 = imgdata[1]
    imgT2 = transform1(img2)
    TenTemp1=torch.stack((imgT1,imgT2),0)
    for i in range(2,len(imgdata),2):
        img1 = imgdata[i]
        imgT1 = transform1(img1)
        img2 = imgdata[i+1]
        imgT2 = transform1(img2)
        TenTemp2 = torch.stack((imgT1, imgT2), 0)
        TenTemp1=torch.cat((TenTemp1,TenTemp2),0)
    return TenTemp1
def ImageofCVToTensorSingle(imgdata):
    print(len(imgdata))
    exit()
    img=transform1(imgdata)
    return img
def ImageofPILToTensor(imgdata):
    b=0

class main():
    def __init__(self, grid_size=3, index=1):
        self.g = grid_size
        self.idx = index
        self.model = LeNet(grid_size=self.g)
        self.criterian = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self, u, f):
        # u, f = data.processed_data(self.g, self.idx)
        # print(u)
        # exit()
        for i in range(epoches):
            running_loss = 0
            # for j in range(len(u)):
            user = torch.from_numpy(np.array(u))
            user = user.long()

            frame = torch.from_numpy(f)
            output = self.model(frame)
            # print (len(output))
            # print(output)
            # print(user)
            self.optimizer.zero_grad()
            loss = self.criterian(output, user)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            print("epoch: {0:3d}, loss: {1:f}".format(i, loss))
        return output


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size,
                                                        shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        #self.model = AlexNet().to(self.device)
        #self.model = LeNet().to(self.device)
        #self.model = AlexNet().to(self.device)
        #self.model = VGG11().to(self.device)
        self.model = VGG13().to(self.device)
        #self.model = VGG16().to(self.device)
        #self.model = VGG19().to(self.device)
        #self.model = GoogLeNet().to(self.device)
        #self.model = resnet18().to(self.device)
        # self.model = resnet34().to(self.device)
        # self.model = resnet50().to(self.device)
        # self.model = resnet101().to(self.device)
        #self.model = resnet152().to(self.device)
        #self.model = DenseNet121().to(self.device)
        # self.model = DenseNet161().to(self.device)
        # self.model = DenseNet169().to(self.device)
        # self.model = DenseNet201().to(self.device)
        #self.model = WideResNet(depth=28, num_classes=2).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        #self.criterion= nn.


    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            # print("target",target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self,IndeV,IndeU,Tk,EpoMax):
        #self.load_data()
        self.load_model()
        '''
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/200" % epoch)
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()
        '''
        Tile_Status = []
        '''
            video infomation
            '''
        FileList = ["1-7-Cooking BattleB", "2-3-RioVRB", "2-4-FemaleBasketballB", "2-5-FightingB", "2-6-AnittaB",
                    "1-1-Conan Gore FlyB","1-5-TahitiSurfB","2-8-reloadedB","2-2-VoiceToyB",
                    "1-2-FrontB", "1-6-FallujaB", "1-8-FootballB", "1-9-RhinosB", "2-1-KoreanB"]  # 14 个

        K_V = IndeV  # 
        FL = FileList[K_V]

        VideoName = FL  # 1-2-FrontB  1-1-Conan Gore FlyB  1-9-RhinosB
        videofile = '../../DataProcess/Anittia/' + VideoName + ".mp4"  # 2-4-FemaleBasketballB 2-6-AnittaB 1-6-FallujaB 2-3-RioVRB  2-5-FightingB  2-8-reloadedB
        tmp1 = FL[0]
        tmp2 = int(FL[2]) - 1
        if tmp1 == '1':
            VideoUserName = "video_" + str(tmp2) + "_D1_"
        else:
            VideoUserName = "video_" + str(tmp2) + "_"

        cap = cv.VideoCapture(videofile)
        capB = cv.VideoCapture(videofile)
        '''
        用户数据
        '''
        UserID = IndeU
        UserFile = '../../DataProcess/Anittia/' + VideoUserName + str(UserID) + ".csv"
        UserDataCSV = UserFile
        '''
        视频信息
        '''
        W_Frame = cap.get(3)
        H_Frame = cap.get(4)
        # check the parameters of get frome:  https://blog.csdn.net/qhd1994/article/details/80238707
        print("===============frame width============")
        print(W_Frame)
        print("===============frame height============")
        print(H_Frame)
        FrameRate = int(round(cap.get(5)))  # 29.9 fps changed to 30
        SubSampleRate = 4  # 4 frames per second
        # SubSampleStep = int(math.floor(FrameRate / SubSampleRate))
        SubSampleStep = int(math.ceil(FrameRate / SubSampleRate))
        TotalFrames = cap.get(7)
        TotalSeconds = int(round(TotalFrames / FrameRate))
        print("framerate and totalframes is:  ", FrameRate, TotalFrames)
        print("total second is: ", TotalSeconds)

        LocationPerFrame = data.userLocal_One(FrameRate, UserDataCSV, 1, TotalSeconds, H_Frame, W_Frame)
        print("total frame from user data:", len(LocationPerFrame))

        '''
                some parameters
                '''
        TileNO = 5
        bufInSen = 2
        bufLen = FrameRate * bufInSen
        for i in range(TileNO * TileNO):
            Tile_Status.append(0)

        """
        Data operation
        """
        CSVfilename='00A'+FL+'_'+str(IndeU)+'Tk_EpoMax'+str(Tk)+'_'+str(EpoMax)+'_AccAndBandC.csv'
        fileobj = open(CSVfilename, 'w', newline='')  # 
        CSVfilenameTime ='00A'+ FL +'_'+str(IndeU)+'Tk_EpoMax'+str(Tk)+'_'+str(EpoMax)+'_TimeConsumptionC.csv'
        fileobjT = open(CSVfilenameTime, 'w', newline='')  # 
        # fileobj.write('\xEF\xBB\xBF')#
        writer = csv.writer(fileobj)  # csv.writer(fileobj)
        writerT = csv.writer(fileobjT)  # csv.writer(fileobj)

        #head
        sortedValues = ['VideoName', 'UserIndex', 'tile total1/2 or + 1/8', 'Max epoch ']
        writer.writerow(sortedValues)
        writerT.writerow(sortedValues)

        sortedValues=[FL,str(IndeU),str(Tk),str(EpoMax)]
        writer.writerow(sortedValues)
        writerT.writerow(sortedValues)
        TotalSize='Size/'+str(TileNO*TileNO)
        sortedValues = ['Predicted accurate', TotalSize,'countMatched','ACC Feedback','ACC both','NewBand']
        # writerow()
        # writerows
        writer.writerow(sortedValues)
        ValueTime=['running time','epoches','FirstLoss','FinalLoss','CountMatch']
        writerT.writerow(ValueTime)
        TmpName='DEMO_0A'+FL+'_'+str(IndeU)+'Tk_EpoMax'+str(Tk)+'_'+str(EpoMax)+'_B.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(TmpName, fourcc, FrameRate, (int(W_Frame), int(H_Frame)))

        #exit()



        '''

        parser = argparse.ArgumentParser()
        parser.add_argument('--grid', type=int, help='set the grid size', default=5)
        parser.add_argument('--index', type=int, help='the index of the box', default=1)
        args = parser.parse_args()
        '''

        # M = main(args.grid, args.index)
        # M = main(TileNO, args.index)

        #：
        countMain = 0  #
        countShow = 0
        point_size = 100
        point_color = (0, 0, 255)  # BGR
        thickness = 8  #  0 、4、8
        ColorPrediction = (255, 0, 0)
        FirstLoss=0
        FinalLoss=0
        EpocNO=0
        EpichoSDy=0
        CountMatch=0
        BreakFlag=0
        TotalT=TileNO*TileNO
        TileStaByUF = [] #tile status by user feedback
        for k in range(TotalT):
            TileStaByUF.append(0)
        while (True):
            P_f = []
            p_u = []
            # 
            startT=time()
            CountError = 0
            if countMain + bufLen < TotalFrames:  # 
                for i in range(bufLen):
                    ret, frame = cap.read()
                    '''
                    if ret!=True:
                        BreakFlag=1
                        break
                    '''
                    # 
                    # if i%SubSampleStep ==0 and i!=0:        #
                    if i % SubSampleStep == 0:  # 
                        P_f.append(frame)  # 
                        p_u.append(LocationPerFrame[countMain])  # 
                        #print(p_u)
                        CountError += 1
                        # print(i)
                    countMain = countMain + 1
            # u, f = data.processed_dataB(TileNO, TileNO,p_u,P_f)
            # M.train(u, f)
            else:
                break
            '''
            
            5*5*8:processed_dataC
            '''
            u, f ,v= data.processed_data200IOUC(TileNO, 1, p_u, P_f)  # 
            for index in range(TileNO * TileNO - 1):  # 
                au, af ,av= data.processed_data200IOUC(TileNO, index + 2, p_u, P_f)
                #u = np.append(u, au)
                u.extend(au)
                f.extend(af)
                v.extend(av)
                #f = np.append(f, af)
            # print(len(u))
            # print(CountError,SubSampleStep,bufLen,"FrameRate：",FrameRate)
            # exit()
            #  prediction 
            #cv2.imshow('test',f[0])
            #cv2.waitKey()
            #print(len(u),u.count(1))
            #print(p_u,u)
            #exit()
            '''
            if u.count(1)!=8:
                print(u)
                exit()
            '''
            TenFram=ImageofCVToTensor(f)
            TenFram=TenFram.to(self.device)
            user = torch.from_numpy(np.array(u))
            target = user.long().to(self.device)

            #epochesNew=3+EpichoSDy
            epochesNew = EpoMax
            Times=0
            loss=1
            #print(countMain,Times,loss)
            #while Times<=epochesNew:
            for Times in range(epochesNew):
                #print(countMain,"in")
                train_loss = 0
                train_correct = 0
                total = 0

                self.optimizer.zero_grad()
                output = self.model(TenFram)
                if Times ==0:               # 0 not 1
                    Prediction=output


                #print(output)
                #print(target)
                #target=torch.Tensor(u).to(self.device)
                #target.target.float()
                #print(type(target))
                #exit()
                loss = self.criterion(output, target)
                if Times == 1:
                    FirstLoss=loss
                print(loss)
                if loss < 0.2:
                    break
                #print(loss)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
                total += target.size(0)

                # train_correct incremented by one if predicted right
                train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                batch_num=200
                if loss < 0.4:
                    self.lr = 0.001


            FinalLoss=loss
            stopT = time()
            ToTalTime=stopT-startT
            print("==========",ToTalTime)
            writerT.writerow([str(ToTalTime),str(int(epochesNew)),str(float(FirstLoss)),str(float(FinalLoss)),str(CountMatch),str(Times)])




            #print(Prediction)
            #exit()
            #Prediction = M.train(u, f)
            # print(Prediction)
            # exit()
            # exit()

            '''
            New updated result output
            '''
            TotalT = TileNO * TileNO
            # print(len(Prediction))
            # exit()
            PredictionB = []
            UB=[]
            for i in range(8):
                for j in range(TotalT):
                    #print(countMain)
                    PredictionB.append(Prediction[j * 8 + i])
                    UB.append(u[j * 8 + i])
            #print(u)
            #print(UB)
            #print(len(UB))
            #exit()

            '''
            display the prediction result in one buffer
            '''

            Tile_Status = []
            Tile_Status0=[]
            Tile_Statusu=[]
            #Uframe=v[0]
            counV=0
            for i in range(TotalT):
                Tile_Status.append(PredictionB[i])
                Tile_Status0.append(PredictionB[i][0])
                Tile_Statusu.append(UB[0])

            SortTile_Status0 = Tile_Status0.copy()
            SortTile_Status0.sort()
            Thres = SortTile_Status0[int(TotalT / 2+(TotalT / 8)*Tk)]
            Step = 0
            CountFlag = 0
            UWL = 0
            UHL = 0
            UWH = 0
            UHH = 0
            for i in range(bufLen):
                ret, frameB = capB.read()
                x = int(LocationPerFrame[countShow][0])
                y = int(LocationPerFrame[countShow][1])
                # if i % SubSampleStep == 0 and i != 0:
                if i % SubSampleStep == 0 and i != 0:
                    counV+=1
                    #Uframe=v[counV]
                    CountFlag += 1
                    Step += TotalT
                    #print("===========")
                    Tile_Status = []
                    Tile_Status0 = []
                    Tile_Statusu = []
                    for k in range(TotalT):
                        if k + Step >= 200:
                            print("if k + Step >= 200:")
                            print(i, k, Step, TotalT, CountFlag, SubSampleStep, bufLen, CountError)
                            exit()
                        Tile_Status.append(PredictionB[k + Step])
                        Tile_Status0.append(PredictionB[k + Step][0])
                        Tile_Statusu.append(UB[k + Step])
                    SortTile_Status0=Tile_Status0.copy()
                    SortTile_Status0.sort()
                    Thres=SortTile_Status0[int(TotalT/2+(TotalT / 8)*Tk)]       #best 4


                #cv.circle(frameB, (x, y), point_size, point_color, thickness)
                UWL, UHL ,UWH, UHH =CalculateUserView(x, y)
                cv.rectangle(frameB, (UWL, UHL), (UWH, UHH), (255,255,0), 2)
                UWL, UHL, UWH, UHH = CalculateUserView(x, y)
                #print(Tile_Status)
                AccFlag=0
                CountTile=0
                EXT=0  #72
                TileStaForCheck=[]
                for k in range(TotalT):
                    #cv2.rectangle(frameB, (j * 256, i * 144), ((j + 1) * 256, (i + 1) * 144), ColorPrediction, 2)
                    # if Tile_Status[k][0]-Tile_Status[k][1]>0.4:
                    #if Tile_Status[k][0] > Tile_Status[k][1
                    if Tile_Status0[k] < Thres:
                        TileStaForCheck.append(1)
                        CountTile+=1
                    #if UB[k] ==1:
                        row = math.floor(k / TileNO)
                        col = (k) % TileNO
                        #WL=int(col * 256-128)
                        WL = int(col * 256)
                        HL=int(row * 144-EXT)
                        #WH=int((col + 1) * 256+128)
                        WH = int((col + 1) * 256)
                        HH=int((row + 1) * 144+EXT)
                        if WL<0:
                            WL=0
                        if HL<0:
                            HL=0
                        if WH>W_Frame:
                            WH=W_Frame
                        if HH>H_Frame:
                            HH=H_Frame
                        WL=int(WL)
                        HL=int(HL)
                        WH=int(WH)
                        HH=int(HH)

                        #cv.rectangle(frameB, (WL, HL),(WH, HH),ColorPrediction, 2)


                        cv.rectangle(frameB, (int(col * 256), int(row * 144)),(int((col + 1) * 256), int((row + 1) * 144)),ColorPrediction, 2)

                        '''
                        if row==2:
                            cv.rectangle(frameB, (int(col * 256), int((row+1) * 144)),
                                         (int((col + 1) * 256), int((row + 2) * 144)),
                                         ColorPrediction, 2)
                        if row ==3:
                            cv.rectangle(frameB, (int(col * 256), int(2 * 144)),
                                         (int((col + 1) * 256), int((2 + 1) * 144)),
                                         ColorPrediction, 2)
                        #cv2.rectangle(frameB,(WL,HL), (WH,HH), ColorPrediction, 2)
                        '''
                        if UB[k] == 1:
                            AccFlag=1
                    else:
                        TileStaForCheck.append(0)
                ''''''
                for k in range(TotalT):
                    #cv2.rectangle(frameB, (j * 256, i * 144), ((j + 1) * 256, (i + 1) * 144), ColorPrediction, 2)
                    # if Tile_Status[k][0]-Tile_Status[k][1]>0.4:
                    #if Tile_Status[k][0] > Tile_Status[k][1
                    if TileStaByUF[k] ==1:

                        row = math.floor(k / TileNO)
                        col = (k) % TileNO
                        #WL=int(col * 256-128)
                        WL = int(col * 256)
                        HL=int(row * 144-EXT)
                        #WH=int((col + 1) * 256+128)
                        WH = int((col + 1) * 256)
                        HH=int((row + 1) * 144+EXT)
                        if WL<0:
                            WL=0
                        if HL<0:
                            HL=0
                        if WH>W_Frame:
                            WH=W_Frame
                        if HH>H_Frame:
                            HH=H_Frame
                        WL=int(WL)
                        HL=int(HL)
                        WH=int(WH)
                        HH=int(HH)

                        cv.rectangle(frameB, (WL, HL),(WH, HH),ColorPrediction, 2)
                ''''''
                AccNew,CountMatch=CheckPredictionResult(TileStaForCheck,UWL, UHL ,UWH, UHH)
                ACCUF,NewTileTotal=CheckPredictionResultUF(TileStaForCheck,TileStaByUF,UWL, UHL ,UWH, UHH)
                NewTileTotal=NewTileTotal+CountTile
                AccNewAnUF, CountMatchB = CheckPredictionResult(TileStaForCheck, UWL, UHL, UWH, UHH)
                EpichoSDy=4-CountMatch
                if EpichoSDy==1:
                    self.lr=0.003
                if EpichoSDy==2:
                    self.lr=0.006
                if EpichoSDy==3:
                    self.lr=0.008
                if EpichoSDy==4:
                    self.lr=0.01
                #sortedValues = [str(AccFlag), str(CountTile)]
                sortedValues = [str(AccNew), str(CountTile),str(CountMatch),str(ACCUF),str(AccNewAnUF),str(NewTileTotal)]
                writer.writerow(sortedValues)
                cv.imshow('Frames', frameB)
                #cv.imshow('user view',Uframe)
                out.write(frameB)
                countShow += 1
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            UpdateTileStatUF(TileStaByUF,UWL, UHL ,UWH, UHH)

        cap.release()
        capB.release()
        cv.destroyAllWindows()


if __name__ == '__main__':  # run python main.py --grid 5 --index 1

    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')  #0.001
    parser.add_argument('--epoch', default=20, type=int, help='number of epochs tp train for')  # epoch =200
    #parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--trainBatchSize', default=200, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    #TKL=[0,1]
    TKL = [ 1]
    #EpomaxL=[8,10,12,14]
    EpomaxL = [10]
    #EpomaxL = [ 6]
    for j in range(1, 2):#49
        for i in range(14):#9
            for TK in TKL:
                for Epomax in EpomaxL:
                    solver = Solver(args)
                    solver.run(i,j,TK,Epomax)



# u, f = data.processed_data(3, 4)
# print(u[0], f[0].shape)
# cv.imshow('frame', f[3])
# cv.waitKey(0)
# x = f[0]
# x = torch.from_numpy(x)
# m = LeNet()
# print(m(x))
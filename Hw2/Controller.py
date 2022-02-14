import sys
import cv2
import numpy
import glob
import time
import random
import tensorflow

from HW2 import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from scipy import signal
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

class Controller(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.draw_contour)
        self.pushButton_2.clicked.connect(self.count_coins)
        self.pushButton_3.clicked.connect(self.find_corners)
        self.pushButton_4.clicked.connect(self.find_intrinsic)
        self.pushButton_5.clicked.connect(self.find_extrinsic)
        self.pushButton_6.clicked.connect(self.find_distortion)
        self.pushButton_7.clicked.connect(self.argmented_reality)
        self.pushButton_8.clicked.connect(self.stereo_disparity_map)
        self.pushButton_9.clicked.connect(self.show_tensorboard)
        self.pushButton_10.clicked.connect(self.test)
        self.pushButton_11.clicked.connect(self.random_erasing)
        self.show()

        self.ret = 0
        self.mtx = 0
        self.dist = 0
        self.rvecs = 0
        self.tvecs = 0
        self.disparity = 0
        self.disparity1 = 0
        self.contours1 = 0
        self.contours2 = 0
         
    @pyqtSlot()
    def draw_contour(self):
        img1 = cv2.imread('./Datasets/Q1_Image/coin01.jpg')
        img2 = cv2.imread('./Datasets/Q1_Image/coin02.jpg')
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        blurred1 = cv2.GaussianBlur(gray1, (11, 11), 0)
        blurred2 = cv2.GaussianBlur(gray2, (11, 11), 0)

        canny1 = cv2.Canny(blurred1, 50, 150)
        canny2 = cv2.Canny(blurred2, 50, 150)

        self.contours1, hierarchy1 = cv2.findContours(canny1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours2, hierarchy2 = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img1, self.contours1, -1, (0, 255, 0), 2)
        cv2.drawContours(img2, self.contours2, -1, (0, 255, 0), 2)

        cv2.imshow('coin01', img1)
        cv2.imshow('coin02', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def count_coins(self):
        self.label.setText("There are %d coins in coin01.jpg" % len(self.contours1))
        self.label_2.setText("There are %d coins in coin01.jpg" % len(self.contours2))

    def find_corners(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = numpy.zeros((11 * 8, 3), numpy.float32)
        objp[:, : 2] = numpy.mgrid[0: 11, 0: 8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane
         
        images = glob.glob('./Datasets/Q2_Image/*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(img, (11, 8), corners2, ret)

                cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
                cv2.imshow('my_image', img)
                cv2.waitKey(0)

        cv2.destroyAllWindows()

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
         
    def find_intrinsic(self):
        print(self.mtx)

    def find_extrinsic(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = numpy.zeros((11 * 8, 3), numpy.float32)
        objp[:, : 2] = numpy.mgrid[0: 11, 0: 8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane
         
        text = self.comboBox.currentText()
        img = cv2.imread('./Datasets/Q2_Image/'+text+'.bmp')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        objpoints.append(objp)
        imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
         
        rotation = rvecs.pop()
        transition = tvecs.pop()
        m_rotation , jacobin = cv2.Rodrigues(rotation)
        extrinsic = numpy.hstack((m_rotation,transition))
        print(extrinsic)
         
    def find_distortion(self):
        print(self.dist)
    
    def argmented_reality(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = numpy.zeros((11 * 8, 3), numpy.float32)
        objp[:, : 2] = numpy.mgrid[0: 11, 0: 8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane
        axis = numpy.float32([[3,3,-3],[1,1,0],[3,5,0],[5,1,0]])
         
        images = glob.glob('./Datasets/Q3_Image/*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

                objpoints.pop()
                imgpoints.pop()
                rotation = rvecs.pop()
                transition = tvecs.pop()
                   
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rotation, transition, mtx, dist)

                img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 4)
                img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 4)
                img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 4)
                img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 4)
                img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 4)
                img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 4)

                cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
                cv2.imshow('my_image', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()
    def on_EVENT_LBUTTONDOWN(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            
            img_block = numpy.zeros((200, 500))
            self.disparity1[1700:1900, 2000:2500] = img_block

            depth = 2826 * 178 / (self.disparity[y][x] + 123)

            #cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(self.disparity1, 'Disparity: %d pixels' % self.disparity[y][x], (2000, 1800), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(self.disparity1, 'Depth: %d mm' % depth, (2000, 1860), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow('my_image', self.disparity1)
            cv2.waitKey(0)

            cv2.destroyAllWindows()
    
    def stereo_disparity_map(self):
        imgL = cv2.imread('./Datasets/Q4_Image/imgL.png',0)
        imgR = cv2.imread('./Datasets/Q4_Image/imgR.png',0)

        cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
        
        stereo = cv2.StereoBM_create(256, 25)
        self.disparity = stereo.compute(imgL,imgR)
        self.disparity = self.disparity / 16
        self.disparity1 = cv2.convertScaleAbs(self.disparity)

        cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
        cv2.imshow('my_image', self.disparity1)
        #cv2.waitKey(0)
        cv2.setMouseCallback("my_image", self.on_EVENT_LBUTTONDOWN)

    def show_tensorboard(self):
        accuracy = cv2.imread('./Datasets/Q5_Image/epoch_accuracy.png')
        loss = cv2.imread('./Datasets/Q5_Image/epoch_loss.png')
        result = cv2.imread('./Datasets/Q5_Image/result.png')

        cv2.namedWindow('accuracy', cv2.WINDOW_NORMAL)
        cv2.namedWindow('loss', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('accuracy',accuracy)
        cv2.imshow('loss',loss)
        cv2.imshow('result',result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def test(self):
        model = tensorflow.keras.models.load_model("my_model.h5")

        # for i in range(100):

        index = random.randint(1,9999)

        img = cv2.imread('./Datasets/Q5_Image/test1/%d.jpg' %index)

        new_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

        new_img1 = numpy.reshape(new_img, (1, 224, 224, 3))

        ans = model.predict(new_img1)

        img_block = numpy.zeros((20, 224, 3))
        new_img[185:205, 0:224] = img_block

        if(ans[0][0] > ans[0][1]):
            cv2.putText(new_img, 'Class : cat', (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else :
            cv2.putText(new_img, 'Class : dog', (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('result',new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def random_erasing(self):
        label = ["Before Random-Erasing","After Random-Erasing"]

        result = cv2.imread('./Datasets/Q5_Image/result.png')
        result_erasing = cv2.imread('./Datasets/Q5_Image/result1.png')

        # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('result_erasing', cv2.WINDOW_NORMAL)

        cv2.imshow('result',result)
        cv2.imshow('result_erasing',result_erasing)

        ans = [0.94,0.98]

        fig = plt.figure()
        plt.xlabel("methods")
        plt.ylabel("accuracy")
        plt.xticks(range(10))
        plt.bar(label,ans)
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
         
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Controller()
    sys.exit(app.exec_())



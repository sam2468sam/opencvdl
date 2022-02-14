import sys
import cv2
import numpy

from HW1_1_UI import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot

class Controller(QMainWindow, Ui_Form):
     def __init__(self, parent=None):
         super(QMainWindow, self).__init__(parent)
         self.setupUi(self)
         self.pushButton.clicked.connect(self.load)
         self.pushButton_2.clicked.connect(self.color)
         self.pushButton_3.clicked.connect(self.flip)
         self.pushButton_4.clicked.connect(self.blend)
         self.show()
         self.b = 0
         
     @pyqtSlot()
     def load(self):
         img = cv2.imread('Uncle_Roger.jpg')
         cv2.namedWindow('my_image')
         cv2.imshow('my_image',img)
         height = img.shape[0]
         width = img.shape[1]
         print('Height = ',height)
         print('Width = ',width)
         cv2.waitKey(0)
         cv2.destroyWindow('my_image')
     def color(self):
         img = cv2.imread('Flower.jpg')
         cv2.namedWindow('my_image')
         cv2.imshow('my_image',img)
         (B,G,R) = cv2.split(img)
         zeros = numpy.zeros(img.shape[:2],dtype="uint8")
         cv2.imshow("Red",cv2.merge([zeros,zeros,R]))
         cv2.imshow("Green",cv2.merge([zeros,G,zeros]))
         cv2.imshow("Blue",cv2.merge([B,zeros,zeros]))
         cv2.waitKey(0)
         cv2.destroyWindow('my_image')
         cv2.destroyWindow('Red')
         cv2.destroyWindow('Green')
         cv2.destroyWindow('Blue')
     def flip(self):
         img = cv2.imread('Uncle_Roger.jpg')
         cv2.namedWindow('my_image')
         cv2.imshow('my_image',img)
         cv2.imshow('Flipping',cv2.flip(img,1))
         cv2.waitKey(0)
         cv2.destroyWindow('my_image')
         cv2.destroyWindow('Flipping')
     def do(self,x):
         self.b = cv2.getTrackbarPos('BLEND','my_image')
         img1 = cv2.imread('Uncle_Roger.jpg')
         img2 = cv2.flip(img1,1)
         img = cv2.addWeighted(img1, self.b/255, img2, (255-self.b)/255, 0)
         cv2.imshow('my_image',img)
         cv2.waitKey(0)
         cv2.destroyWindow('my_image')
     def blend(self):
         cv2.namedWindow('my_image')
         cv2.createTrackbar('BLEND','my_image',0,255,self.do)
         cv2.setTrackbarPos('BLEND','my_image',1)
         
if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())



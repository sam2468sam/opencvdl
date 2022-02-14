import sys
import cv2
import numpy

from HW1_2_UI import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot

class Controller(QMainWindow, Ui_Form):
     def __init__(self, parent=None):
         super(QMainWindow, self).__init__(parent)
         self.setupUi(self)
         self.pushButton.clicked.connect(self.median)
         self.pushButton_2.clicked.connect(self.gaussian)
         self.pushButton_3.clicked.connect(self.bilateral)
         self.show()
         
     @pyqtSlot()
     def median(self):
         img = cv2.imread('Cat.png')
         img_median = cv2.medianBlur(img, 7)
         cv2.imshow('my_image',img)
         cv2.imshow('median',img_median)
         cv2.waitKey(0)
         cv2.destroyWindow('my_image')
         cv2.destroyWindow('median')
     def gaussian(self):
         img = cv2.imread('Cat.png')
         img_gaussian = cv2.GaussianBlur(img,(3, 3),0)
         cv2.imshow('my_image',img)
         cv2.imshow('gaussian',img_gaussian)
         cv2.waitKey(0)
         cv2.destroyWindow('my_image')
         cv2.destroyWindow('gaussian')
     def bilateral(self):
         img = cv2.imread('Cat.png')
         img_bilateral = cv2.bilateralFilter(img, 9, 90, 90)
         cv2.imshow('my_image',img)
         cv2.imshow('bilateral',img_bilateral)
         cv2.waitKey(0)
         cv2.destroyWindow('my_image')
         cv2.destroyWindow('bilateral')
         
if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())



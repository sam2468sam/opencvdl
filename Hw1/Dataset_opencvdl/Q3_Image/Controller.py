import sys
import cv2
import numpy
import scipy

from HW1_3_UI import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from scipy import signal

class Controller(QMainWindow, Ui_Form):
     def __init__(self, parent=None):
         super(QMainWindow, self).__init__(parent)
         self.setupUi(self)
         self.pushButton.clicked.connect(self.gaussian)
         self.pushButton_2.clicked.connect(self.sobel_x)
         self.pushButton_3.clicked.connect(self.sobel_y)
         self.pushButton_4.clicked.connect(self.magnitude)
         self.show()
         
     @pyqtSlot()
     def gaussian(self):
         img1 = cv2.imread('Chihiro.jpg')
         img2 = cv2.imread('Chihiro.jpg',0)
         x, y = numpy.mgrid[-1:2, -1:2]

         gaussian_kernel = numpy.exp(-(x**2+y**2))
         gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
         grad = signal.convolve2d(img2, gaussian_kernel, boundary='symm', mode='same')
         grad = cv2.convertScaleAbs(grad)

         cv2.imshow('my_image',img1)
         cv2.imshow('gray',img2)
         cv2.imshow('gaussian',grad)
         cv2.waitKey(0)
         cv2.destroyWindow('my_image')
         cv2.destroyWindow('gray')
         cv2.destroyWindow('gaussian')
     def sobel_x(self):
         img1 = cv2.imread('Chihiro.jpg')
         img2 = cv2.imread('Chihiro.jpg',0)
         x, y = numpy.mgrid[-1:2, -1:2]

         gaussian_kernel = numpy.exp(-(x**2+y**2))
         gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
         grad = signal.convolve2d(img2, gaussian_kernel, boundary='symm', mode='same')
         grad1 = grad.astype('uint8')
         z = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]])

         grad_x = signal.convolve2d(grad1, z, boundary='symm', mode='same')
         grad_x = cv2.convertScaleAbs(grad_x)
         #grad_x = grad_x-numpy.min(grad_x)
         #grad_x = grad_x/numpy.max(grad_x)*255
         #grad_x = grad_x.astype('uint8')

         cv2.imshow('sobel_x',grad_x)
         cv2.waitKey(0)
         cv2.destroyWindow('sobel_x')
     def sobel_y(self):
         img1 = cv2.imread('Chihiro.jpg')
         img2 = cv2.imread('Chihiro.jpg',0)
         x, y = numpy.mgrid[-1:2, -1:2]

         gaussian_kernel = numpy.exp(-(x**2+y**2))
         gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
         grad = signal.convolve2d(img2, gaussian_kernel, boundary='symm', mode='same')
         grad1 = grad.astype('uint8')
         z = numpy.array([[1,2,1],[0,0,0],[-1,-2,-1]])

         grad_y = signal.convolve2d(grad1, z, boundary='symm', mode='same')
         grad_y = cv2.convertScaleAbs(grad_y)
         #grad_y = grad_y-numpy.min(grad_y)
         #grad_y = grad_y/numpy.max(grad_y)*255
         #grad_y = grad_y.astype('uint8')

         cv2.imshow('sobel_y',grad_y)
         cv2.waitKey(0)
         cv2.destroyWindow('sobel_y')
     def magnitude(self):
         img1 = cv2.imread('Chihiro.jpg')
         img2 = cv2.imread('Chihiro.jpg',0)
         x, y = numpy.mgrid[-1:2, -1:2]

         gaussian_kernel = numpy.exp(-(x**2+y**2))
         gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
         grad = signal.convolve2d(img2, gaussian_kernel, boundary='symm', mode='same')
         grad1 = grad.astype('uint8')
         X = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]])
         Y = numpy.array([[1,2,1],[0,0,0],[-1,-2,-1]])

         grad_x = signal.convolve2d(grad1, X, boundary='symm', mode='same')
         grad_x = cv2.convertScaleAbs(grad_x)
         grad_x = grad_x.astype('uint16')
         grad_x = numpy.power(grad_x,2)

         grad_y = signal.convolve2d(grad1, Y, boundary='symm', mode='same')
         grad_y = cv2.convertScaleAbs(grad_y)
         grad_y = grad_y.astype('uint16')
         grad_y = numpy.power(grad_y,2)
         
         grad_z = grad_x + grad_y
         grad_z = numpy.power(grad_z,1/2)
         grad_z = cv2.convertScaleAbs(grad_z)
         
         cv2.imshow('sobel_x+y',grad_z)
         cv2.waitKey(0)
         cv2.destroyWindow('sobel_x+y')
         
if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())



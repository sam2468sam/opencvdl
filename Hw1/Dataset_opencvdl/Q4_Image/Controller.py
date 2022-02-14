import sys
import cv2
import numpy
import scipy

from HW1_4_UI import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from scipy import signal

class Controller(QMainWindow, Ui_Form):
     def __init__(self, parent=None):
         super(QMainWindow, self).__init__(parent)
         self.setupUi(self)
         self.pushButton.clicked.connect(self.gaussian)
         self.show()
         
     @pyqtSlot()
     def gaussian(self):
         img = cv2.imread('Parrot.png')
         rotation = self.lineEdit.text()
         scaling = self.lineEdit_2.text()
         tx = self.lineEdit_3.text()
         ty = self.lineEdit_4.text()
         
         rows = img.shape[0]
         cols = img.shape[1]
         M_m = numpy.array([[1,0,tx],[0,1,ty]])
         M_m = M_m.astype('float32')
         M_r = cv2.getRotationMatrix2D((160+int(tx),84+int(ty)),int(rotation),float(scaling))

         cv2.imshow('my_image',img)
         img1 = cv2.warpAffine(img,M_m,(cols,rows))
         img2 = cv2.warpAffine(img1,M_r,(cols,rows))
         cv2.imshow('my_image2',img2)
         cv2.waitKey(0)
         cv2.destroyWindow('my_image')
         cv2.destroyWindow('my_image2')
         
if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())



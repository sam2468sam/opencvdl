# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HW2.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1203, 229)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 210, 190))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(30, 20, 150, 30))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(25, 120, 160, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(25, 150, 160, 16))
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 70, 150, 30))
        self.pushButton_2.setObjectName("pushButton_2")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(250, 20, 450, 190))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 20, 150, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 80, 150, 30))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_6.setGeometry(QtCore.QRect(30, 140, 150, 30))
        self.pushButton_6.setObjectName("pushButton_5")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(220, 20, 201, 151))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 90, 150, 30))
        self.pushButton_5.setObjectName("pushButton_6")
        self.label_3 = QtWidgets.QLabel(self.groupBox_3)
        self.label_3.setGeometry(QtCore.QRect(30, 40, 60, 16))
        self.label_3.setObjectName("label_3")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox.setGeometry(QtCore.QRect(110, 40, 70, 20))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setGeometry(QtCore.QRect(710, 20, 210, 90))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_7.setGeometry(QtCore.QRect(30, 35, 150, 30))
        self.pushButton_7.setObjectName("pushButton_7")
        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setGeometry(QtCore.QRect(710, 120, 210, 90))
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_8.setGeometry(QtCore.QRect(30, 35, 150, 30))
        self.pushButton_8.setObjectName("pushButton_8")
        self.groupBox_6 = QtWidgets.QGroupBox(Form)
        self.groupBox_6.setGeometry(QtCore.QRect(930, 20, 210, 190))
        self.groupBox_6.setObjectName("groupBox_6")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_9.setGeometry(QtCore.QRect(30, 30, 150, 30))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_10.setGeometry(QtCore.QRect(30, 80, 150, 30))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_11.setGeometry(QtCore.QRect(30, 130, 150, 30))
        self.pushButton_11.setObjectName("pushButton_11")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "1. Find Contour"))
        self.pushButton.setText(_translate("Form", "1.1 Draw Contour"))
        self.label.setText(_translate("Form", "There are ___ coins in coin01.jpg"))
        self.label_2.setText(_translate("Form", "There are ___ coins in coin02.jpg"))
        self.pushButton_2.setText(_translate("Form", "1.2 Count Coins"))
        self.groupBox_2.setTitle(_translate("Form", "2. Calibration"))
        self.pushButton_3.setText(_translate("Form", "2.1 Find Corners"))
        self.pushButton_4.setText(_translate("Form", "2.2 Find Intrinsic"))
        self.pushButton_6.setText(_translate("Form", "2.4 Find Distortion"))
        self.groupBox_3.setTitle(_translate("Form", "2.3 Find Extrinsic"))
        self.pushButton_5.setText(_translate("Form", "2.3 Find Extrinsic"))
        # self.label_3.setText(_translate("Form", "Select image"))
        self.label_3.setText(_translate("Form", "<html><head/><body><p>Select image</p></body></html>"))
        self.comboBox.setItemText(0, _translate("Form", "1"))
        self.comboBox.setItemText(1, _translate("Form", "2"))
        self.comboBox.setItemText(2, _translate("Form", "3"))
        self.comboBox.setItemText(3, _translate("Form", "4"))
        self.comboBox.setItemText(4, _translate("Form", "5"))
        self.comboBox.setItemText(5, _translate("Form", "6"))
        self.comboBox.setItemText(6, _translate("Form", "7"))
        self.comboBox.setItemText(7, _translate("Form", "8"))
        self.comboBox.setItemText(8, _translate("Form", "9"))
        self.comboBox.setItemText(9, _translate("Form", "10"))
        self.comboBox.setItemText(10, _translate("Form", "11"))
        self.comboBox.setItemText(11, _translate("Form", "12"))
        self.comboBox.setItemText(12, _translate("Form", "13"))
        self.comboBox.setItemText(13, _translate("Form", "14"))
        self.comboBox.setItemText(14, _translate("Form", "15"))
        self.groupBox_4.setTitle(_translate("Form", "3. Augmented Reality"))
        self.pushButton_7.setText(_translate("Form", "3.1 Augmented Reality"))
        self.groupBox_5.setTitle(_translate("Form", "4. Stereo Disparity Map"))
        self.pushButton_8.setText(_translate("Form", "4.1 Stereo Disparity Map"))
        self.groupBox_6.setTitle(_translate("Form", "5. cats/dogs classification with ResNet50"))
        self.pushButton_9.setText(_translate("Form", "5.2 Show the TensorBoard "))
        self.pushButton_10.setText(_translate("Form", "5.3 Test"))
        self.pushButton_11.setText(_translate("Form", "5.4 Resizing"))

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'canny_edge_detector_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(642, 495)
        self.LowerSlider = QtWidgets.QSlider(Dialog)
        self.LowerSlider.setGeometry(QtCore.QRect(90, 390, 160, 22))
        self.LowerSlider.setOrientation(QtCore.Qt.Horizontal)
        self.LowerSlider.setObjectName("LowerSlider")
        self.Loadbtn = QtWidgets.QPushButton(Dialog)
        self.Loadbtn.setGeometry(QtCore.QRect(210, 430, 75, 23))
        self.Loadbtn.setObjectName("Loadbtn")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 430, 81, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(150, 10, 71, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(390, 10, 71, 21))
        self.label_2.setObjectName("label_2")
        self.splitter = QtWidgets.QSplitter(Dialog)
        self.splitter.setGeometry(QtCore.QRect(90, 50, 481, 301))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.originalImage = QtWidgets.QLabel(self.splitter)
        self.originalImage.setAutoFillBackground(True)
        self.originalImage.setObjectName("originalImage")
        self.edgeImage = QtWidgets.QLabel(self.splitter)
        self.edgeImage.setAutoFillBackground(True)
        self.edgeImage.setObjectName("edgeImage")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(90, 360, 101, 21))
        self.label_3.setObjectName("label_3")
        self.UpperSlider = QtWidgets.QSlider(Dialog)
        self.UpperSlider.setGeometry(QtCore.QRect(310, 390, 160, 22))
        self.UpperSlider.setOrientation(QtCore.Qt.Horizontal)
        self.UpperSlider.setObjectName("UpperSlider")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(310, 360, 101, 21))
        self.label_4.setObjectName("label_4")
        self.LowerSlider.setMinimum(20)  # Adjust the minimum threshold value
        self.LowerSlider.setMaximum(150)  # Adjust the maximum threshold value
        self.LowerSlider.setValue(50)  # Adjust the initial threshold value
        self.UpperSlider.setMinimum(50)  # Adjust the minimum threshold value
        self.UpperSlider.setMaximum(200)  # Adjust the maximum threshold value
        self.UpperSlider.setValue(80)  # Adjust the initial threshold value

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.Loadbtn.setText(_translate("Dialog", "Load Image"))
        self.pushButton_2.setText(_translate("Dialog", "Edge Detection"))
        self.label.setText(_translate("Dialog", "Original Image"))
        self.label_2.setText(_translate("Dialog", "Edge Image"))
        self.originalImage.setText(_translate("Dialog", "TextLabel"))
        self.edgeImage.setText(_translate("Dialog", "TextLabel"))
        self.label_3.setText(_translate("Dialog", "Lower Threshold"))
        self.label_4.setText(_translate("Dialog", "Upper Threshold"))

import math
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        #A1 - A8
        super(ShowImage, self).__init__()
        loadUi('untitled.ui', self)
        self.Image = None
        self.button_loadCitra.clicked.connect(self.fungsi)
        self.button_prosesCitra.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.stretching)
        self.actionNegative_Image.triggered.connect(self.negative)
        self.actionBinary_Image.triggered.connect(self.binary)

        #A9 - C2
        self.actionHistogram_Grayscale.triggered.connect(self.histogram)
        self.actionHistogram_RGB.triggered.connect(self.rgbhist)
        self.actionHistogram_Equalization.triggered.connect(self.equalizehist)

        #B1 - B4
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotasi90d)
        self.action_90_Derajat.triggered.connect(self.rotasi_90d)
        self.action45_Derajat.triggered.connect(self.rotasi45d)
        self.action_45_Derajat.triggered.connect(self.rotasi_45d)
        self.action180_Derajat.triggered.connect(self.rotasi180d)
        self.actionTranspose.triggered.connect(self.transpose)
        self.action2x.triggered.connect(self.zoomIn2x)
        self.action3x.triggered.connect(self.zoomIn3x)
        self.action4x.triggered.connect(self.zoomIn4x)
        self.action1_2.triggered.connect(self.zoomOuthalf)
        self.action1_4.triggered.connect(self.zoomOutquarter)
        self.action3_4.triggered.connect(self.zoomOut75)
        self.actionCrop.triggered.connect(self.crop)

        #C1-C2
        self.actionAdd.triggered.connect(self.adder)
        self.actionSub.triggered.connect(self.subtractor)
        self.actionMul.triggered.connect(self.multipler)
        self.actionDiv.triggered.connect(self.divider)
        self.actionAnd.triggered.connect(self.operasiAnd)
        self.actionOr.triggered.connect(self.operasiOr)
        self.actionXor.triggered.connect(self.operasiXor)

    #A1 - A8
    def fungsi(self):
        self.Image = cv2.imread('3.jpeg')
        self.displayImage(1)

    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)

    def brightness(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 50
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.Image.itemset((i, j), b)

        self.displayImage(2)

    def contrast(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i, j), b)

        self.displayImage(2)

    def stretching(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.Image.itemset((i, j), b)

        self.displayImage(2)

    def negative(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = math.ceil(255 - a)

                self.Image.itemset((i, j), b)

        self.displayImage(2)

    def binary(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                if a < 89:
                    b = 0
                else:
                    b = 255

                self.Image.itemset((i, j), b)

        self.displayImage(2)

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        if windows == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))

            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)

        if windows == 2:
            self.hasilLabel.setPixmap(QPixmap.fromImage(img))

            self.hasilLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.hasilLabel.setScaledContents(True)


    #A9 - A11
    def histogram(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)

        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()

    def rgbhist(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.Image], [i], None, [256], [0, 256])
            plt.plot(histo, color=col)
            plt.xlim([0, 256])
        self.displayImage(2)
        plt.show()

    def equalizehist(self):
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.Image = cdf[self.Image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.Image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    # B1 - B4
    def translasi(self):
        h, w = self.Image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w, h))
        self.Image = img
        self.displayImage(2)

    def rotasi(self, degree):

        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))
        self.Image = rot_image
        self.displayImage(2)



    def rotasi90d(self):
        self.rotasi(90)
    def rotasi_90d(self):
        self.rotasi(-90)
    def rotasi45d(self):
        self.rotasi(45)
    def rotasi_45d(self):
        self.rotasi(-45)
    def rotasi180d(self):
        self.rotasi(180)

    def transpose(self):
        img = cv2.transpose(self.Image)
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imshow('Original', self.Image)
        cv2.imshow('Transposed', img)
        cv2.waitKey()

    def zoomIn2x(self):
        scale=2
        resize_img = cv2.resize(self.Image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_img)
        cv2.waitKey()

    def zoomIn3x(self):
        scale=3
        resize_img = cv2.resize(self.Image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_img)
        cv2.waitKey()

    def zoomIn4x(self):
        scale=4
        resize_img = cv2.resize(self.Image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_img)
        cv2.waitKey()

    def zoomOuthalf(self):
        scale=0.5
        resize_img = cv2.resize(self.Image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom Out', resize_img)
        cv2.waitKey()

    def zoomOutquarter(self):
        scale=0.25
        resize_img = cv2.resize(self.Image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom Out', resize_img)
        cv2.waitKey()

    def zoomOut75(self):
        scale=0.75
        resize_img = cv2.resize(self.Image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom Out', resize_img)
        cv2.waitKey()

    def crop(self):
        h, w = self.Image.shape[:2]
        start_x=1
        start_y=1
        end_x=100
        end_y=100
        img = self.Image
        crop_img = img[start_x:end_x, start_y:end_y]
        cv2.imshow('Original', self.Image)
        cv2.imshow('Cropped', crop_img)
        cv2.waitKey()

    #C1-C2
    def adder(self):
        img1 = cv2.imread('1.jpeg', 0)
        img2 = cv2.imread('2.jpeg', 0)
        add = img1 + img2
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Added', add)
        cv2.waitKey()

    def subtractor(self):
        img1 = cv2.imread('1.jpeg', 0)
        img2 = cv2.imread('2.jpeg', 0)
        subtract = img1 - img2
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Subtracted', subtract)
        cv2.waitKey()

    def multipler(self):
        img1 = cv2.imread('1.jpeg', 0)
        img2 = cv2.imread('2.jpeg', 0)
        mul = img1 * img2
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Multiplied', mul)
        cv2.waitKey()

    def divider(self):
        img1 = cv2.imread('1.jpeg', 0)
        img2 = cv2.imread('2.jpeg', 0)
        div = img1 / img2
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Divided', div)
        cv2.waitKey()

    def operasiAnd(self):
        img1 = cv2.imread('1.jpeg', 1)
        img2 = cv2.imread('2.jpeg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        and_op = cv2.bitwise_and(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('And result', and_op)
        cv2.waitKey()

    def operasiOr(self):
        img1 = cv2.imread('1.jpeg', 1)
        img2 = cv2.imread('2.jpeg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        or_op = cv2.bitwise_or(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Or result', or_op)
        cv2.waitKey()

    def operasiXor(self):
        img1 = cv2.imread('1.jpeg', 1)
        img2 = cv2.imread('2.jpeg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        xor_op = cv2.bitwise_xor(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Xor result', xor_op)
        cv2.waitKey()



app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 2')
window.show()
sys.exit(app.exec_())

# image = cv2.imread('dodge.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('Gambar 1', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

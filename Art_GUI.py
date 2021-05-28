import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
import qimage2ndarray

import numpy as np

from torchvision.utils import make_grid
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.InitUI()

    def InitUI(self):
        self.model = None
        # Set Label to discription
        label_original_img = QLabel('Original Image', self)
        label_original_img.move(195, 540)

        label_converted_img = QLabel('Converted Image', self)
        label_converted_img.move(675, 540)

        # Set Label to show image
        self.original_img = QLabel('Nothing',self)
        self.original_img.setFixedSize(420,420)
        self.original_img.move(30,60)
        self.original_img.setAlignment(Qt.AlignCenter)

        self.result_img = QLabel('Nothing', self)
        self.result_img.setFixedSize(420, 420)
        self.result_img.move(510, 60)
        self.result_img.setAlignment(Qt.AlignCenter)

        # Set button action
        btn_open_img = QPushButton('Open Image',self)
        btn_open_img.move(195,20)
        btn_open_img.clicked.connect(self.OpenImage)

        btn_open_img = QPushButton('Load model', self)
        btn_open_img.move(435, 20)
        btn_open_img.clicked.connect(self.loadmodel)

        btn_open_img = QPushButton('Convert Image', self)
        btn_open_img.move(675, 20)
        btn_open_img.clicked.connect(self.convert)

        # Set MainWindow
        self.setWindowTitle('Art Converter with GAN')
        self.setGeometry(300, 300, 960, 640)
        self.show()

    def OpenImage(self):
        fname = QFileDialog.getOpenFileName(self,'Open file', './',"Image files (*.jpg *.gif *.png)")

        imagepath = fname[0]
        self.input = Image.open(imagepath)
        self.pixmap = QPixmap(imagepath).scaled(420, 420, aspectRatioMode=Qt.IgnoreAspectRatio,
                                                transformMode=Qt.SmoothTransformation)

        self.original_img.setPixmap(self.pixmap)
        self.original_img.adjustSize()


    def convert(self):
        to_tensor = transforms.ToTensor()

        input_img_T = to_tensor(self.input)

        if input_img_T.size() != torch.Size([3,256,256]):
            input_img_T = transforms.functional.resize(input_img_T, (256, 256),
                                                interpolation=transforms.InterpolationMode.BICUBIC)
        output = self.run(input_img_T)

        qimg = qimage2ndarray.array2qimage(output, normalize=True)

        pixmap = QPixmap.fromImage(qimg).scaled(420, 420, aspectRatioMode=Qt.IgnoreAspectRatio,
                                                transformMode=Qt.SmoothTransformation)
        self.result_img.setPixmap(QPixmap(pixmap))
        self.result_img.adjustSize()

    def loadmodel(self):
        model_file,_= QFileDialog.getOpenFileName(self,'Open file', './',"Model (*.pt)")
        self.model = torch.load(model_file,map_location=torch.device('cpu'))
        if self.model != None:
            message = QMessageBox.information(self, 'Message', 'Model load Success!',QMessageBox.Yes)
        else:
            message = QMessageBox.Warning(self, 'Message', 'Model load Fail!', QMessageBox.Yes)

    def unnormalize(self, image, mean_=0.5, std_=0.5):
        if torch.is_tensor(image):
            image = image.detach().numpy()
        un_normalized_img = image * std_ + mean_
        un_normalized_img = un_normalized_img * 255
        return np.uint8(un_normalized_img)

    def run(self,input):
        mean_ = 0.5
        std_ = 0.5

        # Create fake pictures for both cycles
        converted = self.model(input.unsqueeze(0))

        # Generate grids
        grid_converted = make_grid(converted).permute(1, 2, 0).detach().numpy()

        # Normalize pictures to pixel range rom 0 to 255
        output = self.unnormalize(grid_converted, mean_, std_)

        return output

# Create Generator
class Resblock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            nn.InstanceNorm2d(256)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            nn.InstanceNorm2d(256)
        )

    def forward(self, inputs):
        output = torch.nn.functional.relu(self.conv1(inputs))
        return inputs + self.conv2(output)

    ''' for 1,2    def forward(self,inputs):
        output = torch.nn.functional.relu(self.conv1(inputs))
        return torch.nn.functional.relu(inputs + self.conv2(output))

  
'''
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )

        resblock_layer = []
        for i in range(9):
            resblock_layer += [Resblock()]

        self.resblock = nn.Sequential(*resblock_layer)

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):

        output = self.conv1(inputs)
        output = self.downsampling(output)
        output = self.resblock(output)
        output = self.upsampling(output)
        output = self.conv2(output)

        return output

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    app.exec_()
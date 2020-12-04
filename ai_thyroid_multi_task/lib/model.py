import torch
from torch import nn, optim
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
    
    
class DoubleConv_v2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv_v2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Net_v2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Net_v2, self).__init__()

        # U-Net
        self.conv1 = DoubleConv_v2(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv_v2(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv_v2(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv_v2(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv_v2(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv_v2(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv_v2(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv_v2(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv_v2(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        
        # Auxiliary task: Malignancy
        self.aux_conv6 = nn.Conv2d(1024, 1, 1)
        self.aux_linear1 = nn.Linear(1024, 1)

    def forward(self, x):
        # U-Net
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        #segmentation = nn.Sigmoid()(c10)
        
        # Auxiliary task: Malignancy
        aux_c6 = self.aux_conv6(c5)
        aux_l1 = self.aux_linear1(nn.Flatten()(aux_c6))
        malignant = nn.Sigmoid()(aux_l1)
        
        return c10, malignant

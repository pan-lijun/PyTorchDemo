from model.unet_parts import *


class MyUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MyUNet, self).__init__()
        self.train_mode = True
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = MyDown(64, 128)
        self.down2 = MyDown(128, 256)
        self.down3 = MyDown(256, 512)
        self.down4 = MyDown(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)

        return x

    def set_train_mode(self, is_train):
        self.train_mode = is_train
        self.down1.set_train_mode(is_train)
        self.down2.set_train_mode(is_train)
        self.down3.set_train_mode(is_train)
        self.down4.set_train_mode(is_train)

    def merge_weights(self):
        self.down1.merge_weights()
        self.down2.merge_weights()
        self.down3.merge_weights()
        self.down4.merge_weights()

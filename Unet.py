import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, 
        kernel_size = 3, stride=1, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, block_list = [Block(3, 64), Block(64, 128), 
                Block(128, 256), Block(256, 512), Block(512, 2024)]):
        super().__init__()
        self.enc_blocks = nn.ModuleList(block_list)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        outputs_block=[]
        for block in self.enc_blocks:
            x = block(x)
            outputs_block.append(x)
            x=self.pool(x)
        return outputs_block


class Decoder(nn.module):
    def __init__(self, l = [1024, 512, 256, 128, 64]):
        super().__init__()
        self.chs = l
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(l[i], l[i+1], 2, 2) for i in range(l)])
        self.dec_blocks = nn.ModuleList([Block(l[i], l[i+1]) for i in range(l)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W =  x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.module):
    def __init__(self, enc_chs = [3, 64, 128, 256, 512, 2024], 
    dec_chs =[1024, 512, 256, 128, 64], num_class = 1, retain_dim=False, 
    out_sz=(572, 572) ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        x = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        x = self.head(x)
        if self.retain_dim:
            x = F.interpolate(x, self.out_sz)
        return x







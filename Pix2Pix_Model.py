import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import models
from torchsummary import summary
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import sys 

# PatchGAN Discriminator Model Implementation

'''PatchGAN  only penalizes structure at the scale of patches. This discriminator tries to 
classify if each NxN patch in an image is real or fake. We run this discriminator convolutionally across 
the image, averaging all responses to provide the ultimate output of D.'''

# example of calculating the receptive field for the PatchGAN

# receptive field = (output size  1) * stride + kernel size

# calculate the effective receptive field size


def receptive_field(output_size, kernel_size, stride_size):
    return (output_size - 1) * stride_size + kernel_size

'''The PatchGAN configuration is defined using a shorthand notation as: C64-C128-C256-C512, where C refers to 
a block of Convolution-BatchNorm-LeakyReLU layers and the number indicates the number of filters. 
Batch normalization is not used in the first layer. As mentioned, the kernel size is fixed at 4×4 and a stride
of 2×2 is used on all but the last 2 layers of the model (use 1x1 stride). The slope of the LeakyReLU is set to 
0.2, and a sigmoid activation function is used in the output layer.'''


class PatchGAN_Discriminator(nn.Module):
    # define the discriminator model

    def __init__(self, input_channel):

        super(PatchGAN_Discriminator, self).__init__()

        kernel_s = 4
        stride_s = 2

        # C64 - image input (256x256)
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=kernel_s,
                      stride=stride_s, padding=1),
            # Batch normalization is not used in the first layer
            nn.LeakyReLU(0.2, True)
        )

        # C128 - image input (128x128)
        self.layer2 = nn.Sequential(
            # no need to use bias as BatchNorm2d has affine parameters
            nn.Conv2d(64, 128, kernel_size=kernel_s,
                      stride=stride_s, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )

        # C256 - image input (64x64)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=kernel_s,
                      stride=stride_s, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        # 512 - image input (32x32)
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=kernel_s,
                      stride=stride_s, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        # second last output layer - image input (16x16)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_s,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        # patch output
        self.layer6 = nn.Conv2d(
            512, 1, kernel_size=kernel_s, stride=1, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return self.sigmoid(out)

# Unet Generator Model Implementation


'''Let Ck denote a Convolution-BatchNorm-ReLU layer with k filters. CDk denotes a 
ConvolutionTranpose-BatchNormDropout-ReLU layer with a dropout rate of 50%. All convolutions are 4× 4 
spatial filters applied with stride 2.

Encoder: C64-C128-C256-C512-C512-C512-C512-C512
Decoder: CD512-CD512-CD512-C512-C256-C128-C64'''


class UnetGenerator(nn.Module):

    def __init__(self, input_channel):

        super(UnetGenerator, self).__init__()

        kernel_s = 4
        stride_s = 2

        def encoder_block(channel_in, n_filters, batch_norm=True):
            # no need to use bias when BatchNorm2d is true (has already affine parameters)
            if(batch_norm):
                use_bias = False
            else:
                use_bias = True

            conv_layer = nn.Conv2d(
                channel_in, n_filters, kernel_size=kernel_s, stride=stride_s, padding=1, bias=use_bias)
            layers = [conv_layer]

            if(batch_norm):
                batc_layer = nn.BatchNorm2d(n_filters)
                layers = layers + [batc_layer]

            leaky_layer = nn.LeakyReLU(0.2, True)
            layers = layers + [leaky_layer]

            out_block = nn.Sequential(*layers)

            return out_block

        def decoder_block(channel_in, n_filters, dropout=True):
            # add upsampling conv. layer
            conv_layer = nn.ConvTranspose2d(
                channel_in, n_filters, kernel_size=kernel_s, stride=stride_s, padding=1, bias=False)

            batc_layer = nn.BatchNorm2d(n_filters)

            layers = [conv_layer, batc_layer]

            if(dropout):
                drop = nn.Dropout(0.5)
                layers = layers + [drop]

            out_block = nn.Sequential(*layers)

            return out_block  # For each decoder_block, concatenate the output with corresponding coder output and pass activ relu
        # C64
        self.enc_layer1 = encoder_block(input_channel, 64, False)
        # C128
        self.enc_layer2 = encoder_block(64, 128)
        # C256
        self.enc_layer3 = encoder_block(128, 256)
        # C512
        self.enc_layer4 = encoder_block(256, 512)
        # C512-C512-C512
        self.enc_layer5_7 = encoder_block(512, 512)
        # bottleneck, no batch norm and relu - last layer of encoder block -C512
        self.bottleneck = nn.Conv2d(
            512, 512, kernel_size=kernel_s, stride=stride_s, padding=1)
        # CD512
        self.dec_layer1 = decoder_block(512, 512)
        # CD512-CD512 - input 512 concat. with 512 = 1024
        self.dec_layer2_3 = decoder_block(1024, 512)
        # C512 - last layer without dropout - 512 concat 512
        self.dec_layer4 = decoder_block(1024, 512, dropout=False)
        # C256 - dec.layer4 (512) concat with enc_layer4(512)
        self.dec_layer5 = decoder_block(1024, 256, dropout=False)
        # C128 - concat (256) with (256)
        self.dec_layer6 = decoder_block(512, 128, dropout=False)
        # C64 - concat (128) with (128)
        self.dec_layer7 = decoder_block(256, 64, dropout=False)
        # output
        self.out = nn.ConvTranspose2d(
            64, 1, kernel_size=kernel_s, stride=stride_s, padding=1)

        # Activation functions
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out_enc1 = self.enc_layer1(x)
        out_enc2 = self.enc_layer2(out_enc1)
        out_enc3 = self.enc_layer3(out_enc2)
        out_enc4 = self.enc_layer4(out_enc3)
        out_enc5 = self.enc_layer5_7(out_enc4)
        out_enc6 = self.enc_layer5_7(out_enc5)
        out_enc7 = self.enc_layer5_7(out_enc6)
        out_enc8 = self.bottleneck(out_enc7)
        out_enc8 = self.relu(out_enc8)
        out_dec1 = self.dec_layer1(out_enc8)
        in_dec2 = concatenate_torch(out_dec1, out_enc7)
        in_dec2 = self.relu(in_dec2)
        out_dec2 = self.dec_layer2_3(in_dec2)
        in_dec3 = concatenate_torch(out_dec2, out_enc6)
        in_dec3 = self.relu(in_dec3)
        out_dec3 = self.dec_layer2_3(in_dec3)
        in_dec4 = concatenate_torch(out_dec3, out_enc5)
        in_dec4 = self.relu(in_dec4)
        out_dec4 = self.dec_layer4(in_dec4)
        in_dec5 = concatenate_torch(out_dec4, out_enc4)
        in_dec5 = self.relu(in_dec5)
        out_dec5 = self.dec_layer5(in_dec5)
        in_dec6 = concatenate_torch(out_dec5, out_enc3)
        in_dec6 = self.relu(in_dec6)
        out_dec6 = self.dec_layer6(in_dec6)
        in_dec7 = concatenate_torch(out_dec6, out_enc2)
        in_dec7 = self.relu(in_dec7)
        out_dec7 = self.dec_layer7(in_dec7)
        out = self.out(out_dec7)
        out = self.tanh(out)
        return out

def _initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean=0, std=0.02)


def concatenate_torch(torch_listA, torch_listB):
    torch_cat = torch.cat((torch_listA, torch_listB), 1)
    return torch_cat

def print_models(print_D, print_G):
    # Printing models:
    if print_D is True:
        print("Discriminator model: \n\n")
        D = PatchGAN_Discriminator(6)
        D.apply(_initialize_weights)
        D.cuda()
        summary(D, (6, 256, 256))
    if print_G is True:
        print("\n\nGenerator model: \n\n")
        G = UnetGenerator(3)
        G.apply(_initialize_weights)
        G.cuda()
        summary(G, (3, 256, 256))

#Set if the net requires gradient or not
def set_requires_grad(net, requires_grad=False):
    if net is not None:
        for param in net.parameters():
                    param.requires_grad = requires_grad

def load_dataset(batch_size, data_path, grayscale):
    if grayscale is True:
        transf = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
    else:
        transf = transforms.ToTensor()
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transf
    )
    #print(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size
    )
    return loader



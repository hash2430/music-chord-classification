'''
model_archive.py

A file that contains neural network models.
You can also make different model like CNN if you follow similar format like given RNN.
'''
import torch.nn as nn
import numpy as np
import torch

class RNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, 1, bidirectional=True, dropout=0.25) #(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(2*64, num_classes)  # 2 for bidirection

    def forward(self, x):
        output, hidden = self.lstm(x, None) # input(16,32,12) output(16, 32, 128)
        output = self.linear(output[-1]) #output(16,32,128) => (32,25)

        return output

import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, num_classes):
        super(BidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, 64, 1, bidirectional=True, dropout=0.25)
        #self.embedding = nn.LeakyReLU(0.2, inplace=True)
        self.embedding = nn.Linear(2 * 64, num_classes)

    def forward(self, input):
        recurrent, _ = self.lstm(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, input_size, num_class):
        super(CRNN, self).__init__()

        self.cnn0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.cnn1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(3, stride=3)
        )
        # self.cnn2 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=0),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        self.rnn = nn.Sequential(
            #BidirectionalLSTM(2, 25)
            RNN(128, 25)
        )

    def forward(self, input):
        # conv features
        output = torch.unsqueeze(input, 1) # add channel dimension
        output = output.permute(0, 1, 3, 2) #b, c, f, t
        conv = self.cnn0(output)
        b, c, h, w = conv.size()
        conv = self.cnn1(conv)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [T, b, c]
        # rnn features
        output = self.rnn(conv)

        return output

#######
# import torch.nn as nn
# import torch
#
# class BidirectionalLSTM(nn.Module):
#
#     def __init__(self, nIn, nHidden, nOut):
#         super(BidirectionalLSTM, self).__init__()
#
#         self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
#         self.embedding = nn.Linear(nHidden * 2, nOut)
#
#     def forward(self, input):
#         recurrent, _ = self.rnn(input)
#         T, b, h = recurrent.size()
#         t_rec = recurrent.view(T * b, h)
#
#         output = self.embedding(t_rec)  # [T * b, nOut]
#         output = output.view(T, b, -1)
#
#         return output
#
#
# class CRNN(nn.Module):
#
#     def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
#         super(CRNN, self).__init__()
#         assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
#
#         ks = [3, 3, 3, 3, 3, 3, 2]
#         ps = [1, 1, 1, 1, 1, 1, 0]
#         ss = [1, 1, 1, 1, 1, 1, 1]
#         nm = [64, 128, 256, 256, 512, 512, 512]
#
#         cnn = nn.Sequential()
#
#         def convRelu(i, batchNormalization=False):
#             nIn = nc if i == 0 else nm[i - 1]
#             nOut = nm[i]
#             cnn.add_module('conv{0}'.format(i),
#                            nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
#             if batchNormalization:
#                 cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
#             if leakyRelu:
#                 cnn.add_module('relu{0}'.format(i),
#                                nn.LeakyReLU(0.2, inplace=True))
#             else:
#                 cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
#
#         convRelu(0)
#         cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
#         convRelu(1)
#         cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
#         convRelu(2, True)
#         convRelu(3)
#         cnn.add_module('pooling{0}'.format(2),
#                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
#         convRelu(4, True)
#         convRelu(5)
#         cnn.add_module('pooling{0}'.format(3),
#                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
#         convRelu(6, True)  # 512x1x16
#
#         self.cnn = cnn
#         self.rnn = nn.Sequential(
#             BidirectionalLSTM(512, nh, nh),
#             BidirectionalLSTM(nh, nh, nclass))
#
#     def forward(self, input):
#         # conv features
#         #input = torch.unsqueeze(input, 1)
#         conv = self.cnn(input)
#         b, c, h, w = conv.size()
#         assert h == 1, "the height of conv must be 1"
#         conv = conv.squeeze(2)
#         conv = conv.permute(2, 0, 1)  # [w, b, c]
#
#         # rnn features
#         output = self.rnn(conv)
#
#         return output

#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random


class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        return res




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
#         self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        
        chun1, chun2, chun3, chun4  = torch.chunk(in_tensor, 4, dim=3) # chun3, chun4 
        # random FLCAM
#         print("f_check",in_tensor.shape[3])
#         freq_r = torch.randint(1, in_tensor.shape[3], (1,))
#         chun1, chun2 = in_tensor[:,:,:,:freq_r],in_tensor[:,:,:,freq_r:]

#         print(chun1.shape, chun2.shape, chun3.shape)
#         print(chun1.shape)
        
#         chunck1 = chun1.permute(0,2,1,3)
#         chunck2 = chun2.permute(0,2,1,3)
#         chunck3 = chun3.permute(0,2,1,3)
#         chunck4 = chun4.permute(0,2,1,3)
        
        
        avg_pool1 = F.avg_pool2d( chun1, (chun1.size(2), chun1.size(3)), stride=(chun1.size(2), chun1.size(3)) )
#         max_pool1 = F.max_pool2d( chun1, (chun1.size(2), chun1.size(3)), stride=(chun1.size(2), chun1.size(3)) )
        avg_pool2 = F.avg_pool2d( chun2, (chun2.size(2), chun2.size(3)), stride=(chun2.size(2), chun2.size(3)) )
#         max_pool2 = F.max_pool2d( chun2, (chun2.size(2), chun2.size(3)), stride=(chun2.size(2), chun2.size(3)) )
        avg_pool3 = F.avg_pool2d( chun3, (chun3.size(2), chun3.size(3)), stride=(chun3.size(2), chun3.size(3)) )
        avg_pool4 = F.avg_pool2d( chun4, (chun4.size(2), chun4.size(3)), stride=(chun4.size(2), chun4.size(3)) )
#         print(self.gate_c)
#         print(avg_pool1.shape)
        t1 = self.gate_c( avg_pool1 ).unsqueeze(2).unsqueeze(3).expand_as(chun1)
#         t2 = self.gate_c( avg_pool1 ).unsqueeze(2).unsqueeze(3).expand_as(chun1)
#         mfm_t1 = torch.maximum(t1,t2)
#         max_t1 = self.gate_c( max_pool1 ).unsqueeze(2).unsqueeze(3).expand_as(chun1)
        t2 = self.gate_c( avg_pool2 ).unsqueeze(2).unsqueeze(3).expand_as(chun2)
#         t4 = self.gate_c( avg_pool2 ).unsqueeze(2).unsqueeze(3).expand_as(chun2)
#         mfm_t2 = torch.maximum(t3,t4)
#         max_t2 = self.gate_c( max_pool2 ).unsqueeze(2).unsqueeze(3).expand_as(chun2)
        t3 = self.gate_c( avg_pool3 ).unsqueeze(2).unsqueeze(3).expand_as(chun3)
        t4 = self.gate_c( avg_pool4 ).unsqueeze(2).unsqueeze(3).expand_as(chun4)
#         final_t1 = t1 + max_t1
#         final_t2 = t2 + max_t2

        
        out_tensor = torch.cat((t1,t2,t3,t4),dim=3)  #,t3,t4
        
#         print(out_tensor.shape)
        
        return out_tensor
    
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
#         self.FreqChan_att = FreqChanGate(gate_channel)
    def forward(self,in_tensor):
#         x = self.channel_att(in_tensor)
#         x_out = self.FreqChan_att(x)
        att  = 1+ torch.sigmoid( self.channel_att(in_tensor)) #* self.spatial_att(in_tensor), self.channel_att(in_tensor) *1 +  F.sigmoid(  self.FreqChan_att(in_tensor) )
        return att * in_tensor

    
class CNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 n_basis_kernels=4,
                 DY_layers=[0, 1, 1, 1, 1, 1, 1],
                 atte_layers=[0, 1, 1, 1, 1, 1, 1],
                 temperature=31,
                 pool_dim='freq'):
        super(CNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
#             if DY_layers[i] == 1:
#                 cnn.add_module("conv{0}".format(i), Dynamic_conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i],
#                                                                    n_basis_kernels=n_basis_kernels,
#                                                                    temperature=temperature, pool_dim=pool_dim))
#             else:
            cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))
                
                
            if atte_layers[i] == 1:
                cnn.add_module("attention{0}".format(i), BAM(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):    #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        #self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x


class CRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 n_class=10,
                 activation="glu",
                 conv_dropout=0.5,
                 
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 **convkwargs):
        super(CRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.cnn = CNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)
        self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(dim=1)          # softmax on time dimension
            elif self.attention == "class":
                self.softmax = nn.Softmax(dim=-1)         # softmax on class dimension

    def forward(self, x): #input size : [bs, freqs, frames]
        #cnn
        if self.n_input_ch > 1:
            x = x.transpose(2, 3)
        else:
            x = x.transpose(1, 2).unsqueeze(1) #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
#             print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(bs, frame, ch*freq)   # x.contiguous.view(bs, frame, ch*freq) 
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1) # x size : [bs, frames, chan]

        #rnn
        x = self.rnn(x) #x size : [bs, frames, 2 * chan]
        x = self.dropout(x)

        #classifier
        strong = self.dense(x) #strong size : [bs, frames, n_class]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x) #sof size : [bs, frames, n_class]
            sof = self.softmax(sof) #sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1) # [bs, n_class]
        else:
            weak = strong.mean(1)

        return strong.transpose(1, 2), weak




#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#             )
#         self.pool_types = pool_types
#     def forward(self, x):
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type=='avg':
#                 avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( avg_pool ) 
#             elif pool_type=='max':
#                 max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( max_pool )   
#             elif pool_type=='lp':
#                 lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( lp_pool )
#             elif pool_type=='lse':
#                 # LSE pool only
#                 lse_pool = logsumexp_2d(x)
#                 channel_att_raw = self.mlp( lse_pool )

#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#             else:
#                 channel_att_sum = channel_att_sum + channel_att_raw

#         scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x * scale 

# def logsumexp_2d(tensor):
#     tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#     s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#     outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
#     return outputs

# class ChannelPool(nn.Module):  
#     def forward(self, x):
#         return torch.cat( (torch.max(x,dim=2)[0].unsqueeze(1), torch.mean(x,dim=2).unsqueeze(1)), dim=1 )

# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = F.sigmoid(x_out) # broadcasting
#         scale = scale.permute(0,2,1,3)
#         return x * scale

# class CBAM(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
#         super(CBAM, self).__init__()
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#     def forward(self, x):
#         x_out = self.ChannelGate(x)
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x_out)
#         return x_out
    
    


# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, pool_types=['avg', 'max']):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.atte = nn.Sequential(
#             nn.Conv1d(self.gate_channels, 4, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm1d(4),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(4, 4, 1, bias=True),
#         )
#         self.pool_types = pool_types
#     def forward(self, x):
#         Freq_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type=='avg':
#                 avg_pool = torch.mean(x,dim=2)   #   ( bz, c, freq )
# #                 print(avg_pool.shape)
#                 Freq_att_raw = self.atte(avg_pool)
#             elif pool_type=='max':
#                 max_pool = torch.max( x, dim=2)[0] #   ( bz, c, freq )
# #                 print(max_pool.shape)
#                 Freq_att_raw = self.atte(max_pool)
            
#             if Freq_att_sum is None:
#                 Freq_att_sum = torch.sum(Freq_att_raw, dim= 1)
#             else:
#                 Freq_att_sum = Freq_att_sum + torch.sum(Freq_att_raw, dim=1)

#         scale = F.sigmoid(Freq_att_sum).unsqueeze(1).unsqueeze(1).expand_as(x)
#         return x * scale



# class ChannelPool(nn.Module):
#     def forward(self, x):
#         return torch.cat( (torch.max(x,dim=2)[0].unsqueeze(1), torch.mean(x,dim=2).unsqueeze(1)), dim=1 )

# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = F.sigmoid(x_out) # broadcasting
#         scale = scale.permute(0,2,1,3)
#         return x * scale

# class CBAM(nn.Module):
#     def __init__(self, gate_channels, pool_types=['avg', 'max'], no_spatial=False):
#         super(CBAM, self).__init__()
#         self.ChannelGate = ChannelGate(gate_channels)
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#     def forward(self, x):
#         x_out = self.ChannelGate(x)
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x_out)
#         return x_out



# class SKConv(nn.Module):
#     def __init__(self, features, M=2, G=8, r=16, stride=1 ,L=32):   #WH
#         """ Constructor
#         Args:
#             features: input channel dimensionality.
#             WH: input spatial dimensionality, used for GAP kernel size.
#             M: the number of branchs.
#             G: num of convolution groups.
#             r: the radio for compute d, the length of z.
#             stride: stride, default 1.
#             L: the minimum dim of the vector z in paper, default 32.
#         """
#         super(SKConv, self).__init__()
#         d = max(int(features/r), L)
#         self.M = M
#         self.features = features
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
#                 nn.BatchNorm2d(features),
#                 nn.ReLU()
#             ))
#         # self.gap = nn.AvgPool2d(int(WH/stride))
#         self.fc = nn.Linear(features, d)
#         self.fcs = nn.ModuleList([])
#         for i in range(M):
#             self.fcs.append(
#                 nn.Linear(d, features)
#             )
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         for i, conv in enumerate(self.convs):
#             fea = conv(x).unsqueeze(1)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)
#         fea_U = torch.sum(feas, dim=1)
#         # fea_s = self.gap(fea_U).squeeze_()
#         fea_s = fea_U.mean(-1).mean(-1)
#         fea_z = self.fc(fea_s)
#         for i, fc in enumerate(self.fcs):
#             vector = fc(fea_z).unsqueeze(1)
#             if i == 0:
#                 attention_vectors = vector
#             else:
#                 attention_vectors = torch.cat([attention_vectors, vector], dim=1)
#         attention_vectors = self.softmax(attention_vectors)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         fea_v = (feas * attention_vectors).sum(dim=1)
#         return fea_v


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
                 pool_dim='time'):
        super(CNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ='cg'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
#             if DY_layers[i] == 1:
#                 cnn.add_module("conv{0}".format(i), Dynamic_conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i],
#                                                                    n_basis_kernels=n_basis_kernels,
#                                                                    temperature=temperature, pool_dim=pool_dim))
#             else:
            cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
    
    
#             if atte_layers[i] == 1:
#                 cnn.add_module("attention{0}".format(i), CBAM(out_dim))
                
    
    
    
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
                      
#             if atte_layers[i] == 1:
#                 cnn.add_module("attention{0}".format(i), CBAM(out_dim))
                
            

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))  #Avg
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
        self.rnn = BiGRU(n_in=256, n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

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
        print(x.shape)
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
        print("wowowowow",x.shape)
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





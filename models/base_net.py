import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from typing import Callable, Optional, Tuple, Union
import math
import sys
import numbers
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


class Darionet(nn.Module):
    def __init__(self, mean, std ,filter_size, reg_beta, input_dims,out_dims=1,p = 0.1):
        super(Darionet, self).__init__()

        ## Dimensions of the input layer
        self.input_dims = input_dims
        self.mean = nn.Parameter(torch.tensor(mean).float(), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(), requires_grad=False)

        ## Conv layer parameters
        self.conv1d_dims = self.input_dims
        self.k_number = 1
        self.k_width = filter_size
        self.k_stride = 1

        ## Fully connected layers dimensions
        self.fc1_dims = 128
        self.fc2_dims = 64
        self.fc3_dims = 16
        self.out_dims = out_dims

        ## L2 regularization (implemented later in the forward pass)
        self.beta = reg_beta / 2.0

        ## Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.k_number, kernel_size=self.k_width,
                               stride=self.k_stride, padding='same')

        ## Fully connected layers
        self.fc1 = nn.Linear(self.conv1d_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        ## manual dropout layer
        self.dropout = ManualDropout(p=p)
        ## Output layer
        self.output_layer = nn.Linear(self.fc3_dims, self.out_dims)

        ## He initialization (Kaiming initialization in PyTorch)
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply He normal initialization to layers"""
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Reshape input to (batch_size, 1, input_dims) for 1D convolution
        x = (x - self.mean) / self.std

        # Convolutional layer followed by ELU activation
        x = F.elu(self.conv1(x))

        # Flatten the tensor after convolution
        x = x.view(x.size(0), -1)

        # Fully connected layers with ELU activation
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        #Dropout layer
        x = self.dropout(x)
        # Output layer with linear activation
        output = self.output_layer(x)

        # Return the final output
        return output


class ManualDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ManualDropout, self).__init__()
        self.p = p


    def forward(self, x):

        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)


class CuiNet(nn.Module):
    def __init__(self, input_dims, mean,std,out_dims=1):
        super(CuiNet, self).__init__()
        
        # Layers dimensions
        self.conv1d_dims = input_dims-4         # size of spectrum - (kernel_size-1) is the size of the spectrum after first conv1D
        self.k_number = 1
        self.k_width = 5
        self.k_stride = 1
        self.fc1_dims = 36
        self.fc2_dims = 18
        self.fc3_dims = 12
        self.out_dims = out_dims
        self.mean = nn.Parameter(torch.tensor(mean).float(),requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(),requires_grad=False)
        
        # Convolutional layer
        self.conv1d = nn.Conv1d(1,1, kernel_size= 5 , stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv1d_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.out = nn.Linear(self.fc3_dims, self.out_dims)
        
    #     # Initialize weights with He normal
    #     self._initialize_weights()
        
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Reshape input
        x = (x-self.mean)/self.std
        # Convolutional layer with ELU activation
        x = F.elu(self.conv1d(x))
        
        # Flatten the output from conv1d
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ELU activation
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        
        # Output layer with linear activation
        x = self.out(x)
        return x


###############################################################################   
    
class ConvBlock1D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock1D, self).__init__()

        # 1d convolution
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # batchnorm
        self.batchnorm1d = nn.BatchNorm1d(out_channels)

        # relu layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batchnorm1d(x)
        x = self.relu(x)
        return x
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1 = ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock1D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock1D(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out
   
 
    
class DeepSpectraCNN(nn.Module):
    def __init__(self, input_dim,mean,std,dropout=0.5,out_dims=1):
        super(DeepSpectraCNN, self).__init__()
        self.conv1d_dims = input_dim
        self.dropout=dropout
        self.mean = nn.Parameter(torch.tensor(mean).float(),requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(),requires_grad=False)
        self.out_dims=out_dims
               
        kernel_size = 7
        stride = 3
        padding = 3
               
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 8, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Inception modules
        self.inception2 = InceptionModule(8, 4)
        self.inception3 = InceptionModule(16,4)
        
        # Other layers
        self.flatten = nn.Flatten()
        flat_dim=((input_dim + 2 * padding - kernel_size) // stride + 1)
             
        self.fc1 = nn.Linear(16*flat_dim,64)
        self.fc2 = nn.Linear(64, self.out_dims)
               

    def forward(self, x):
        x = (x-self.mean)/self.std
        x = F.relu(self.conv1(x))
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.flatten(x)       
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x
    
# ###############################################################################


class ResidualBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def ResNet18_1D(**kwargs):
    return ResNet1D(ResidualBlock1D, [2, 2, 2, 2], **kwargs)

def ResNet34_1D(**kwargs):
    return ResNet1D(ResidualBlock1D, [3, 4, 6, 3], **kwargs)

def ResNet50_1D(**kwargs):
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], **kwargs)

def ResNet101_1D(**kwargs):
    return ResNet1D(Bottleneck1D, [3, 4, 23, 3], **kwargs)

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, in_channel=1, out_dims=1, zero_init_residual=False, head='linear',mean=0.0, std=1.0,dropout=0.5,inplanes=8):
        super(ResNet1D, self).__init__()
        self.in_planes = inplanes
        self.dropout=dropout
        self.mean = nn.Parameter(torch.tensor(mean).float(), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(), requires_grad=False)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.in_planes),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, inplanes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*inplanes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*inplanes, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*inplanes, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out_dims = out_dims


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResidualBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

        if head == 'linear':
            self.head = nn.Linear(8*inplanes * block.expansion, out_dims)
        elif head == 'mlp':
            dim_in = 8*inplanes * block.expansion
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, out_dims)
            )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_channels, stride))
            self.in_planes = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.head(out)
        return out
    
    
class FullyConvNet(nn.Module):
    def __init__(self, input_dims, mean, std,dropout=0.5, out_dims=1):
        super(FullyConvNet, self).__init__()

        # Layers dimensions
        self.input_dims = input_dims
        self.dropout=dropout
        self.out_dims = out_dims
        self.mean = nn.Parameter(torch.tensor(mean).float(), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(), requires_grad=False)

        # Convolutional layer
        self.conv1d_1 = nn.Conv1d(1, 2, kernel_size=9, stride=1)
        self.avg_1 = nn.AvgPool1d(2)
        self.conv1d_2 = nn.Conv1d(2, 2, kernel_size=7, stride=1)
        self.avg_2 = nn.AvgPool1d(2)
        self.conv1d_3 = nn.Conv1d(2, 4, kernel_size=7, stride=1)
        self.avg_3 = nn.AvgPool1d(2)
        self.conv1d_4 = nn.Conv1d(4, 8, kernel_size=5, stride=1)
        self.avg_4 = nn.AvgPool1d(2)
        self.conv1d_5 = nn.Conv1d(8, 12, kernel_size=3, stride=1)
        self.dp = nn.Dropout(self.dropout)
        self.head = nn.Conv1d(12, out_dims, kernel_size=1, stride=1)

    def forward(self, x):
        # Reshape input
        x = (x - self.mean) / self.std
        # Convolutional layer with ELU activation
        x = F.relu(self.conv1d_1(x))
        x = self.avg_1(x)
        x = F.relu(self.conv1d_2(x))
        x = self.avg_2(x)
        x = F.relu(self.conv1d_3(x))
        x = self.avg_3(x)
        x = F.relu(self.conv1d_4(x))
        x = self.avg_4(x)
        x = F.relu(self.conv1d_5(x))
        x = self.dp(x)
        x = self.head(x)
        x = F.adaptive_avg_pool1d(x,(1))
        return x[...,0]




# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        if dim % heads != 0 :
            raise ValueError(
                f"The hidden size {dim} is not a multiple of the number of attention "
                f"heads {heads}."
            )
        self.heads = heads
        self.attention_head_size = int(dim / heads)
        self.dim = dim
        # self.all_head_size = self.heads * self.attention_head_size # line remooved from the code of transformers because all_head_size always equals to dim...

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3) # shape : batch_size, heads_nb, patches_nb, attention_head_size
    
    def forward(self, x):
        key_layer = self.transpose_for_scores(self.key(x))
        value_layer = self.transpose_for_scores(self.value(x))
        query_layer = self.transpose_for_scores(self.query(x))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer =  torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        context_layer = context_layer.view(new_context_layer_shape) # reshape to the original dim : batch_size, patches_nb, dim

        return context_layer 

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads = heads, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT_1D(nn.Module):
    """ Please note that in the HuggingFace implementation of ViT there is a supplementary Dense layer and dropout just after the self-attention layer. 
    This was not implemented here.
    """
    def __init__(self, mean, std, seq_len, patch_size, dim_embed, trans_layers, heads, mlp_dim, out_dims, dropout = 0., emb_dropout = 0.):
        super().__init__()
        
        if (seq_len % patch_size) != 0 :
            self.crop = seq_len % patch_size
            self.seq_len = (seq_len - self.crop)
            self.crop = -self.crop
        else :
            self.crop = None

        self.out_dims=out_dims
        self.mean = nn.Parameter(torch.tensor(mean).float(), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(), requires_grad=False)

        num_patches = seq_len // patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size), # batch_size channel seq_length -> batch_size patches_nb patch_size, note that channel=1
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, dim_embed),
            nn.LayerNorm(dim_embed),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim_embed))
        self.cls_token = nn.Parameter(torch.randn(dim_embed))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim_embed, trans_layers, heads, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_embed),
            nn.Linear(dim_embed,  self.out_dims)
        )

    def forward(self, x):

        x = (x - self.mean) / self.std
        x = x[...,:self.crop]

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape # batch_size seq_length

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')
        out = self.mlp_head(cls_tokens)
        return out


# modified from AstroCLIP



class CrossAttentionHead(nn.Module):
    """Cross-attention head with dropout.

    This module is a single head of a cross-attention layer. It takes a query and a key
    tensor, computes the attention weights, and returns the weighted sum of the values
    tensor. The attention weights are also returned.

    :param embed_dim: dimensionality of the input tensors
    :param n_head: number of heads
    :param model_embed_dim: dimensionality of the model tensors
    :param dropout: amount of dropout
    """

    embed_dim: int
    n_head: int
    model_embed_dim: int
    dropout: float

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        model_embed_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_head,
            batch_first=True,
            kdim=model_embed_dim,
            vdim=model_embed_dim,
        )
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor):
        batch_size = x.shape[0]
        attentions = self.multihead_attn(
            query=self.query.repeat(batch_size, 1, 1),
            key=x,
            value=x,
            average_attn_weights=False,
        )[0]
        x = self.layernorm(self.dropout(attentions))
        return x, attentions[1]


class MLP(nn.Module):
    """A two-layer MLP.

    This uses a fully-connected layer to encode the input, then applies a non-linearity,
    then uses another fully-connected layer to decode back to the initial dimension, and
    finally applies (optional) dropout.

    :param in_features: size of input layer
    :param hidden_features: size of hidden layer
    :param activation: activation function to use after the expansion; default: GELU
    :param dropout: amount of dropout
    :param bias: whether to use bias in the layers
    """

    in_features: int
    hidden_features: int
    activation: Callable
    dropout: float
    bias: bool

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        activation: Optional[Callable] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.activation = activation if activation is not None else nn.GELU()
        self.dropout = dropout
        self.bias = bias

        self.encoder = nn.Linear(in_features, hidden_features, bias=bias)
        self.decoder = nn.Linear(hidden_features, in_features, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x


class SelfAttention_old(nn.Module):
    """Collection of self-attention heads.

    :param embedding_dim: total dimensionality of the model (equal to
        `head_size * num_heads`)
    :param num_heads: number of heads
    :param bias: whether to include bias terms
    :param dropout: amount of dropout; used both for the attention and for the residual
        pathways
    :param causal: if true, use causal self-attention
    """

    embedding_dim: int
    num_heads: int
    dropout: float
    uses_flash: bool

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool,
        dropout: float,
        bias: bool = True,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim should be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal

        # key, query, value projections for all heads, but in a batch
        self.attention = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)

        # output projection
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # regularization
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # flash attention makes GPU go brrrrr but support is only in PyTorch >= 2.0
        self.uses_flash = hasattr(F, "scaled_dot_product_attention")
        if not self.uses_flash:
            print("Using slow attention. Flash Attention requires PyTorch >= 2.0.")

            if self.causal:
                self.register_buffer("mask", torch.empty((1, 1, 0, 0), dtype=bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality
        B, T, C = x.shape
        if C != self.embedding_dim:
            raise ValueError(
                f"Expected input shape (..., {self.embedding_dim}, got {x.shape})"
            )

        # calculate and separate out query, key, values for all heads
        # each has shape (B, T, C)
        q, k, v = self.attention(x).split(self.embedding_dim, dim=2)

        # separate out head index and move it up next to the batch dimension
        # final shape (B, num_heads, T, head_size), where C = num_heads * head_size
        nh = self.num_heads
        hs = C // nh
        k = k.view(B, T, nh, hs).transpose(1, 2)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.uses_flash:
            # efficient attention using Flash Attention CUDA kernels
            dropout_p = self.dropout if self.training else 0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=self.causal
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

            if self.causal:
                # cache the causal mask, if we're using one
                if self.mask.shape[2] < T:
                    self.mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T) == 0
                att = att.masked_fill(self.bias[:, :, :T, :T], float("-inf"))

            att = F.softmax(att, dim=-1)
            att = self.attention_dropout(att)
            # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.residual_dropout(self.projection(y))
        return y


class TransformerBlock(nn.Module):
    """A transformer block, including layer norm, self-attention, another layer norm,
    and a two-layer MLP.

    :param embedding_dim: total dimensionality of the self-attention model (equal to
        `head_size * num_heads`)
    :param num_heads: number of self-attention heads
    :param bias: whether to include bias terms; used for layernorms, attention, and MLP
    :param dropout: amount of dropout; used for attention, resiudal pathway, and MLP
    :param causal: if true, use causal self-attention
    :param mlp_expansion: ratio between embedding dimension and side of MLP hidden layer
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool,
        dropout: float,
        bias: bool = True,
        mlp_expansion: int = 4,
    ):
        super().__init__()

        self.layernorm1 = LayerNorm(embedding_dim, bias=bias)
        self.attention = SelfAttention(
            embedding_dim, num_heads, bias=bias, dropout=dropout, causal=causal
        )
        self.layernorm2 = LayerNorm(embedding_dim, bias=bias)

        hidden_dim = mlp_expansion * embedding_dim
        self.mlp = MLP(embedding_dim, hidden_dim, nn.GELU(), dropout=dropout, bias=bias)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x


class LayerNorm(nn.Module):
    """Layer normalized with optional bias.

    This is based on PyTorch's :class:`~torch.nn.LayerNorm` module but is needed because
    PyTorch's version does not support disabling the bias.

    :param shape: shape of the input, following an arbitrary number of batch dimensions;
        that is, the input has dimensions `[d1, ..., dk, shape[0], ..., shape[-1]]`
    :param eps: value added to the denominator for numerical stability
    :param bias: whether to include a bias term
    :param dtype: data type to use for the parameters
    """

    normalized_shape: Tuple[int, ...]
    eps: float

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...], torch.Size],
        eps: float = 1e-5,
        bias: bool = True,
        dtype=None,
    ):
        super().__init__()

        self.eps = eps
        if isinstance(shape, numbers.Integral):
            self.normalized_shape = (shape,)
        else:
            self.normalized_shape = tuple(shape)

        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(shape)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

class SpecFormer(nn.Module):
    def __init__(
        self,
        mean,
        std,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int,
        out_dims: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean).float(), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(), requires_grad=False)
        self.max_len = max_len
        self.out_dims = out_dims
        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    causal=False,
                    dropout=dropout,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, out_dims, bias=True)

    def forward(self, x: Tensor):
        """Forward pass through the model."""
        x = torch.squeeze(x)
        x = x[...,None]
        x = (x - self.mean) / self.std
        t = x.shape[1]
        if t > self.max_len:
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.max_len}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)  # shape (t)

        # forward the GPT model itself
        data_emb = self.data_embed(x)  # to shape (b, t, embedding_dim)
        pos_emb = self.position_embed(pos)  # to shape (t, embedding_dim)

        x = self.dropout(data_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.final_layernorm(x)
        # average pooling over the sequence
        x = x.mean(dim=1)
        output = self.head(x)
        return output


def loadModel(model:dict, spec_dims:int, mean:torch.Tensor, std:torch.Tensor, out_dims:int):
    if model["name"] == "CuiNet":
        return CuiNet(spec_dims, mean=mean, std=std, out_dims=out_dims)
    elif model["name"] == "ViT_1D":
        # return ViT_1D(mean,std,spec_dims,patch_size=8,dim_embed=128,trans_layers=8,heads=8,mlp_dim=64,out_dims=out_dims)
        return ViT_1D(mean,std,spec_dims,model["PS"],model["DE"],model["TL"],model["HDS"],model["MLP"],out_dims=out_dims)
    else : 
        sys.exit("for now only CuiNet is available")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


class Darionet(nn.Module):
    def __init__(self, mean, std, filter_size, reg_beta, input_dims, out_dims=1, p=0.1):
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

    def forward(self, x,dp = True):
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
        # Dropout layer
        if dp :
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
    def __init__(self, input_dims, mean, std, dropout=0.1, out_dims=1):
        super(CuiNet, self).__init__()

        # Layers dimensions
        self.conv1d_dims = input_dims - 4  # size of spectrum - (kernel_size-1) is the size of the spectrum after first conv1D
        self.k_number = 1
        self.k_width = 5
        self.k_stride = 1
        self.fc1_dims = 36
        self.fc2_dims = 18
        self.fc3_dims = 12
        self.out_dims = out_dims
        self.dropout = ManualDropout(p=dropout)
        self.mean = nn.Parameter(torch.tensor(mean).float(), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(), requires_grad=False)

        # Convolutional layer
        self.conv1d = nn.Conv1d(1, 1, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv1d_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.out = nn.Linear(self.fc3_dims, self.out_dims)

        # Initialize weights with He normal
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x,dp = True):
        # Reshape input
        x = (x - self.mean) / self.std
        # Convolutional layer with ELU activation
        x = F.elu(self.conv1d(x))

        # Flatten the output from conv1d
        x = x.view(x.size(0), -1)

        # Fully connected layers with ELU activation
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        if dp:
            x = self.dropout(x)
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT_1D(nn.Module):
    def __init__(self, *, mean, std, seq_len, patch_size, dim_embed, trans_layers, heads, mlp_dim, out_dims, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
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
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim_embed),
            nn.LayerNorm(dim_embed),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim_embed))
        self.cls_token = nn.Parameter(torch.randn(dim_embed))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim_embed, trans_layers, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_embed),
            nn.Linear(dim_embed,  self.out_dims)
        )

    def forward(self, x):
        
        x = (x - self.mean) / self.std
        x = x[...,:self.crop]

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')
        out = self.mlp_head(cls_tokens)
        return out
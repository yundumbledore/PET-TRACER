import torch
import torch.nn as nn
from utility import *

class OneDUnet(nn.Module):
  # A time-dependent Unet

    def __init__(self, x_dim, y_dim, embed_dim, channels=[32, 64, 128, 256], embedy = 'Linear', sigma_data=1.0):
        super().__init__()
        self.sigma_data = sigma_data
        # Embed time as gaussian random features
        self.embedt = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                    nn.Linear(embed_dim, embed_dim))
        
        if embedy == 'Linear':
            self.embedy = nn.Linear(y_dim, embed_dim)
        elif embedy == 'Conv1D':
            self.embedy = Conv1DEmbedding(input_size=y_dim, embed_dim=embed_dim)
        elif embedy == 'BiGRU':
            self.embedy = BiGRUEmbedding(input_size=y_dim, embed_dim=embed_dim)
        elif embedy == 'LSTM':
            self.embedy = LSTMEmbedding(input_size=y_dim, embed_dim=embed_dim)
        elif embedy == 'GRU':
            self.embedy = GRUEmbedding(input_size=y_dim, embed_dim=embed_dim)
        elif embedy == 'MLP':
            self.embedy = nn.Sequential(
                nn.Linear(y_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        self.encodex = nn.Linear(x_dim, 28)
        self.decodex = nn.Linear(28, x_dim)

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv1d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv1d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv1d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose1d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose1d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose1d(channels[0] + channels[0], 1, 3, stride=1)

        # Activation functions
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, y, t):
        embedt = self.act(self.embedt(t))
        embedy = self.act(self.embedy(y))
        embed = embedt + embedy
        _x = self.encodex(x)
        # Encoding path
        h1 = self.conv1(_x)
        ## concatenate t with x
        h1 += self.dense1(embed)
        ## group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = self.decodex(h)

        # Compute c_skip(t) and c_out(t)
        c_skip = self.sigma_data**2 / (t**2 + self.sigma_data**2)
        c_out = self.sigma_data * t / torch.sqrt(t**2 + self.sigma_data**2)
        
        # Enforce boundary condition using skip connections
        return c_skip.unsqueeze(1).unsqueeze(2) * x + c_out.unsqueeze(1).unsqueeze(2) * h
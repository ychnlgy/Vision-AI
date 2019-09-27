import torch


class Shortcut(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        assert out_channels % 4 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = torch.nn.ReLU()
        self.bnorm = torch.nn.BatchNorm2d(out_channels)
        self.conv1 = self._create_conv((0, 0, 0, 0))
        self.conv2 = self._create_conv((-1, 1, 0, 0))
        self.conv3 = self._create_conv((0, 0, -1, 1))
        self.conv4 = self._create_conv((-1, 1, -1, 1))

    def forward(self, X):
        h0 = self.act(X)
        h1 = self.conv1(h0)
        h2 = self.conv2(h0)
        h3 = self.conv3(h0)
        h4 = self.conv4(h0)
        hx = torch.cat([h1, h2, h3, h4], axis=1)
        return self.bnorm(hx)

    def _create_conv(self, padding):
        return torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            1,
            padding=padding,
            stride=2,
            bias=False
        )


class ResBlock(torch.nn.Module):

    def __init__(self, block, shortcut=None):
        self.block = block
        self.shortcut = [shortcut, torch.nn.Sequential()][shortcut is None]

    def forward(self, X):
        return self.shortcut(X) + self.block(X)

import torch

import vision_ai


class Unet(torch.nn.Module):

    def __init__(self, layers, shortcuts):
        assert layers
        assert len(layers) == len(shortcuts) + 1
        self.layers = torch.nn.ModuleList(layers)
        self.shortcuts = torch.nn.ModuleList(shortcuts)

    def forward(self, X0):
        hidden = []
        X = X0
        for layer in self.layers:
            X = layer(X)
            hidden.append(X)
        n = len(hidden)
        for d in range(n - 1):
            i = n - d
            j = i - 1
            Xa = hidden[i]
            Xb = hidden[j]
            Na, Ca, Wa, Ha = Xa.size()
            Nb, Cb, Wb, Hb = Xb.size()
            assert Na == Nb
            assert Wa * 2 == Wb
            assert Ha * 2 == Hb
            Xp = vision_ai.tools.local_repeat(Xa, factor=2)
            Xh = torch.cat([Xp, Xb], dim=1)
            hidden[j] = self.shortcuts[j](Xh)
        return hidden[0]

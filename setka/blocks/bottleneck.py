import torch

class BottleNeck(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 bottleneck,
                 out_channels,
                 normalization=torch.nn.BatchNorm2d,
                 activation=torch.nn.ReLU,
                 mode='identity',
                 groups=1,
                 kernel_size=3):
        
        super().__init__()
        
        self.module = [
            normalization(in_channels),
            torch.nn.Conv2d(in_channels, bottleneck, 1),
            normalization(bottleneck),
            torch.nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, groups=groups, padding=(kernel_size - 1) // 2),
            normalization(bottleneck),
            torch.nn.Conv2d(bottleneck, out_channels, 1),
            activation()
        ]

        if mode == 'identity':
            pass
        elif mode == 'upsample':
            self.module.append(torch.nn.UpsamplingBilinear2d(scale_factor=2.0))
        elif mode == 'downsample':
            self.module.append(torch.nn.UpsamplingBilinear2d(scale_factor=0.5))

        self.module = torch.nn.Sequential(*self.module)


    def __call__(self, input):
        return self.module(input)
import torch

class UpDown(torch.nn.Module):
    def __init__(self, backbone, normalization=torch.nn.BatchNorm2d, activation=torch.nn.ReLU):
        super().__init__()
        self.backbone = backbone

        self.n_features = []

        self.horizontal1 = torch.nn.ModuleList([])
        self.vertical1 = []

        self.horizontal2 = torch.nn.ModuleList([])
        self.vertical2 = []

        for n_features in self.backbone.n_features:
            self.horizontal1.append(
                torch.nn.Sequential(
                        ResidualBlock(
                            BottleNeck(
                                n_features, 
                                n_features * 2, 
                                n_features, 
                                groups=n_features*2))
                    )
                )

            self.horizontal2.append(
                torch.nn.Sequential(
                        ResidualBlock(
                            BottleNeck(
                                n_features, 
                                n_features * 2, 
                                n_features, 
                                groups=n_features*2))
                    )
                )
            self.n_features.append(n_features)
            
        for index in reversed(range(len(self.backbone.n_features) - 1)):
            out_features = self.backbone.n_features[index]
            in_features = self.backbone.n_features[index + 1]
            self.vertical1.append(
                torch.nn.Sequential(
                    normalization(in_features), 
                    torch.nn.Conv2d(in_features, out_features, 1), 
                    activation(),
                    torch.nn.UpsamplingBilinear2d(scale_factor=2))
                )
            
            self.vertical2.append(
                torch.nn.Sequential(
                    normalization(out_features), 
                    torch.nn.Conv2d(out_features, in_features, 1), 
                    activation(),
                    torch.nn.UpsamplingBilinear2d(scale_factor=0.5))
                )
        self.vertical1 = torch.nn.ModuleList(self.vertical1[::-1])
        self.vertical2 = torch.nn.ModuleList(self.vertical2[::-1])


    def __call__(self, input):
        results = input
        for index in range(len(input)):
            results[index] = self.horizontal1[index](results[index])

        for index in reversed(range(len(input) - 1)):
            results[index] = results[index] + self.vertical1[index](results[index + 1])

        for index in range(len(input)):
            results[index] = self.horizontal2[index](results[index])

        for index in range(len(input) - 1):
            results[index + 1] = results[index + 1] + self.vertical2[index](results[index])

        return results
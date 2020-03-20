import efficientnet_pytorch
import torch
import re
import torchvision

class BackBone(torch.nn.Module):
    def __init__(self, backbone, pretrained=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.n_features = []

        if re.match('efficientnet*', backbone):
            if pretrained:
                model = efficientnet_pytorch.EfficientNet.from_pretrained(
                    backbone,
                    advprop=True)
            else:
                model = efficientnet_pytorch.EfficientNet.from_name(
                    backbone)

            self.layers.append(torch.nn.Sequential())

            layer = [
                    model._conv_stem,
                    model._bn0,
                    model._swish]

            for module in model._blocks._modules:
                if model._blocks._modules[module]._depthwise_conv.stride == [2, 2]:
                    self.layers.append(torch.nn.Sequential(*layer))
                    layer = []
                layer.append(model._blocks._modules[module])

            self.layers.append(torch.nn.Sequential(*layer))

            self.classifier = torch.nn.Sequential(
                model._conv_head,
                model._bn1,
                model._swish,
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                model._dropout,
                model._fc
            )


        elif (
            re.match('resnet*', backbone) or
            re.match('resnext*', backbone) or
            re.match('wide_resnet*', backbone)):

            model = getattr(torchvision.models, backbone)(pretrained=pretrained)

            self.layers.append(torch.nn.Sequential())

            self.layers.append(
                torch.nn.Sequential(
                    model.conv1,
                    model.bn1,
                    model.relu
                )
            )

            self.layers.append(
                torch.nn.Sequential(model.maxpool, model.layer1))
            for key in ['layer2', 'layer3', 'layer4']:
                self.layers.append(model._modules[key])

            self.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                model.fc
            )

        elif re.match('mobilenet_v2', backbone):
            model = torchvision.models.mobilenet_v2(pretrained=True)

            self.layers.append(torch.nn.Sequential())

            layer = [model.features[0],
                    model.features[1]]
            for index in range(2, len(model.features) - 1):
                if model.features[index].conv[1][0].stride == (2, 2):
                    self.layers.append(torch.nn.Sequential(*layer))
                    layer = []
                layer.append(model.features[index])
            self.layers.append(torch.nn.Sequential(*layer))

            self.classifier = torch.nn.Sequential(
                model.features[-1],
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                model.classifier
            )


        elif re.match('densenet*', backbone):
            model = getattr(torchvision.models, backbone)(pretrained=pretrained).eval()

            self.layers.append(torch.nn.Sequential())

            self.layers.append(
                torch.nn.Sequential(
                    model.features.conv0,
                    model.features.norm0,
                    model.features.relu0
                )
            )

            self.layers.append(
                torch.nn.Sequential(
                    model.features.pool0,
                    model.features.denseblock1
                )
            )

            self.layers.append(
                torch.nn.Sequential(
                    model.features.transition1,
                    model.features.denseblock2
                )
            )

            self.layers.append(
                torch.nn.Sequential(
                    model.features.transition2,
                    model.features.denseblock3
                )
            )

            self.layers.append(
                torch.nn.Sequential(
                    model.features.transition3,
                    model.features.denseblock4,
                    model.features.norm5
                )
            )

            self.classifier = torch.nn.Sequential(
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                model.classifier
            )

        elif re.match('vgg*', backbone):
            model = getattr(torchvision.models, backbone)(pretrained=pretrained)

            layer = []
            for index in range(len(model.features)):
                if isinstance(model.features[index], torch.nn.MaxPool2d):
                    layer = torch.nn.Sequential(*layer)
                    self.layers.append(layer)
                    layer = []

                layer.append(model.features[index])

            layer = torch.nn.Sequential(*layer)
            self.layers.append(layer)

            self.classifier = torch.nn.Sequential(
                model.avgpool,
                torch.nn.Flatten(),
                model.classifier
            )

        self.eval()
        x = torch.zeros(1, 3, 224, 224)

        for layer in self.layers:
            x = layer(x)
            self.n_features.append(x.shape[1])

    def __call__(self, input):
        x = input
        features = []

        for index in range(len(self.layers)):
            x = self.layers[index](x)
            features.append(x)

        return features


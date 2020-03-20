import efficientnet_pytorch
import setka.blocks
import torch
import re
import torchvision

x = torch.zeros([2, 3, 512, 512])

for backbone in ['vgg11', 
                'densenet121',
                'wide_resnet101_2',
                'resnext50_32x4d',
                'mobilenet_v2',
                'efficientnet-b0']:

    bb = setka.blocks.BackBone(backbone, pretrained=False)
    bb = setka.blocks.BackBone(backbone, pretrained=True)
    if re.match('efficientnet*', backbone):
        model = efficientnet_pytorch.EfficientNet.from_pretrained(backbone, advprop=True)
    else:
        model = getattr(torchvision.models, backbone)(pretrained=True)

    bb.eval()
    model.eval()

    levels = bb(x)
    res = model(x)

    for level_index in range(len(levels)):
        assert(levels[level_index].shape[-2:] == (x.shape[-2] // 2**level_index, x.shape[-1] // 2**level_index))

    assert((res - bb.classifier(levels[-1])).abs().sum() < 1.0e-4)
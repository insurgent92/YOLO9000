import torch
from torch import nn
from base import BaseModel

__all__ = [
    'darknet', 'darknet19'
]

class Darknet(BaseModel):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(Darknet, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 1000, kernel_size=1, padding=1),
            nn.AvgPool2d(num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, self.num_classes)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            kernel_size = int(v.split('_')[0])
            out_channels = int(v.split('_')[1])
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels
    return nn.Sequential(*layers)

cfg = {
    '19': ['3_32', 'M', '3_64', 'M', '3_128', '1_64', '3_128', 'M', '3_256', '1_128', '3_256', 'M', 
    '3_512', '1_256', '3_512', '1_256', '3_512', 'M', '3_1024', '1_512', '3_1024', '1_512', '3_1024']
}

def darknet19(pretrained=False, **kwargs):
    """Dartnet 19-layer model (configuration "19")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = Darknet(make_layers(cfg['19']), **kwargs)
    # if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def darknet19_bn(pretrained=False, **kwargs):
    """Dartnet 19-layer model (configuration "19")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = Darknet(make_layers(cfg['19'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model
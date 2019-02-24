from collections import OrderedDict
import torch.nn as nn
from torchsummary.torchsummary import summary
from base import BaseModel
from model import backbone

class Yolo9000(BaseModel):
    def __init__(self,
                 num_classes=20,
                 num_prior_boxes=5,
                 prior_boxes=None,
                 device="cpu",
                 input_size=(416, 416),
                 pretrained_backbone=True):
        assert isinstance(num_prior_boxes, int)
        assert num_prior_boxes == len(prior_boxes)
        for box in prior_boxes:
            assert len(box) == 2
            for element in box:
                assert isinstance(element, float)

        super(Yolo9000, self).__init__()

        self.device = device
        self.input_size = input_size
        self.num_coordinates = 5
        self.num_prior_boxes = num_prior_boxes
        self.num_classes = num_classes
        self.output_shape = (self.num_coordinates + self.num_classes) * self.num_prior_boxes
        
        self.darknet19 = backbone.Darknet19(pretrained=pretrained_backbone)

        self.backbone = self.darknet19.features

        self.layer7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, self.output_shape, kernel_size=1, stride=1, padding=0))

        self.detector = nn.Sequential(OrderedDict([('layer7', self.layer7)]))

    def forward(self, *input):
        """
        output structure
        [objness, tx, ty, tw, th, c1, ..., cn] x 5 = 125

        shape : [batch, n, n, 125]
        """
        x = input[0]
        out = self.backbone(x)
        out = self.detector(out)
        out = self.post_processing(out)

        return out

    def post_processing(self, out):

        for box_idx in range(self.num_prior_boxes):
            idx = box_idx * (self.num_coordinates + self.num_classes)

            out[:, idx + 1, :, :] = nn.Sigmoid()(out[:, idx + 1, :, :])
            out[:, idx + 2, :, :] = nn.Sigmoid()(out[:, idx + 2, :, :])
            out[:, idx + 3, :, :] = out[:, idx + 3, :, :].exp()
            out[:, idx + 4, :, :] = out[:, idx + 4, :, :].exp()
            out[:, idx + 5:idx + 5 + self.num_classes, :, :] = \
                nn.Softmax(dim=1)(out[:, idx + 5:idx + 5 + self.num_classes, :, :])

        return out

    def summary(self):
        summary(self, input_size=(3, self.input_size[0], self.input_size[1]), device=self.device)

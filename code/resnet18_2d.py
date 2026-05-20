import torch
import torch.nn as nn
import torchvision.models as models


class CNN_2D(nn.Module):
    def __init__(self, num_classes=9, pretrained=True):
        super(CNN_2D, self).__init__()
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        # Modify first conv to accept single-channel input
        # (we will replicate grayscale to 3 channels in data loader if needed,
        #  but here we adapt conv1 for 1 channel to keep it explicit)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Average the pretrained 3-channel weights into 1 channel
            with torch.no_grad():
                resnet.conv1.weight = nn.Parameter(
                    resnet.conv1.weight.mean(dim=1, keepdim=True)
                )
        # Remove final fc
        self.sharedNet = nn.Sequential(*list(resnet.children())[:-1])
        self.cls_fc = nn.Linear(512, num_classes)

    def forward(self, source):
        feature = self.sharedNet(source)
        feature = feature.view(feature.size(0), -1)
        source = self.cls_fc(feature)
        return source, feature

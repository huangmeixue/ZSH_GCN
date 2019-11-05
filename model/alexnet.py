# -*- coding:utf-8 -*-

import torchvision
import torch.nn as nn


class AlexNet(nn.Module):
    """在自己的数据集上微调AlexNet(修改最后一层输出类别数量)
    Parameters
        pretrained: bool
        True: 加载预训练模型; False: 加载未训练模型
        num_classes: int
        类别数量
    Returns (此处修改成提取fc7层特征)
        alexnet_model: model
        CNN模型
    """
    def __init__(self,  pretrained=True, num_classes=None):
        super(AlexNet, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features

        fc1 = nn.Linear(256 * 6 * 6, 4096)
        fc1.weight = model.classifier[1].weight
        fc1.bias = model.classifier[1].bias

        fc2 = nn.Linear(4096, 4096)
        fc2.weight = model.classifier[4].weight
        fc2.bias = model.classifier[4].bias

        self.classifier = nn.Sequential(
            nn.Dropout(),
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc2,
            #nn.ReLU(inplace=True),
            #nn.Linear(4096, num_classes),
        )
        ## 另一种方法
        #model = torchvision.models.alexnet(pretrained=pretrained)
        #classifier = nn.Sequential(*list(model.classifier.children())[:-1])  #fc2-ReLU feature
        #classifier = nn.Sequential(*list(model.classifier.children())[:-2])  #fc2 feature
        #model.classifier = classifier
        #return model
    def forward(self, input_x):
        maxpooling5_feature = self.features(input_x).view(input_x.size(0), 256 * 6 * 6)
        fc7_faeture = self.classifier(maxpooling5_feature)
        return fc7_faeture
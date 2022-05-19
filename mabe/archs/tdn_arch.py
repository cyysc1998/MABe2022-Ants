# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn


import torch
import torch.nn as nn
import torch.nn.functional as F
from mabe.archs.tdn_base_arch import ConsensusModule, fbresnet50, fbresnet101
from torch.nn.init import constant_, normal_


class TDN_Net(nn.Module):

    def __init__(self,resnet_model,resnet_model1,apha,belta, num_segments, consensus_type):
        super(TDN_Net, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)
        
        # implement conv1_5 and inflate weight 
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * 4,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = nn.Sequential(nn.Conv2d(12,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels

        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 =nn.Sequential(*list(resnet_model1.children())[4])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avg_diff = nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc = list(resnet_model.children())[8]
        self.apha = apha
        self.belta = belta
        self.num_segments = num_segments
        self.consensus = ConsensusModule(consensus_type)

    def forward(self, x):
        x1, x2, x3, x4, x5 = x[:,0:3,:,:], x[:,3:6,:,:], x[:,6:9,:,:], x[:,9:12,:,:], x[:,12:15,:,:]
        x_c5 = self.conv1_5(self.avg_diff(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],1).view(-1,12,x2.size()[2],x2.size()[3])))
        x_diff = self.maxpool_diff(1.0/1.0*x_c5)
        
        temp_out_diff1 = x_diff 
        x_diff = self.resnext_layer1(x_diff)

        x = self.conv1(x3)
        x = self.bn1(x)
        x = self.relu(x)
        #fusion layer1
        x = self.maxpool(x)
        temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:])
        x = self.apha*x + self.belta*temp_out_diff1
        #fusion layer2
        x = self.layer1_bak(x)
        x_diff = F.interpolate(x_diff, x.size()[2:])
        x = self.apha*x + self.belta*x_diff
        
        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # fusion
        x = x.view((-1, self.num_segments) + x.size()[1:])
        x = self.consensus(x).squeeze(1)

        return x

def tdn_net(base_model=None,num_segments=8,pretrained=True, consensus_type='avg', **kwargs):
    if("50" in base_model):
        resnet_model = fbresnet50(num_segments, pretrained)
        resnet_model1 = fbresnet50(num_segments, pretrained)
    else:
        resnet_model = fbresnet101(num_segments, pretrained)
        resnet_model1 = fbresnet101(num_segments, pretrained)

    if(num_segments == 8):
        model = TDN_Net(resnet_model,resnet_model1,apha=0.5,belta=0.5, num_segments=num_segments, consensus_type=consensus_type)
    else:
        model = TDN_Net(resnet_model,resnet_model1,apha=0.75,belta=0.25, num_segments=num_segments, consensus_type=consensus_type)

    model.last_layer_name = 'fc'
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    
    return model


class TDN(nn.Module):
    def __init__(self, opt):
        super(TDN, self).__init__()
        base_model = opt["base_model"]
        num_segments = opt["num_segments"]
        pretrained = opt["pretrained"]
        consensus_type = opt["consensus_type"]
        out_emb_size = opt["out_emb_size"]

        self.base_model = tdn_net(base_model, num_segments=num_segments, pretrained=pretrained, consensus_type=consensus_type)
        
        # MLP projector
        n_features = self.base_model.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, out_emb_size, bias=False),
        )
        # temperature
        self.temperature = nn.Parameter(torch.ones(()), requires_grad=True)

    def forward(self, x1, x2):
        h1, h2 = self.base_model(x1), self.base_model(x2)
        z1, z2 = self.fc(h1), self.fc(h2)
        return z1, z2 


if __name__ == '__main__':
    batch_size = 4
    num_segments = 3
    tdn = tdn_net('resnet50', num_segments, pretrained=False)
    x = torch.rand(batch_size * num_segments, 15, 224, 224)
    y = tdn(x)
    print(y.shape)

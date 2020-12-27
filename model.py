import torch.nn as nn
import torch.nn.functional as F
import torch
from efficientnet_pytorch import EfficientNet
from src.utils import *

def add_features(feature_list):
    features = []

    for feat in feature_list:
        if feat[-1] == 'conv':
            features += [conv_block(feat[0], feat[1])]
        
        elif feat[-1] == 'convnew':
            features += [conv_block(feat[0], feat[1], feat[2], feat[3], feat[4], mode=feat[-1])]
        
        elif feat[-1] == 'convnew_nobatch':
            features += [conv_block(feat[0], feat[1], feat[2], feat[3], feat[4], mode=feat[-1])]
        
        elif feat[-1] == 'inception':
            features += [inception_module(feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], feat[6])]

        elif feat[-1] == 'inception_nobatch':
            features += [inception_nbatch_module(feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], feat[6])]
        
        elif feat[-1] == 'residual_block':
            sub_features = add_features(feat[:-1])
            residual_block = nn.Sequential(*sub_features)
            features += [residual_block_module(residual_block)]
        
        elif feat == 'MaxP2_2':
            features += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        elif feat == 'MaxP3_2':
            features += [nn.MaxPool2d(kernel_size=3, stride=2)]
        
        elif feat == 'MaxP3_2_1':
            features += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        
        elif feat == 'MaxP4_4':
            features += [nn.MaxPool2d(kernel_size=4, stride=4)]
        
        elif feat == 'AvgP_7': 
            features += [nn.AvgPool2d(kernel_size=7, stride=7)]

        elif feat == 'AvgP_6': 
            features += [nn.AvgPool2d(kernel_size=6, stride=6)]
        
        elif feat == 'AvgP_4': 
            features += [nn.AvgPool2d(kernel_size=4, stride=4)]

        else:
            print("Not implemented")
            exit()
    
    return features
class residual_block_module(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        out = self.module(x) + x
        return out

class inception_nbatch_module(nn.Module):
    def __init__(self, in_channels, conv_dim_3_3_mid, conv_dim_5_5_mid, conv_dim_1_1_out, conv_dim_3_3_out, conv_dim_5_5_out, max_dim_out):
        super(inception_nbatch_module,self).__init__()
        self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels, conv_dim_1_1_out, 1, 1),
                                    nn.ReLU())
        self.conv_1_3 = nn.Sequential(nn.Conv2d(in_channels, conv_dim_3_3_mid, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(conv_dim_3_3_mid, conv_dim_3_3_out, 3, 1, 1),
                                    nn.ReLU())
        self.conv_1_5 = nn.Sequential(nn.Conv2d(in_channels, conv_dim_5_5_mid, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(conv_dim_5_5_mid, conv_dim_5_5_out, 5, 1, 2),
                                    nn.ReLU())
        self.conv_max = nn.Sequential(nn.MaxPool2d(3, 1, 1),
                                    nn.Conv2d(in_channels, max_dim_out, 1, 1))
    
    def forward(self, x):
        conv_1_1_out = self.conv_1_1(x) 
        conv_1_3_out = self.conv_1_3(x) 
        conv_1_5_out = self.conv_1_5(x) 
        conv_max_out = self.conv_max(x)
        module_out = torch.cat([conv_1_1_out, conv_1_3_out, conv_1_5_out, conv_max_out], 1)

        return module_out 
class inception_module(nn.Module):
    def __init__(self, in_channels, conv_dim_3_3_mid, conv_dim_5_5_mid, conv_dim_1_1_out, conv_dim_3_3_out, conv_dim_5_5_out, max_dim_out):
        super(inception_module,self).__init__()
        self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels, conv_dim_1_1_out, 1, 1),
                                    nn.BatchNorm2d(conv_dim_1_1_out),
                                    nn.ReLU())
        self.conv_1_3 = nn.Sequential(nn.Conv2d(in_channels, conv_dim_3_3_mid, 1, 1),
                                    nn.BatchNorm2d(conv_dim_3_3_mid),
                                    nn.ReLU(),
                                    nn.Conv2d(conv_dim_3_3_mid, conv_dim_3_3_out, 3, 1, 1),
                                    nn.BatchNorm2d(conv_dim_3_3_out),
                                    nn.ReLU())
        self.conv_1_5 = nn.Sequential(nn.Conv2d(in_channels, conv_dim_5_5_mid, 1, 1),
                                    nn.BatchNorm2d(conv_dim_5_5_mid),
                                    nn.ReLU(),
                                    nn.Conv2d(conv_dim_5_5_mid, conv_dim_5_5_out, 5, 1, 2),
                                    nn.BatchNorm2d(conv_dim_5_5_out),
                                    nn.ReLU())
        self.conv_max = nn.Sequential(nn.MaxPool2d(3, 1, 1),
                                    nn.Conv2d(in_channels, max_dim_out, 1, 1),
                                    nn.BatchNorm2d(max_dim_out),)
    
    def forward(self, x):
        conv_1_1_out = self.conv_1_1(x) 
        conv_1_3_out = self.conv_1_3(x) 
        conv_1_5_out = self.conv_1_5(x) 
        conv_max_out = self.conv_max(x)
        module_out = torch.cat([conv_1_1_out, conv_1_3_out, conv_1_5_out, conv_max_out], 1)

        return module_out 

""" Optional conv block """
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, mode='conv'):

    if mode == 'conv':
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        )
    
    if mode == 'convnew':
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        )
    
    if mode == 'convnew_nobatch':
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        )

    elif mode == 'point':
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        )

    elif mode == 'depth':
        return nn.Sequential(
        nn.Conv2d(in_channels, in_channels*out_channels, 3, padding=1, groups=in_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        )

    else:
        print("Not implemented conv block")
        exit()
""" Define your own model """
# AlexNet base
AlexRef = [[3, 64, 11, 4, 2, 'convnew'], 'MaxP3_2', [64, 192, 5, 1, 2, 'convnew'], 'MaxP3_2', [192, 384, 3, 1, 1, 'convnew'], [384, 256, 3, 1, 1, 'convnew'], [256, 256, 3, 1, 1, 'convnew'], 'MaxP3_2']
AlexRefBrief0 = [[3, 64, 11, 4, 2, 'convnew'], 'MaxP3_2', [64, 192, 5, 1, 2, 'convnew'], 'MaxP3_2', [192, 384, 3, 1, 1, 'convnew'], [384, 256, 3, 1, 1, 'convnew'], 'MaxP3_2', [256, 128, 3, 1, 1, 'convnew'], 'MaxP3_2']
AlexRefBrief1 = [[3, 64, 11, 4, 2, 'convnew'], 'MaxP3_2', [64, 192, 5, 1, 2, 'convnew'], 'MaxP3_2', [192, 384, 3, 1, 1, 'convnew'], [384, 128, 3, 1, 1, 'convnew'], [128, 64, 3, 1, 1, 'convnew'], 'MaxP3_2']

AlexRefNbatch = [[3, 64, 11, 4, 2, 'convnew_nobatch'], 'MaxP3_2', [64, 192, 5, 1, 2, 'convnew_nobatch'], 'MaxP3_2', [192, 384, 3, 1, 1, 'convnew_nobatch'], \
                [384, 256, 3, 1, 1, 'convnew_nobatch'], [256, 256, 3, 1, 1, 'convnew_nobatch'], 'MaxP3_2']
AlexRefBriefNbatch0 = [[3, 64, 11, 4, 2, 'convnew_nobatch'], 'MaxP3_2', [64, 192, 5, 1, 2, 'convnew_nobatch'], 'MaxP3_2', [192, 384, 3, 1, 1, 'convnew_nobatch'], \
                        [384, 256, 3, 1, 1, 'convnew_nobatch'], 'MaxP3_2', [256, 128, 3, 1, 1, 'convnew'], 'MaxP3_2']
AlexRefBriefNbacth1 = [[3, 64, 11, 4, 2, 'convnew_nobatch'], 'MaxP3_2', [64, 192, 5, 1, 2, 'convnew_nobatch'], 'MaxP3_2', [192, 384, 3, 1, 1, 'convnew_nobatch'], \
                        [384, 128, 3, 1, 1, 'convnew_nobatch'], [128, 64, 3, 1, 1, 'convnew_nobatch'], 'MaxP3_2']

# AlexNet 오류 있는 모델
AlexBriefMain = [[3, 64, 3, 1, 1, 'convnew'], 'MaxP3_2', [64, 192, 3, 1, 1, 'convnew'], 'MaxP3_2', [192, 384, 3, 1, 1, 'convnew'], [384, 128, 3, 1, 1, 'convnew'], 'MaxP3_2', [128, 64, 3, 1, 1, 'convnew'], 'AvgP_6'] # same(alexbrief0)

AlexBrief0 = [[3, 64, 11, 4, 2, 'conv'], 'MaxP3_2', [64, 192, 5, 1, 2, 'conv'], 'MaxP3_2', [192, 384, 'conv'], [384, 128, 'conv'], 'MaxP3_2', [128, 64, 'conv'], 'AvgP_6'] # best
AlexBrief1 = [[3, 64, 11, 4, 2, 'conv'], 'MaxP3_2', [64, 192, 5, 1, 2, 'conv'], 'MaxP3_2', [192, 384, 'conv'], [384, 256, 'conv'], 'MaxP3_2', [256, 64, 'conv'], 'AvgP_6']

AlexBrief2 = [[3, 64, 11, 4, 2, 'conv'], 'MaxP3_2', [64, 192, 5, 1, 2, 'conv'], 'MaxP3_2', [192, 256, 'conv'], [256, 128, 'conv'], 'MaxP3_2', [128, 64, 'conv'], 'AvgP_6']
AlexBrief3 = [[3, 64, 11, 4, 2, 'conv'], 'MaxP3_2', [64, 192, 5, 1, 2, 'conv'], 'MaxP3_2', [192, 128, 'conv'], [128, 64, 'conv'], 'MaxP3_2', [64, 32, 'conv'], 'AvgP_6']

Alex = [[3, 64, 11, 4, 2, 'conv'], 'MaxP3_2', [64, 192, 5, 1, 2, 'conv'], 'MaxP3_2', [192, 384, 'conv'], [384, 256, 'conv'], 'MaxP3_2', [256, 128, 'conv'], 'AvgP_6']
AlexDeep = [[3, 64, 11, 4, 2, 'conv'], 'MaxP3_2', [64, 192, 5, 1, 2, 'conv'], 'MaxP3_2', [192, 384, 'conv'], [384, 384, 'conv'], 'MaxP3_2', [384, 256, 'conv'], [256, 256, 'conv'], 'AvgP_6']

AlexChangePool = [[3, 64, 11, 4, 2, 'conv'], 'MaxP2_2', [64, 192, 5, 1, 2, 'conv'], 'MaxP2_2', [192, 384, 'conv'], [384, 256, 'conv'], 'MaxP2_2', [256, 128, 'conv'], 'AvgP_6']
AlexChangePoolDeep = [[3, 64, 11, 4, 2, 'conv'], 'MaxP2_2', [64, 192, 5, 1, 2, 'conv'], 'MaxP2_2', [192, 384, 'conv'], [384, 384, 'conv'], 'MaxP2_2', [384, 256, 'conv'], [256, 128, 'conv'], 'AvgP_6']

Normal = [[3, 64, 'conv'], 'MaxP2_2', [64, 64, 'conv'], 'MaxP2_2', [64, 64, 'conv'], 'MaxP2_2', [64, 64, 'conv'], 'MaxP2_2', [64, 64, 'conv'], 'MaxP2_2', [64, 64, 'conv'], 'MaxP2_2']
Brief = [[3, 64, 'conv'], 'MaxP2_2', [64, 64, 'conv'], 'MaxP2_2', [64, 64, 'conv'], 'MaxP2_2', [64, 64, 'conv'], 'MaxP2_2']

# Googlenet base
GoogleRef = [[3, 64, 7, 2, 3, 'convnew'], 'MaxP3_2_1', [64, 192, 3, 1, 1, 'convnew'], 'MaxP3_2_1', [192, 96, 16, 64, 128, 32, 32, 'inception'], [256, 128, 32, 128, 192, 96, 64, 'inception'], \
            'MaxP3_2_1', [480, 96, 16, 192, 208, 48, 64, 'inception'], [512, 112, 24, 160, 224, 64, 64, 'inception'], [512, 128, 24, 128, 256, 64, 64, 'inception'], \
            [512, 144, 32, 112, 288, 64, 64, 'inception'], [528, 160, 32, 256, 320, 128, 128, 'inception'], 'MaxP3_2_1', [832, 160, 32, 256, 320, 128, 128, 'inception'], \
            [832, 192, 48, 384, 384, 128, 128, 'inception'], 'AvgP_7']

GoogleRefNbatch = [[3, 64, 7, 2, 3, 'convnew_nobatch'], 'MaxP3_2_1', [64, 192, 3, 1, 1, 'convnew_nobatch'], 'MaxP3_2_1', [192, 96, 16, 64, 128, 32, 32, 'inception_nobatch'], [256, 128, 32, 128, 192, 96, 64, 'inception_nobatch'], \
                'MaxP3_2_1', [480, 96, 16, 192, 208, 48, 64, 'inception_nobatch'], [512, 112, 24, 160, 224, 64, 64, 'inception_nobatch'], [512, 128, 24, 128, 256, 64, 64, 'inception_nobatch'], \
                [512, 144, 32, 112, 288, 64, 64, 'inception_nobatch'], [528, 160, 32, 256, 320, 128, 128, 'inception_nobatch'], 'MaxP3_2_1', [832, 160, 32, 256, 320, 128, 128, 'inception_nobatch'], \
                [832, 192, 48, 384, 384, 128, 128, 'inception_nobatch'], 'AvgP_7']

# Residual Googlenet
ResGoogleRef = [[3, 64, 7, 2, 3, 'convnew'], 'MaxP3_2_1', [64, 192, 3, 1, 1, 'convnew'], 'MaxP3_2_1', \
                [[192, 96, 16, 64, 64, 32, 32, 'inception'], [192, 96, 16, 64, 64, 32, 32, 'inception'], 'residual_block'], [192, 256, 1, 1, 0, 'convnew'], 'MaxP3_2_1', \
                [[256, 48, 8, 96, 104, 24, 32, 'inception'], [256, 56, 12, 80, 112, 32, 32, 'inception'], [256, 64, 12, 64, 128, 32, 32, 'inception'], 'residual_block'], [256, 512, 1, 1, 0, 'convnew'], \
                [[512, 112, 24, 160, 224, 64, 64, 'inception'], [512, 128, 24, 128, 256, 64, 64, 'inception'], 'residual_block'], [512, 832, 1, 1, 0, 'convnew'], 'MaxP3_2_1', \
                [832, 160, 32, 256, 320, 128, 128, 'inception'], [832, 192, 48, 384, 384, 128, 128, 'inception'], 'AvgP_7' \
                ]

cur_model = GoogleRef

class FewShotModel(nn.Module):
    def __init__(self, c=3, h=400, w=400, num_class=5):
        super().__init__()

        self.feature_list = cur_model
        print("Current Model Structure is", cur_model)

        features = add_features(self.feature_list)
        
        self.features = nn.Sequential(*features)

    def forward(self, x):
        
        x = self.features(x)

        return x

def init_weights(m):
    if type(m) == nn.Conv2d:
        print('Initialize', m)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)

if __name__ == '__main__':
    temp_shot = torch.randint(0, 200, (25, 3, 224, 224)).float()
    temp_query = torch.randint(0, 200, (20, 3, 224, 224)).float()
    temp_label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])

    model = FewShotModel()
    model.apply(init_weights)

    print(model)

    model_in = torch.cat((temp_shot, temp_query), 0)
    model_out = model(model_in)

    shot_out = model_out[:temp_shot.size(0)]
    query_out = model_out[temp_shot.size(0):]

    print(model_in.shape)
    print(shot_out.shape)
    print(query_out.shape)

    mean_shot_out = mean_vector_cal(shot_out, 5, 5)
    logits = square_euclidean_metric(query_out.view(20, -1), mean_shot_out)

    loss = class_vector_distance_softmax_loss(logits)

    print(loss)

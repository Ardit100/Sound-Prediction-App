from torchvision import models
import torch.nn.functional as F
import torch


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def vgg16_(pretrained):
    vgg16 = models.vgg16(pretrained=pretrained)
    return vgg16

class model_vgg16_gn(torch.nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(model_vgg16_gn, self).__init__()
        vgg16 = vgg16_(pretrained=pretrained)
        if freeze:
            vgg16 = vgg16_(pretrained=True)
            freeze_model(vgg16)
        self.groupnorm1 = torch.nn.GroupNorm(num_groups=1, num_channels=1, affine=True)
        self.conv1 = torch.nn.Conv2d(1, 3, (1,1), stride=(1,1), padding=0, bias=False)
        self.groupnorm2 = torch.nn.GroupNorm(num_groups=3, num_channels=3, affine=True)
        self.vgg16 = vgg16
        self.drop1 = torch.nn.Dropout(0.3)
        self.lin1 = torch.nn.Linear(in_features=1000, out_features=100, bias=True)
        self.drop2 = torch.nn.Dropout(0.2)
        self.lin2 = torch.nn.Linear(in_features=100, out_features=2, bias=True)
    def forward(self, x):
        x = self.groupnorm1(x)
        x = F.relu(self.groupnorm2(self.conv1(x)))  
        x = F.relu(self.vgg16(x)).to(device)
        x = self.drop1(x)
        x = F.relu(self.lin1(x))
        x = self.drop2(x)
        x = self.lin2(x)
        return x


# class model_crnn(torch.nn.Module):
#     def __init__(self, pretrained=False, freeze=False):
#         super(model_crnn, self).__init__()
#         vgg16 = vgg16_(pretrained=pretrained)
#         if freeze:
#             vgg16 = vgg16_(pretrained=True)
#             freeze_model(vgg16)
#         self.groupnorm1 = torch.nn.GroupNorm(num_groups=1, num_channels=1, affine=True)
#         self.conv1 = torch.nn.Conv2d(1, 3, (1,1), stride=(1,1), padding=0, bias=False)
#         self.groupnorm2 = torch.nn.GroupNorm(num_groups=3, num_channels=3, affine=True)
#         new_vgg16 = torch.nn.Sequential(*list(vgg16.features.children())[:-1])
#         self.vgg16 = new_vgg16
#         self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
#         self.lstm = torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)
#         self.lin1 = torch.nn.Linear(in_features=512*80, out_features=64, bias=True)
#         self.drop2 = torch.nn.Dropout(0.2)
#         self.lin2 = torch.nn.Linear(in_features=64, out_features=2, bias=True)

#     def forward(self, x):
#         x = self.groupnorm1(x)
#         x = F.relu(self.groupnorm2(self.conv1(x)))  
#         x = F.relu(self.vgg16(x))
#         x = self.maxpool(x)
#         x_swap = torch.transpose(x,-3,-1) # [8,20,4,512]
#         x = torch.reshape(x_swap, (x_swap.size(0),x_swap.size(1)*x_swap.size(2),x_swap.size(3))) # input for lstm [8,80,512]
#         h0 = torch.zeros(2,x.size(0),512).to(device)
#         c0 = torch.zeros(2,x.size(0),512).to(device)
#         x, _ = self.lstm(x, (h0,c0))
#         x = x.reshape(x.size(0), -1) # input for linear layer [8, 512*80]
#         x = F.relu(self.lin1(x))
#         x = self.drop2(x)
#         x = self.lin2(x)
#         return x



class model_crnn(torch.nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(model_crnn, self).__init__()
        vgg16 = vgg16_(pretrained=pretrained)
        if freeze:
            vgg16 = vgg16_(pretrained=True)
            freeze_model(vgg16)
        self.groupnorm1 = torch.nn.GroupNorm(num_groups=1, num_channels=1, affine=True)
        self.conv1 = torch.nn.Conv2d(1, 3, (1,1), stride=(1,1), padding=0, bias=False)
        self.groupnorm2 = torch.nn.GroupNorm(num_groups=3, num_channels=3, affine=True)
        new_vgg16 = torch.nn.Sequential(*list(vgg16.features.children())[:-1])
        self.vgg16 = new_vgg16
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.lstm = torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True, bidirectional = True)
        self.lin1 = torch.nn.Linear(in_features=2*512*80, out_features=100, bias=True)
        self.drop2 = torch.nn.Dropout(0.2)
        self.lin2 = torch.nn.Linear(in_features=100, out_features=2, bias=True)

    def forward(self, x):
        x = self.groupnorm1(x)
        x = F.relu(self.groupnorm2(self.conv1(x)))  
        x = F.relu(self.vgg16(x))
        x = self.maxpool(x)
        x_swap = torch.transpose(x,-3,-1) # [8,20,4,512]
        x = torch.reshape(x_swap, (x_swap.size(0),x_swap.size(1)*x_swap.size(2),x_swap.size(3))) # input for lstm [8,80,512]
        h0 = torch.zeros(2*2,x.size(0),512).to(device)
        c0 = torch.zeros(2*2,x.size(0),512).to(device)
        x, _ = self.lstm(x, (h0,c0))
        x = x.reshape(x.size(0), -1) # input for linear layer [8, 512*80]
        x = F.relu(self.lin1(x))
        x = self.drop2(x)
        x = self.lin2(x)
        return x
    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

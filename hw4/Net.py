import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.fc1 = nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=10)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x),0.2))
        x = self.pool(F.leaky_relu(self.conv2(x),0.2))
        x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(self.fc1(x),0.2)
        x = F.leaky_relu(self.fc2(x),0.2)
        x = self.fc3(x)
        return x

class mobile_Net(nn.Module):
    def __init__(self):
        super(mobile_Net, self).__init__()
        self.conv1_a = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=5,groups=3)
        self.conv1_b = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2_a = nn.Conv2d(in_channels=6,out_channels=6,kernel_size=5,groups=6)
        self.conv2_b = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=1)
        self.fc1 = nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=10)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1_b(self.conv1_a(x)),0.2))
        x = self.pool(F.leaky_relu(self.conv2_b(self.conv2_a(x)),0.2))
        x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(self.fc1(x),0.2)
        x = F.leaky_relu(self.fc2(x),0.2)
        x = self.fc3(x)
        return x
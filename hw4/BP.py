import torch
import numpy as np

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

import torchvision
import torchvision.transforms as transforms

batch_size=1024

transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

full_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform)
train_length = int(full_dataset.__len__()*0.8)
val_length = full_dataset.__len__() - train_length
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_length, val_length])
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
valloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, hidden_size=None):
        super(Net, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(hidden_size)-1):
            self.model.add_module(name='lin{}'.format(i + 1), module=nn.Linear(in_features=hidden_size[i], out_features=hidden_size[i + 1], bias=True))
            if i == len(hidden_size) - 2:
                self.model.add_module(name='sft', module=nn.Softmax(dim=-1))
            else:
                self.model.add_module(name='tanh{}'.format(i+1), module=nn.Tanh())

    def forward(self, x):
        return self.model(x)

model = Net([3072, 1000, 1000, 1000, 10])
print(model)

import torch.optim as optim

criterion = nn.NLLLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model)

for epoch in range(30):
    train_loss = 0.0
    train_acc = 0.0
    train_correct_count = 0
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        inputs=inputs.view([inputs.shape[0],3072])
        labels=labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        train_correct_count += (torch.argmax(outputs,dim=1)==labels).sum().cpu().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().item()
    train_loss=train_loss/(i+1)
    train_acc=train_correct_count/(i+1)/batch_size

    val_loss = 0.0
    val_acc = 0.0
    val_correct_count = 0
    model.eval()
    for i, data in enumerate(valloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        inputs=inputs.view([inputs.shape[0],3072])
        labels=labels.to(device)

        outputs = model(inputs)
        val_correct_count += (torch.argmax(outputs,dim=1)==labels).sum().cpu().item()
        loss = criterion(outputs, labels)
        val_loss += loss.cpu().item()
    val_loss=val_loss/(i+1)
    val_acc=val_correct_count/(i+1)/batch_size

    print('Epoch %d|Train_loss:%.3f Eval_loss:%.3f Train_acc:%.3f Eval_acc:%.3f' % (epoch + 1, train_loss, val_loss, train_acc, val_acc))

class_count=np.zeros(10,dtype=int)
correct_count=np.zeros(10,dtype=int)
model.eval()
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    inputs=inputs.view([inputs.shape[0],3072])
    outputs = torch.argmax(model(inputs),dim=1).cpu().item()
    class_count[labels]+=1
    if outputs==labels:
        correct_count[labels]+=1
for i in range(10):
    print('Test|Class{}('.format(i+1)+classes[i]+')-acc={}'.format(correct_count[i]/class_count[i]))
print('\nTest|Overall-acc={}'.format(np.sum(correct_count/10000)))
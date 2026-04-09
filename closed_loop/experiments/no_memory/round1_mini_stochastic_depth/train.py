'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import json
import random

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=200, type=int, help='number of epochs to train')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--out_dir', default='.', type=str, help='output directory for results')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--stochastic_depth_prob', default=0.5, type=float, 
                    help='probability of keeping a residual block during training')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Modified ResNet-18 with Stochastic Depth
class StochasticDepthResNet18(nn.Module):
    def __init__(self, stochastic_depth_prob=0.5):
        super().__init__()
        self.stochastic_depth_prob = stochastic_depth_prob
        self.base_model = ResNet18()
        
        # Get all residual blocks
        self.residual_blocks = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, BasicBlock):
                self.residual_blocks.append(module)
        
        # Calculate survival probabilities for each block
        # Linear decay from 1.0 to stochastic_depth_prob
        num_blocks = len(self.residual_blocks)
        self.survival_probs = [1.0 - (1.0 - stochastic_depth_prob) * (i / (num_blocks - 1)) 
                              for i in range(num_blocks)]
    
    def forward(self, x):
        # Initial convolution
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        # Layer1
        block_idx = 0
        for block in self.base_model.layer1:
            if self.training:
                survival_prob = self.survival_probs[block_idx]
                if random.random() > survival_prob:
                    # Skip this block during training
                    block_idx += 1
                    continue
            x = block(x)
            block_idx += 1
        
        # Layer2
        for block in self.base_model.layer2:
            if self.training:
                survival_prob = self.survival_probs[block_idx]
                if random.random() > survival_prob:
                    # Skip this block during training
                    block_idx += 1
                    continue
            x = block(x)
            block_idx += 1
        
        # Layer3
        for block in self.base_model.layer3:
            if self.training:
                survival_prob = self.survival_probs[block_idx]
                if random.random() > survival_prob:
                    # Skip this block during training
                    block_idx += 1
                    continue
            x = block(x)
            block_idx += 1
        
        # Layer4
        for block in self.base_model.layer4:
            if self.training:
                survival_prob = self.survival_probs[block_idx]
                if random.random() > survival_prob:
                    # Skip this block during training
                    block_idx += 1
                    continue
            x = block(x)
            block_idx += 1
        
        # Final layers
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        
        return x

# Model
print('==> Building model..')
net = StochasticDepthResNet18(stochastic_depth_prob=args.stochastic_depth_prob)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss / len(trainloader), 100. * correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    return test_loss / len(testloader), acc


results = []
for epoch in range(start_epoch, start_epoch + args.epoch):
    tr_loss, tr_acc = train(epoch)
    te_loss, te_acc = test(epoch)
    scheduler.step()
    results.append({
        'epoch': epoch,
        'train_loss': tr_loss,
        'train_acc': tr_acc,
        'test_loss': te_loss,
        'test_acc': te_acc,
    })

best_test_acc = max(r['test_acc'] for r in results)
output = {
    'best_test_acc': best_test_acc,
    'history': results,
}
results_path = os.path.join(args.out_dir, 'results.json')
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'Results saved to {results_path}')
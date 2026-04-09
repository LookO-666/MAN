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
import math

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=200, type=int, help='number of epochs to train')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--out_dir', default='.', type=str, help='output directory for results')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--seed', default=None, type=int, help='random seed')
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

os.makedirs(args.out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# RandAugment implementation
class RandAugment:
    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        self.augment_list = [
            (transforms.AutoContrast, 0, 1),
            (transforms.Equalize, 0, 1),
            (transforms.Invert, 0, 1),
            (transforms.Rotate, -30, 30),
            (transforms.Posterize, 4, 8),
            (transforms.Solarize, 0, 256),
            (transforms.ColorJitter, 0.1, 1.9, 0.1, 1.9, 0.1, 1.9),
            (transforms.Contrast, 0.1, 1.9),
            (transforms.Brightness, 0.1, 1.9),
            (transforms.Sharpness, 0.1, 1.9),
            (transforms.ShearX, -0.3, 0.3),
            (transforms.ShearY, -0.3, 0.3),
            (transforms.TranslateX, -0.45, 0.45),
            (transforms.TranslateY, -0.45, 0.45),
        ]
    
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            if random.random() > 0.5:
                img = self._apply_op(img, op)
        return img
    
    def _apply_op(self, img, op):
        transform_fn, *params = op
        if transform_fn in [transforms.AutoContrast, transforms.Equalize, transforms.Invert]:
            return transform_fn()(img)
        elif transform_fn == transforms.Rotate:
            angle = random.uniform(params[0], params[1])
            return transforms.functional.rotate(img, angle)
        elif transform_fn == transforms.Posterize:
            bits = int(random.uniform(params[0], params[1]))
            return transforms.functional.posterize(img, bits)
        elif transform_fn == transforms.Solarize:
            threshold = random.uniform(params[0], params[1])
            return transforms.functional.solarize(img, threshold)
        elif transform_fn == transforms.ColorJitter:
            brightness = random.uniform(params[0], params[1])
            contrast = random.uniform(params[2], params[3])
            saturation = random.uniform(params[4], params[5])
            hue = random.uniform(-0.5, 0.5)
            return transforms.functional.adjust_brightness(
                transforms.functional.adjust_contrast(
                    transforms.functional.adjust_saturation(
                        transforms.functional.adjust_hue(img, hue),
                        saturation),
                    contrast),
                brightness)
        elif transform_fn == transforms.Contrast:
            factor = random.uniform(params[0], params[1])
            return transforms.functional.adjust_contrast(img, factor)
        elif transform_fn == transforms.Brightness:
            factor = random.uniform(params[0], params[1])
            return transforms.functional.adjust_brightness(img, factor)
        elif transform_fn == transforms.Sharpness:
            factor = random.uniform(params[0], params[1])
            return transforms.functional.adjust_sharpness(img, factor)
        elif transform_fn == transforms.ShearX:
            shear = random.uniform(params[0], params[1])
            return transforms.functional.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[shear * 180 / math.pi, 0])
        elif transform_fn == transforms.ShearY:
            shear = random.uniform(params[0], params[1])
            return transforms.functional.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, shear * 180 / math.pi])
        elif transform_fn == transforms.TranslateX:
            translate = random.uniform(params[0], params[1])
            pixels = translate * img.size[0]
            return transforms.functional.affine(img, angle=0, translate=[pixels, 0], scale=1.0, shear=[0, 0])
        elif transform_fn == transforms.TranslateY:
            translate = random.uniform(params[0], params[1])
            pixels = translate * img.size[1]
            return transforms.functional.affine(img, angle=0, translate=[0, pixels], scale=1.0, shear=[0, 0])
        return img

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    RandAugment(n=2, m=9),
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

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = ResNet18()
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
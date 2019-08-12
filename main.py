"""Train PreActResNet on CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchtest
from torchsummary import summary
from pytorch_lightning import Trainer

import os
import argparse

from models.PreactResNet import PreActResNet18
from models.utils import progress_bar


def create_dataloaders():
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    test_ds = iter(trainloader).next()
    test_dl = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(100)), batch_size=100)
    val_dl = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(100, 200)), batch_size=10)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, test_ds, test_dl, val_dl, classes


def create_model(args):
    print('==> Building model..')
    net = PreActResNet18()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    summary(net, (3, 32, 32))

    return net, criterion, optimizer


# Training
def train(epoch, trainloader, verbose=True):
    if verbose:
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
        if verbose:
            progress_bar(batch_idx, len(trainloader), f'Loss: {train_loss / (batch_idx + 1)} | '
                                                      f'Acc: {100. * correct / total}'
                                                      f' ({correct}/{total})')

    return 100. * correct / total


def test(epoch, testloader, verbose=True):
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
            if verbose:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
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

    return acc


def overfit_test():
    for it in range(500):
        train_acc = train(it, test_dl, verbose=False)
    test_acc = test(it, val_dl)
    print(f'train_acc = {train_acc}')
    print(f'test_acc = {test_acc}')
    if train_acc >= 80:
        print('==> Overfit is Over and success!')
    else:
        raise AssertionError('Overfiting test not passed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch with PreActResNet CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', default=1, type=int, help='number of epochs for training')
    parser.add_argument('--test', action='store_true', help='testing model and train process though unit tests')
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    trainloader, testloader, test_ds, test_dl, val_dl, classes = create_dataloaders()

    # Model
    net, criterion, optimizer = create_model(args)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    if args.test:
        # testing model
        print('==> Testing model and train process...')

        torchtest.assert_vars_change(
            model=net,
            loss_fn=criterion,
            optim=optimizer,
            batch=test_ds,
            device=device)

        torchtest.test_suite(
            model=net,
            loss_fn=criterion,
            optim=optimizer,
            batch=test_ds,
            device=device,
            test_nan_vals=True,
            test_vars_change=True,
            # non_train_vars=None,
            test_inf_vals=True
        )

        overfit_test()

        print('==> All test are passed! Let is train whole network.')

    print('==> Let is TRAIN begin!')
    best_acc = 0  # best test accuracy
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(epoch, trainloader)
        test(epoch, testloader)
    print("==> Train is finished")

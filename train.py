import sys
import os
from os.path import join
from optparse import OptionParser
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torchvision import transforms

from model import UNet
from dataloader import DataLoader


def train_net(net,
              epochs=10,
              data_dir='data/cells/',
              n_classes=2,
              lr=0.001,  # Sara
              val_percent=0.1,
              save_cp=True,
              gpu=False):
    start = time.time()  # Calculating time
    loader = DataLoader(data_dir)
    N_train = loader.n_train()
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.99,
                          weight_decay=0.0005)
    best_acc = 0
    for epoch in range(epochs):

        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0

        for i, (img, label) in enumerate(loader):
            shape = img.shape
            label = label - 1
            # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)
            image_torch = torch.from_numpy(img.reshape(1, 1, shape[0], shape[1])).float()
            label_torch = torch.from_numpy(label).float()
            # todo: load image tensor to gpu

            if gpu:
                image_torch = image_torch.cuda()
                label_torch = label_torch.cuda()

            # todo: get prediction and getLoss()
            prediction = net.forward(image_torch)
            loss = getLoss(prediction, label)  # it's faster to use numpy label, choose function is using numpy
            epoch_loss += loss.item()
            print('Training sample %d / %d - Loss: %.6f' % (i + 1, N_train, loss.item()))

            # optimize weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch + 1, epoch_loss / (i + 1)))

        pred_sm = softmax(prediction)
        _, pred_label = torch.max(pred_sm, 1)

        gt_label = label_torch.long()
        pred_label_train = pred_label
        class_1 = (gt_label == 1).sum().item()
        class_0 = (gt_label == 0).sum().item()
        class_1_correct = torch.mul(gt_label, pred_label_train).sum().item()
        class_0_correct = (gt_label + pred_label_train == 0).sum().item()
        if class_1 != 0:
            class_1_acc = (class_1_correct * 100 / class_1)
        else:
            class_1_acc = 0
        if class_0 != 0:
            class_0_acc = (class_0_correct * 100 / class_0)
        else:
            class_0_acc = 0
        print("Accuracy of class 1 is %{:10.4f}".format(class_1_acc))
        print("Accuracy of class 0 is %{:10.4f}".format(class_0_acc))
        correct = (gt_label == pred_label_train).sum().item()
        if len(gt_label) != 0:
            accuracy = correct * 100 / len(gt_label) ** 2
        else:
            accuracy = 0

        print("Total Accuracy is %{:10.4f}".format(accuracy))

    stop = time.time()
    total_time = stop - start
    print("Total trainig took {0} minutes and {1} seconds.".format(total_time // 60, total_time % 60))

    # displays test images with original and predicted masks after training

    loader.setMode('test')

    net.eval()
    with torch.no_grad():
        print("Testing")
        for _, (img, label) in enumerate(loader):

            shape = img.shape
            label = label - 1
            img_torch = torch.from_numpy(img.reshape(1, 1, shape[0], shape[1])).float()
            lb_torch = torch.from_numpy(label).float()
            if gpu:
                img_torch = img_torch.cuda()
                lb_torch = lb_torch.cuda()

            pred = net(img_torch)
            pred_sm = softmax(pred)
            _, pred_label = torch.max(pred_sm, 1)

            plt.subplot(1, 3, 1)
            plt.imshow(img * 255.)
            plt.subplot(1, 3, 2)
            plt.imshow((label - 1) * 255.)
            plt.subplot(1, 3, 3)
            
            plt.imshow(pred_label.cpu().detach().numpy().squeeze() * 255.)
            plt.show()

            gt_label_test = lb_torch.long()
            pred_label_test = pred_label

            class_1 = (gt_label_test == 1).sum().item()
            class_0 = (gt_label_test == 0).sum().item()
            class_1_correct = torch.mul(gt_label_test, pred_label_test).sum().item()
            class_0_correct = (gt_label_test + pred_label_test == 0).sum().item()
            if class_1 != 0:
                class_1_acc = (class_1_correct * 100 / class_1)
            else:
                class_1_acc = 0
            if class_0 != 0:
                class_0_acc = (class_0_correct * 100 / class_0)
            else:
                class_0_acc = 0
            print("Class 0 accuracy: %{:10.4f}".format(class_0_acc))
            print("Class 1 accuracy: %{:10.4f}".format(class_1_acc))
            correct = (gt_label_test == pred_label_test).sum().item()

            if len(gt_label_test) != 0:
                accuracy = correct * 100 / len(gt_label_test) ** 2
            else:
                accuracy = 0

            print("Total Accuracy: %{:10.4f}".format(accuracy))


def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    return cross_entropy(p, target_label)


def softmax(input):
    # todo: implement softmax function
    p = torch.exp(input)
    sum = torch.sum(p, dim=1).view(p.shape[0], 1, p.shape[2], p.shape[3])
    return p / (sum + 1e-8)  # avoid devison by zero

def cross_entropy(input, targets):
    # todo: implement cross entropy
    # Hint: use the choose function
    ch = choose(input, targets)
    # ce = torch.mean(-1.0 * torch.log(ch))
    ce = torch.mean(-1.0 * torch.log(ch + 1e-8))  # avoid zero
    return ce


# Workaround to use numpy.choose() with PyTorch
def choose(pred_label, true_labels):
    size = pred_label.size()
    ind = np.empty([size[2] * size[3], 3], dtype=int)
    i = 0
    for x in range(size[2]):
        for y in range(size[3]):
            ind[i, :] = [true_labels[x, y], x, y]
            i += 1

    pred = pred_label[0, ind[:, 0], ind[:, 1], ind[:, 2]].view(size[2], size[3])

    return pred


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/cells/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.data_dir))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
              epochs=args.epochs,
              n_classes=args.n_classes,
              gpu=args.gpu,
              data_dir=args.data_dir)

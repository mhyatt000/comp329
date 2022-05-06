import functools
import os
import time
from argparse import ArgumentParser

import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as Parallel
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import torchvision.datasets as dset
import torchvision.transforms as transforms


from cptr import CPTR

'''TODO


encoder
    384 × 384 while maintaining the patch size equals to 16

decoder
    4 layers, 768 dimensions


3. make loss functions and predictions

4. scp to cluster and train
5. get predicrtions when the loss is low

'''


def get_args():

    ap = ArgumentParser(description='args for comp329 image caption')

    ap.add_argument('-s', '--save', action='store_true')
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('-t', '--train', action='store_true')
    ap.add_argument('-e', '--eval', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('-n', '--num_epochs', type=int)

    args = ap.parse_args()

    if not args.num_epochs:
        args.num_epochs = 5

    return args



def main():

    args = get_args()

    net = CPTR()
    net.file = __file__.split('/')[-1].split('.')[0] + '.pt'
    net.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.args = args

    # train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

    if args.load:
        try:
            net.load_state_dict(torch.load(net.file))
        except:
            pass

    if args.verbose:
        print(net)

    if torch.cuda.device_count() > 1:
        net = Parallel(net)

    # batch data for training
    print('getting training data...')
    # coco = dset.CocoCaptions(root = 'data',annFile = 'data/data.json',transform=transforms.ToTensor())

    x1 = torch.ones((9,16,16,3))#.reshape((-1,768))
    x1 = torch.ones((1,64,768))
    x2 = torch.ones((12,5,4,3))
    valid_lens = torch.tensor((3, 2))
        # why
    y = net(x1,x2, valid_lens)
    print(y.shape)
    quit()

    # train
    if args.train:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
        train(net, train_iter, optimizer, env=env)

    # do some predictions
    if args.eval:
        print('evaluation')

        # voc_dir = d2l.download_extract("voc2012", "VOCdevkit/VOC2012")
        # test_images, test_labels = d2l.read_voc_images(voc_dir, False)
        #
        # n, imgs = 4, []
        # for i in tqdm(range(n)):
        #     crop_rect = (0, 0, 320, 480)
        #     crop = lambda imgs: torchvision.transforms.functional.crop(imgs, *crop_rect)
        #
        #     X = crop(test_images[i])
        #     pred = label2image(predict(net, test_iter, X, env=env), env=env)
        #     imgs += [X.permute(1, 2, 0), pred.cpu(), crop(test_labels[i]).permute(1, 2, 0)]
        #
        # d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
        # # plt.pause(2)
        # plt.show()

if __name__ == '__main__':
    main()

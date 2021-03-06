#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: mlp
# Date: 11/22/18
# Author: 😏 <smirk dot cao at gmail dot com>
import sys
import os

p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)
from dataflow.datasets.dummy import load_curve1
import matplotlib.pyplot as plt
import logging
import torch
import glob
import imageio


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        l1_in = 1
        l1_out = 10
        l2_out = 10
        out = 1

        self.l1 = torch.nn.Linear(l1_in, l1_out)
        self.l2 = torch.nn.Linear(l1_out, l2_out)
        self.out = torch.nn.Linear(l2_out, out)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.tanh(x)
        x = self.l2(x)
        x = torch.nn.functional.sigmoid(x)
        o = self.out(x)
        return o


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    x, y = load_curve1()
    x = torch.autograd.Variable(torch.Tensor(x.reshape(-1, 1)), requires_grad=True)
    y = torch.autograd.Variable(torch.Tensor(y.reshape(-1, 1)))
    net = Net()
    logger.info(net)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x.data.numpy(), y.data.numpy())
    plt.ion()
    plt.show()

    for n_iter in range(1500):
        pred = net(x)
        loss = loss_fn(pred, y)
        # 重要
        opt.zero_grad()
        loss.backward()
        opt.step()
        if n_iter % 30 == 0:
            logger.info(loss.data.numpy())
            try:
                ax.texts.remove(text_loss)
                ax.texts.remove(text_iter)
            except NameError:
                pass

            ax.plot(x.data.numpy(), pred.data.numpy(), "-r", alpha=0.2)
            text_loss = plt.text(0.5, 0, 'loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            text_iter = plt.text(0.5, -1, 'n_iter=%d' % n_iter, fontdict={'size': 20, 'color': 'red'})
            plt.savefig(str(n_iter)+".png")
            # plt.pause(0.1)
    # plt.waitforbuttonpress()
    # generate gif
    ofname = "mlp_pytorch.gif"
    filenames = sorted(glob.glob("*.png"), key=os.path.getmtime, reverse=False)
    filenames = [os.getcwd() + "/" + fname for fname in filenames]
    if len(filenames) > 0:
        frames = [imageio.imread(img_name) for img_name in filenames]
        imageio.mimsave(ofname, frames, 'GIF', duration=0.1)

    # clear png file
    fs = glob.glob(r"*.png")
    for f in fs:
        os.remove(f)

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


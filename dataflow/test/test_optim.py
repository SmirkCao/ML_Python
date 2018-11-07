import matplotlib.pyplot as plt
import numpy as np
import unittest
import logging
import sys
import os

p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)

from dataflow.optim.optimizer import *


def f(x):
    # according to: https://zhuanlan.zhihu.com/p/41799394
    return (0.15 * x) ** 2 + np.cos(x) + np.sin(3 * x) / 3 + np.cos(5 * x) / 5 + np.sin(7 * x) / 7


def df(x):
    # derivative
    return (9 / 200) * x - np.sin(x) - np.sin(5 * x) + np.cos(3 * x) + np.cos(7 * x)


def save_fig(x, y, raw_x, raw_y, lr=None, name=None):

    plt.figure(figsize=(8, 6))
    plt.xlim(-20, 20)
    plt.ylim(-3, 10)
    plt.plot(raw_x, raw_y, c="b", alpha=0.5, linestyle="-")
    plt.annotate('start point',
                 xy=(x[0], y[0]), xycoords='data',
                 xytext=(30, 25), textcoords='offset points',
                 arrowprops=dict(arrowstyle="fancy",
                                 fc="0.4", ec="none",
                                 connectionstyle="angle3,angleA=0,angleB=90"))
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                      fontsize=12, fontdict={'family': 'monospace'})
    # data points
    plt.scatter(x, y, c='r', alpha=0.2)
    plt.plot(x, y, c='g', alpha=0.1)
    # lr
    plt.plot(-2 * np.ones(int(lr)),
             linestyle="-", color="y", linewidth=10, alpha=0.3)
    plt.plot(-2.5 * np.ones(int(10 * lr)),
             linestyle="-", color="y", linewidth=10, alpha=0.3)
    plt.text(-0.2, -2, "lr", **text_style)
    plt.text(-0.2, -2.5, "10*lr", **text_style)
    title = "{0},lr={1:.3f}".format(name, lr)
    plt.title(title)
    plt.savefig(title + ".png")
    plt.clf()


class TestOptimizer(unittest.TestCase):

    def test_sgd(self):
        """
        test gradient decent
        :return:
        """
        points_x = np.linspace(-20, 20, 1000)
        points_y = f(points_x)

        max_iter = 1000
        for i in range(10):
            lr = pow(2, -i) * 16
            x = [np.array([-20.0])]
            derivative = [np.array([df(-20)])]

            opt_x, opt_y = [], []
            opt = SGD(params={}, lr=lr)
            opt._vars = x
            opt._grads = derivative
            for _ in range(max_iter):
                opt_x.append(x[0][0]), opt_y.append(f(x[0][0]))
                opt.step()
                for var, grad in zip(opt._vars, opt._grads):
                    grad[:] = df(var[0])

            save_fig(opt_x, opt_y, points_x, points_y, **opt.info)

    def test_momentum(self):
        """
        test momentum optimizer
        :return:
        """
        points_x = np.linspace(-20, 20, 1000)
        points_y = f(points_x)

        max_iter = 1000
        for i in range(10):
            lr = pow(2, -i) * 16
            x = [np.array([-20.0])]
            derivative = [np.array([df(-20)])]

            opt_x, opt_y = [], []
            opt = Momentum(params={}, lr=lr, momentum=0.9)
            opt._vars = x
            opt._grads = derivative
            opt._mv = [np.zeros_like(v) for v in opt._vars]

            for _ in range(max_iter):
                opt_x.append(x[0][0]), opt_y.append(f(x[0][0]))
                opt.step()
                for var, grad, mv in zip(opt._vars, opt._grads, opt._mv):
                    grad[:] = df(var)

            save_fig(opt_x, opt_y, points_x, points_y, **opt.info)

    def test_adagrad(self):
        """
        test AdaGrad optimizer
        :return:
        """
        points_x = np.linspace(-20, 20, 1000)
        points_y = f(points_x)

        max_iter = 1000
        for i in range(10):
            lr = pow(2, -i) * 16
            x = [np.array([-20.0])]
            derivative = [np.array([df(-20)])]

            opt_x, opt_y = [], []
            opt = AdaGrad(params={}, lr=lr)
            opt._vars = x
            opt._grads = derivative
            opt._rs = [np.zeros_like(v) for v in opt._vars]

            for _ in range(max_iter):
                opt_x.append(x[0][0]), opt_y.append(f(x[0][0]))
                opt.step()
                for var, grad in zip(opt._vars, opt._grads):
                    grad[:] = df(var)

            save_fig(opt_x, opt_y, points_x, points_y, **opt.info)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

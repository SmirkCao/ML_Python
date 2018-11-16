#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: test_datasets
# Date: 11/10/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import sys
import os

p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)
from dataflow.datasets.dummy import *
import unittest
import matplotlib.pyplot as plt


class TestDatasets(unittest.TestCase):
    def test_mnist(self):
        load_mnist()
        pass

    def test_line(self):
        x, y = load_line()
        plt.scatter(x, y)
        # plt.show()

    def test_curve(self):
        x, y = load_curve1()
        plt.scatter(x, y)
        # plt.show()

    def test_build_gallery(self):
        # line
        x, y = load_line()
        plt.scatter(x, y, marker="$😏$", label="Smirk", s=150, linewidths=0.1)
        plt.legend(loc='upper left')
        plt.savefig("../datasets/assert/gallery_line.png")

        # curve
        plt.clf()
        x, y = load_curve1()
        plt.scatter(x, y, marker="$😏$", label="Smirk", s=150, linewidths=0.1)
        plt.legend(loc='upper left')
        plt.savefig("../datasets/assert/gallery_curve.png")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

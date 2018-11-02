#! /usr/bin/env python
#! -*- coding=utf-8 -*-
# Project:  ML_Python
# Filename: test_layer
# Date: 10/30/18
# Author: 😏 <smirk dot cao at gmail dot com>
import logging
import unittest
import sys
import os
import torch
p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)

import dataflow.nn as nn
from dataflow.datasets import dummy
import numpy as np


class TestLayer(unittest.TestCase):

    def test_dense(self):
        x, y = dummy.load_dummy()

        l1 = nn.layers.Dense(fan_in=x.shape[1], fan_out=10)
        out_dataflow = l1.forward(x)
        m = torch.nn.Linear(2, 10)
        out_torch = m(torch.tensor(x, dtype=torch.float32))
        logger.info(out_torch.shape)
        logger.info(out_dataflow.shape)
        # np.testing.assert_array_almost_equal(out_dataflow, out_torch.detach().numpy())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


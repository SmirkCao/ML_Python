#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Project:  ML_Python
# Filename: test_utils
# Date: 11/8/18
# Author: 😏 <smirk dot cao at gmail dot com>
import unittest
import logging
import sys
import os

p = os.path.join(os.getcwd(), "../../")
sys.path.append(p)

from dataflow.nn.utils import _single, _pair, _triple

DEBUG = True


class TestUtils(unittest.TestCase):
    def test_ntuple(self):
        # int
        self.assertTupleEqual(_single(1), (1, ))
        self.assertTupleEqual(_pair(2), (2, 2))
        self.assertTupleEqual(_triple(3), (3, 3, 3))
        # tuple
        self.assertTupleEqual(_single((1, )), (1, ))
        self.assertTupleEqual(_pair((2, 2)), (2, 2))
        self.assertTupleEqual(_triple((3, 3, 3)), (3, 3, 3))
        # list
        self.assertTupleEqual(_single([1, ]), (1, ))
        self.assertTupleEqual(_pair([2, 2]), (2, 2))
        self.assertTupleEqual(_triple([3, 3, 3]), (3, 3, 3))

    def get_padded_and_tmp_out(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    unittest.main()

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

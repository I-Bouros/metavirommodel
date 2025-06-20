#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#

import unittest

import pandas as pd
import numpy as np
import math

import metavirommodel as mm


class TestBirthRatePrec(unittest.TestCase):
    """
    Test the 'BirthRatePrec' class.
    """
    def test__init__(self):
        precipitation_data = pd.DataFrame({
            'Day': [1, 3, 7, 34],
            'Precmm': [0, 2, 0, 10]
        })

        parameters = [0.5, 0.008, 2]
        parameters1 = [0.5, 0.008]

        birth_rate = mm.BirthRatePrec(precipitation_data, parameters)
        birth_rate1 = mm.BirthRatePrec(precipitation_data, parameters1)

        self.assertEqual(birth_rate.bM, 0.5)
        self.assertEqual(birth_rate.rho, 0.008)
        self.assertEqual(len(birth_rate.avg_prec), 34)

        self.assertEqual(birth_rate1.bM, 0.5)
        self.assertEqual(birth_rate1.rho, 0.008)
        self.assertEqual(len(birth_rate1.avg_prec), 34)

        # Check that precipitation data meets conditions
        with self.assertRaises(TypeError):
            precipitation_data1 = [0, 2, 0, 10]
            mm.BirthRatePrec(precipitation_data1, parameters)

        with self.assertRaises(ValueError):
            precipitation_data1 = pd.DataFrame({
                'Time': [1, 3, 7, 34],
                'Precmm': [0, 2, 0, 10]
            })
            mm.BirthRatePrec(precipitation_data1, parameters)

        with self.assertRaises(ValueError):
            precipitation_data1 = pd.DataFrame({
                'Day': [1, 3, 7, 34],
                'Mm': [0, 2, 0, 10]
            })
            mm.BirthRatePrec(precipitation_data1, parameters)

        # Check that hyper-parameters meets conditions
        with self.assertRaises(TypeError):
            parameters2 = (0.5, 0.008)
            mm.BirthRatePrec(precipitation_data, parameters2)

        with self.assertRaises(ValueError):
            parameters2 = [0.5, 0.008, 10, 0.2]
            mm.BirthRatePrec(precipitation_data, parameters2)

        with self.assertRaises(ValueError):
            parameters2 = [0.5]
            mm.BirthRatePrec(precipitation_data, parameters2)

        with self.assertRaises(TypeError):
            parameters2 = [0.5, '0.008']
            mm.BirthRatePrec(precipitation_data, parameters2)

        with self.assertRaises(ValueError):
            parameters2 = [-0.5, 0.008, 10]
            mm.BirthRatePrec(precipitation_data, parameters2)

        with self.assertRaises(TypeError):
            parameters2 = [0.5, 0.008, 0.2]
            mm.BirthRatePrec(precipitation_data, parameters2)

    def test_call(self):
        precipitation_data = pd.DataFrame({
            'Day': [1, 3, 7, 34],
            'Precmm': [0, 2, 0, 10]
        })

        parameters = [0.5, 0.008, 2]
        parameters1 = [0.5, 0.008]

        birth_rate = mm.BirthRatePrec(precipitation_data, parameters)
        birth_rate1 = mm.BirthRatePrec(precipitation_data, parameters1)

        self.assertAlmostEqual(birth_rate(4.2), 0.5/1.008)
        self.assertAlmostEqual(birth_rate1(4.2), 0.5/(0.008*15 + 1))

        with self.assertRaises(ValueError):
            birth_rate(35)
        with self.assertRaises(ValueError):
            birth_rate1(35)


class TestBirthRateSeason(unittest.TestCase):
    """
    Test the 'BirthRateSeason' class.
    """
    def test__init__(self):
        parameters = [0.5, 0.15]

        birth_rate = mm.BirthRateSeason(parameters)

        self.assertEqual(birth_rate.b0, 0.5)
        self.assertEqual(birth_rate.w, 0.15)

        # Check that hyper-parameters meets conditions
        with self.assertRaises(TypeError):
            mm.BirthRateSeason((0.5, 0.15))

        with self.assertRaises(ValueError):
            mm.BirthRateSeason([0.5, 0.15, 0.2])

        with self.assertRaises(ValueError):
            mm.BirthRateSeason([0.5, -0.15])

    def test_call(self):
        parameters = [0.5, 0.15]

        birth_rate = mm.BirthRateSeason(parameters)

        self.assertAlmostEqual(
            birth_rate(4.2),
            np.abs(0.5*np.sin(-2*math.pi*0.15)) + 0.5*np.sin(-2*math.pi*0.15))
        self.assertAlmostEqual(
            birth_rate(104.2),
            np.abs(0.5*np.sin(2*math.pi*(14/52-0.15))) + 0.5*np.sin(
                2*math.pi*(14/52-0.15)))

import unittest
import torch
import sys
import os
sys.path.insert(1, '../script')

from robustness import RobustnessMetric

class TestCalcFscore(unittest.TestCase):
    def test_calc_fscore(self):
        trans_deriv1 = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0, 9.0])]
        rot_deriv1 = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5, 0.6]), torch.tensor([0.7, 0.8, 0.9])]
        trans_deriv2 = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0, 9.0])]
        rot_deriv2 = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5, 0.6]), torch.tensor([0.7, 0.8, 0.9])]

        trans_threshold = 0.2
        rot_threshold = 0.2

        fscore_trans, fscore_rot = RobustnessMetric.calc_fscore(trans_deriv1, rot_deriv1, trans_deriv2, rot_deriv2, trans_threshold, rot_threshold)

        self.assertAlmostEqual(fscore_trans, 1.0, places=1.0)
        self.assertAlmostEqual(fscore_rot, 1.0, places=1.0)
    
    def test_estimate_velocity(self):
        

if __name__ == '__main__':
    unittest.main()

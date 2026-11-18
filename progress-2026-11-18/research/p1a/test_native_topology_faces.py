import unittest

import numpy as np
import torch

from research.p1a.native_topology_prefix_fit import physical_labels, wall_intrusion
from research.p1a.physical_topology_gate import ACTION_BLOCK, PhysicalRollout, make_scene


class FactualFaceTests(unittest.TestCase):
    def setUp(self):
        self.physical = PhysicalRollout(make_scene(2026112003))

    def one_block(self, starts, commands, *, walls=True, passages=False):
        starts = np.asarray(starts, dtype=np.float32)
        commands = np.asarray(commands, dtype=np.float32)
        position = starts.transpose(1, 0, 2)
        primitive = []
        for _ in range(ACTION_BLOCK):
            position = self.physical.advance(position, commands)
            primitive.append(position)
        raw = np.stack(primitive, axis=-2).transpose(1, 0, 2, 3)[:, None, :, None]
        commands = np.repeat(commands[:, None, None, None], ACTION_BLOCK, axis=-2)
        centers = np.broadcast_to(
            np.asarray([self.physical.top, self.physical.bottom, self.physical.left, self.physical.right]),
            (len(starts), 4),
        ).copy()
        neutral = (raw[:, :, 0] == raw[:, :, 1]).all(axis=(-2, -1))
        labels, _ = physical_labels(raw, starts[:, None, :, None], commands, neutral,
                                    (self.physical.min_xy, self.physical.max_xy), centers, 'cpu',
                                    walls=walls, factual_passages=passages)
        return torch.from_numpy(raw[..., -1, :].copy()), labels

    def test_completed_passages_on_both_sides_leave_aperture_unconstrained(self):
        starts = [[[145., y]] * 2 for y in (113., 111., 113., 94.)]
        commands = [[0., -1.], [0., 1.], [0., -.5], [0., -1.]]
        truth, previous = self.one_block(starts, commands)
        _, factual = self.one_block(starts, commands, passages=True)
        self.assertFalse(previous['wall_side'].any())
        expected = torch.tensor([[1, 1], [-1, -1], [0, 0], [0, 0]], dtype=torch.int8)
        torch.testing.assert_close(factual['wall_side'][:, 0, :, 0, 1], expected)
        self.assertFalse(factual['wall_side'][..., 0].any())

        predicted = truth.clone().requires_grad_(True)
        valid = torch.ones((len(starts), 1), dtype=torch.bool)
        at_truth = wall_intrusion(predicted, factual['wall_side'], valid)
        self.assertEqual(at_truth.item(), 0.)
        self.assertFalse(torch.autograd.grad(at_truth, predicted)[0].any())

        # Wrong-side forecasts on opposite passages must get opposite gradients.
        predicted = (truth + truth.new_tensor([[0., 25.], [0., -25.], [0., 25.], [0., 25.]])[:, None, None, None])
        predicted.requires_grad_(True)
        derivative = torch.autograd.grad(wall_intrusion(predicted, factual['wall_side'], valid), predicted)[0]
        self.assertTrue((derivative[0, ..., 1] > 0).all())
        self.assertTrue((derivative[1, ..., 1] < 0).all())
        self.assertFalse(derivative[2:].any())
        valid[0] = False
        masked = torch.autograd.grad(wall_intrusion(predicted, factual['wall_side'], valid), predicted)[0]
        self.assertFalse(masked[0].any())

    def test_cand231_h3_truthful_prefix_gets_destination_gradient(self):
        # Archived 2026112003 southeast candidate 231 at the start of H3.
        starts = [[[133.81939697265625, 115.92948913574219]] * 2]
        commands = [[-1., -.9033470153808594]]
        truth, legacy_source = self.one_block(starts, commands, walls=False)
        _, legacy_root = self.one_block(starts, commands)
        _, factual_source = self.one_block(starts, commands, passages=True)
        self.assertNotIn('wall_side', legacy_source)
        # H3 crosses the opening and clips once on its destination side.
        torch.testing.assert_close(legacy_root['wall_side'], factual_source['wall_side'])
        torch.testing.assert_close(truth[0, 0, :, 0, 1], torch.full((2,), 98.48326110839844))

        predicted = torch.tensor([[[[[127.98406982421875, 112.5135726928711]],
                                     [[129.20919799804688, 111.92974853515625]]]]], requires_grad=True)
        valid = torch.ones((1, 1), dtype=torch.bool)
        derivative = torch.autograd.grad(wall_intrusion(predicted, factual_source['wall_side'], valid), predicted)[0]
        torch.testing.assert_close(derivative[..., 1], predicted.detach()[..., 1] - self.physical.near)
        self.assertFalse(derivative[..., 0].any())

    def test_cand231_h5_adds_completed_crossing_without_contact(self):
        starts = [[[114.96399688720703, 92.30402374267578], [121., 92.30402374267578]]]
        commands = [[-1., -.607836127281189]]
        truth, legacy = self.one_block(starts, commands)
        _, factual = self.one_block(starts, commands, passages=True)
        torch.testing.assert_close(truth[0, 0, :, 0, 0], torch.tensor([102.46399688720703, 121.]))
        torch.testing.assert_close(legacy['wall_side'][0, 0, :, 0, 0], torch.tensor([0, -1], dtype=torch.int8))
        torch.testing.assert_close(factual['wall_side'][0, 0, :, 0, 0], torch.tensor([1, -1], dtype=torch.int8))

    def test_cand380_h5_contact_value_and_derivative_are_retained(self):
        starts = [[[121., 132.31874084472656], [117.60044860839844, 132.31874084472656]]]
        commands = [[-.8102977275848389, -.42576515674591064]]
        truth, legacy = self.one_block(starts, commands)
        _, factual = self.one_block(starts, commands, passages=True)
        torch.testing.assert_close(legacy['wall_side'], factual['wall_side'])
        torch.testing.assert_close(factual['wall_side'][0, 0, :, 0, 0], torch.tensor([-1, -1], dtype=torch.int8))
        predicted = torch.tensor([[[[[97.00288391113281, 94.99958038330078]],
                                     [[107.34048461914062, 93.5629653930664]]]]], requires_grad=True)
        valid = torch.ones((1, 1), dtype=torch.bool)
        old = wall_intrusion(predicted, legacy['wall_side'], valid)
        new = wall_intrusion(predicted, factual['wall_side'], valid)
        torch.testing.assert_close(new, old)
        old_gradient = torch.autograd.grad(old, predicted)[0]
        new_gradient = torch.autograd.grad(new, predicted)[0]
        torch.testing.assert_close(new_gradient, old_gradient)
        torch.testing.assert_close(new_gradient[..., 0], predicted.detach()[..., 0] - self.physical.far)
        self.assertEqual(wall_intrusion(truth, factual['wall_side'], valid).item(), 0.)


if __name__ == '__main__':
    unittest.main()

import unittest

import torch

from research.p1a.native_topology_prefix_fit import source_hinge
from research.p1a.native_topology_xy_evidence import student_t


class SourceObjectiveTests(unittest.TestCase):
    def test_factual_decoded_centers_are_stationary_for_weak_and_strong_pairs(self):
        factor = torch.tensor([[10., 0.], [.4, 5.]], dtype=torch.double)
        arrived = torch.tensor([[[1.0000001192092896, 2.], [4., 2.1]],
                                [[12., 8.], [12., 8.]],
                                [[0., 0.], [80., 80.]]], dtype=torch.float32)
        informative = torch.tensor([True, False, True])
        issued = arrived.clone().requires_grad_()

        old = source_hinge(issued, arrived, informative, factor)
        old_gradient, = torch.autograd.grad(old, issued)
        weak_direction = arrived[0, 1] - arrived[0, 0]
        self.assertGreater(float((old_gradient[0, 0] - old_gradient[0, 1]) @ weak_direction), 0.)
        torch.testing.assert_close(old_gradient[0].sum(0), torch.zeros_like(weak_direction), atol=0, rtol=0)
        self.assertEqual(torch.count_nonzero(old_gradient[1:]).item(), 0)

        factual = source_hinge(issued, arrived, informative, factor, factual_targets=True)
        factual_gradient, = torch.autograd.grad(factual, issued)
        self.assertEqual(factual.item(), 0.)
        self.assertEqual(torch.count_nonzero(factual_gradient).item(), 0)

    def test_neutral_parity_preserves_existing_informative_normalization(self):
        factor = torch.tensor([[2., 0.], [.4, 1.6]], dtype=torch.double)
        arrived = torch.tensor([[[-8., 0.], [8., 0.]], [[1., 2.], [1., 2.]]], dtype=torch.double)
        issued = torch.tensor([[[0., 0.], [0., 0.]], [[0., 2.], [6., 2.]]], dtype=torch.double,
                              requires_grad=True)
        informative = torch.tensor([True, False])
        residual = torch.linalg.solve_triangular(factor, (arrived[1, 0] - issued[1]).T, upper=False)
        ll = student_t(-.5 * residual.square().sum(0))
        magnitude = (ll[0] - ll[1]).abs()
        control = source_hinge(issued, arrived, informative, factor)
        factual = source_hinge(issued, arrived, informative, factor, factual_targets=True)
        torch.testing.assert_close(factual, control + magnitude / 2, atol=1e-14, rtol=0)
        torch.testing.assert_close(factual, source_hinge(issued.flip(1), arrived.flip(1), informative, factor,
                                                        factual_targets=True), atol=1e-14, rtol=0)
        control_gradient, = torch.autograd.grad(control, issued)
        factual_gradient, = torch.autograd.grad(factual, issued)
        torch.testing.assert_close(factual_gradient[0], control_gradient[0], atol=0, rtol=0)
        self.assertGreater(float(factual_gradient[1].norm()), 0.)
        updated = issued.detach() - .01 * factual_gradient
        neutral = source_hinge(issued[1:], arrived[1:], informative[1:], factor, factual_targets=True)
        updated_neutral = source_hinge(updated[1:], arrived[1:], informative[1:], factor, factual_targets=True)
        torch.testing.assert_close(neutral, magnitude, atol=1e-14, rtol=0)
        self.assertLess(updated_neutral.item(), neutral.item())


if __name__ == "__main__":
    unittest.main()

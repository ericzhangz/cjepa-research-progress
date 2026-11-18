import math
import unittest
from unittest.mock import patch

import torch

if __package__:
    from .native_response_classes import partition_response_classes, solve_activation_relations
else:
    from native_response_classes import partition_response_classes, solve_activation_relations


def _tensor(values, *, dtype=torch.double):
    return torch.tensor(values, dtype=dtype)


def _member_rows(groups):
    return [tuple(group.members.tolist()) for group in groups]


def _expand(groups, read):
    outputs = [None] * sum(len(group.members) for group in groups)
    for group in groups:
        value = read(group.activation)
        for member in group.members.tolist():
            outputs[member] = value
    return torch.stack(outputs)


def _conditional(log_prior, log_likelihood):
    updated = log_prior + log_likelihood
    return updated - updated.logsumexp(dim=0)


def _p_activation(b):
    return torch.stack((b.new_tensor(2), b.new_tensor(1), 1.75 - 1.5 * (b - 1)))


class NativeResponseClassTests(unittest.TestCase):
    def setUp(self):
        self.directions = _tensor([[1, -1, 0], [0, 1, -1]])
        self.thresholds = _tensor([0, 0])
        self.high_strengths = _tensor([1, 1])
        self.assignments = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.probe = _tensor([2, 1, 0])
        self.reference = _tensor([2, 1, 2])
        self.neutral = _tensor([1, 2, 3])
        self.variance = (1 / 3) ** 2 / (2 * math.log(3))

    def _partition(self, v, assignments=None):
        if assignments is None:
            assignments = self.assignments
        level_strengths = torch.stack((torch.zeros_like(self.high_strengths), self.high_strengths), dim=1)
        return partition_response_classes(v, self.directions, self.thresholds, level_strengths, assignments)

    def _dense(self, v, assignments=None):
        if assignments is None:
            assignments = self.assignments
        strengths = assignments * self.high_strengths
        delta, _, _ = solve_activation_relations(v, self.directions, self.thresholds, strengths)
        return v + delta

    def _arrive(self, log_prior, v, y, *, grouped):
        if grouped:
            groups = self._partition(v)
            ll = _expand(groups, lambda q: -(q[1] - y).square() / (2 * self.variance))
        else:
            ll = -(self._dense(v)[:, 1] - y).square() / (2 * self.variance)
        return _conditional(log_prior, ll)

    def _posterior_sequence(self, *, grouped):
        log_uniform = self.probe.new_full((len(self.assignments),), -math.log(len(self.assignments)))
        after_probe = self._arrive(log_uniform, self.probe, 1, grouped=grouped)
        after_reference = self._arrive(after_probe, self.reference, 4 / 3, grouped=grouped)
        after_neutral = self._arrive(after_reference, self.neutral, 2, grouped=grouped)
        return after_probe, after_reference, after_neutral

    def _compile_trace(self, next_activation=_p_activation):
        h1 = self._partition(self.reference)
        h2 = []
        dense_indexed_outputs = [None] * len(self.assignments)
        for parent in h1:
            children = self._partition(next_activation(parent.activation[1]), self.assignments[parent.members])
            for child in children:
                members = parent.members[child.members]
                h2.append(tuple(members.tolist()))
                for mode in members.tolist():
                    dense_indexed_outputs[mode] = child.activation
        return h1, h2, torch.stack(dense_indexed_outputs)

    def _dense_trace(self, next_activation=_p_activation):
        h1 = self._dense(self.reference)
        return h1, torch.stack([
            self._dense(next_activation(q[1]), row.unsqueeze(0))[0]
            for q, row in zip(h1, self.assignments)
        ])

    def test_joint_load_certifies_classes_before_enumerating_outputs(self):
        v = _tensor([2, 1, 1.25])
        groups = self._partition(v)
        self.assertEqual(_member_rows(groups), [(0, 1), (2,), (3,)])
        self.assertEqual([group.representative for group in groups], [0, 2, 3])
        self.assertEqual(groups[0].pinned.tolist(), [True, False])
        self.assertEqual(groups[1].pinned.tolist(), [True, True])
        self.assertLess(groups[0].slack_upper[1].item(), 0)
        self.assertAlmostEqual((groups[1].activation @ self.directions[1]).item(), 1 / 12)
        torch.testing.assert_close(_expand(groups, lambda q: q), self._dense(v))

    def test_real_arrivals_reweight_the_joint_table_and_preserve_conditional_odds(self):
        reference_groups = self._partition(self.reference)
        neutral_groups = self._partition(self.neutral)
        self.assertEqual(_member_rows(reference_groups), [(0, 1), (2, 3)])
        self.assertEqual(_member_rows(neutral_groups), [(0, 1, 2, 3)])
        torch.testing.assert_close(self._dense(self.probe)[:, 1], _tensor([1, 2 / 3, 4 / 3, 1]))
        probe, reference, neutral = self._posterior_sequence(grouped=True)
        torch.testing.assert_close(probe.exp(), _tensor([3, 1, 1, 3]) / 8)
        torch.testing.assert_close(reference.exp(), _tensor([3, 1, 3, 9]) / 16)
        torch.testing.assert_close(neutral, reference)
        for grouped, dense in zip((probe, reference, neutral), self._posterior_sequence(grouped=False)):
            torch.testing.assert_close(grouped, dense)
        self.assertAlmostEqual(probe.exp()[[1, 3]].sum().item(), 0.5)
        self.assertAlmostEqual(reference.exp()[[1, 3]].sum().item(), 0.625)
        torch.testing.assert_close(probe[[1, 3]] - probe[[0, 2]], reference[[1, 3]] - reference[[0, 2]])

    def test_future_rebranches_only_when_its_own_window_loads_a_relation(self):
        before, _, after = self._posterior_sequence(grouped=True)
        with patch(f"{partition_response_classes.__module__}.solve_activation_relations",
                   wraps=solve_activation_relations) as solver:
            h1_groups, h2_members, h2 = self._compile_trace()
        self.assertEqual(solver.call_count, 5)
        h1 = _expand(h1_groups, lambda q: q)
        dense_h1, dense_h2 = self._dense_trace()
        self.assertEqual(_member_rows(h1_groups), [(0, 1), (2, 3)])
        self.assertEqual(h2_members, [(0, 1), (2,), (3,)])
        torch.testing.assert_close(h1, dense_h1)
        torch.testing.assert_close(h2, dense_h2)
        torch.testing.assert_close(h2[:, 1], _tensor([1, 1, 4 / 3, 21 / 16]))

        _, repeat_members, repeat_h2 = self._compile_trace(lambda _: self.reference)
        _, dense_repeat_h2 = self._dense_trace(lambda _: self.reference)
        self.assertEqual(repeat_members, [(0, 1), (2, 3)])
        torch.testing.assert_close(repeat_h2, dense_repeat_h2)
        response = h2[:, 1] - repeat_h2[:, 1]
        self.assertAlmostEqual((before.exp() @ h1[:, 1]).item(), 7 / 6)
        self.assertAlmostEqual((after.exp() @ h1[:, 1]).item(), 5 / 4)
        self.assertAlmostEqual((before.exp() @ response).item(), -1 / 128)
        self.assertAlmostEqual((after.exp() @ response).item(), -3 / 256)
        mean_window = _p_activation(after.exp() @ h1[:, 1])
        mean_window_h2 = _expand(self._partition(mean_window), lambda q: q[1])
        torch.testing.assert_close(mean_window_h2, h1[:, 1])
        self.assertGreater((mean_window_h2[-1] - h2[-1, 1]).item(), 0)

    def test_future_gradient_reaches_factual_source_and_matches_exhaustive_paths(self):
        parameters = (self.probe, self.reference, self.directions, self.thresholds, self.high_strengths)
        for parameter in parameters:
            parameter.requires_grad_()

        _, _, grouped_support = self._posterior_sequence(grouped=True)
        grouped_h2 = self._compile_trace()[2]
        grouped_loss = -torch.logsumexp(grouped_support - (grouped_h2[:, 1] - 1.5).square(), dim=0)
        grouped_grads = torch.autograd.grad(grouped_loss, parameters)

        _, _, dense_support = self._posterior_sequence(grouped=False)
        dense_h2 = self._dense_trace()[1]
        dense_loss = -torch.logsumexp(dense_support - (dense_h2[:, 1] - 1.5).square(), dim=0)
        dense_grads = torch.autograd.grad(dense_loss, parameters)

        self.assertGreater(grouped_grads[0].norm().item(), 1e-8)
        torch.testing.assert_close(grouped_loss, dense_loss)
        for grouped, dense in zip(grouped_grads, dense_grads):
            torch.testing.assert_close(grouped, dense, rtol=1e-10, atol=1e-11)

    def test_boundary_and_near_boundary_rows_stay_pinned(self):
        for dtype, upper in ((torch.double, 1), (torch.float32, 1 + torch.finfo(torch.float32).eps)):
            with self.subTest(dtype=dtype):
                directions, thresholds = self.directions.to(dtype), self.thresholds.to(dtype)
                levels = _tensor([[0, 1], [0, 1]], dtype=dtype)
                v = _tensor([2, 1, upper], dtype=dtype)
                groups = partition_response_classes(v, directions, thresholds, levels, self.assignments)
                self.assertEqual(_member_rows(groups), [(0,), (1,), (2,), (3,)])
                self.assertTrue(groups[0].pinned[1])
                actual = _expand(groups, lambda q: q)
                delta, _, _ = solve_activation_relations(v, directions, thresholds, self.assignments.to(dtype))
                torch.testing.assert_close(actual, v + delta)

    def test_dependent_rows_are_safe_and_need_not_find_every_same_activation(self):
        self.directions = _tensor([[1, 0, 0], [1, 0, 0]])
        v = _tensor([2, 1, 0])
        groups = self._partition(v)
        self.assertEqual(_member_rows(groups), [(0,), (1,), (2,), (3,)])
        actual = _expand(groups, lambda q: q)
        torch.testing.assert_close(actual[1], actual[2])
        torch.testing.assert_close(actual, self._dense(v))
        groups = self._partition(v, self.assignments[[3, 0, 3]])
        self.assertEqual(_member_rows(groups), [(0, 2), (1,)])

    def test_signed_gram_needs_no_condition_for_safe_grouping(self):
        self.directions = _tensor([[1, 0, 0], [1, 1, 0]])
        self.thresholds = _tensor([0, 4])
        v = _tensor([2, 1, 0])
        groups = self._partition(v)
        self.assertEqual(_member_rows(groups), [(0, 1), (2, 3)])
        torch.testing.assert_close(_expand(groups, lambda q: q), self._dense(v))


if __name__ == "__main__":
    unittest.main()

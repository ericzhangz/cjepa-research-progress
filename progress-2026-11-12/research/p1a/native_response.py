"""Native prediction queries with delayed, factual feedback values.

NativeResponseRegulator shares the host prediction and memory path between
the source-conditioned feedback candidate and its canonical reference.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ContactTransferEvidence:
    """Sufficient statistics for one shared two-entity displacement relation."""

    gram: torch.Tensor
    cross: torch.Tensor
    energy: torch.Tensor
    count: int = 0

    @classmethod
    def empty(cls, reference):
        zero = reference.new_zeros((2, 2), dtype=torch.float64)
        return cls(zero, zero.clone(), zero.new_zeros(()))

    def observe_burst(self, arrived_displacements):
        """Commit three consecutive actual [pusher xy, object xy] displacements."""
        if arrived_displacements.shape != (3, 4):
            raise ValueError("A completed chronological burst needs three displacement pairs")
        first, middle, last = arrived_displacements.to(self.gram)
        pusher, object_motion = (first - 2 * middle + last).split(2)
        return ContactTransferEvidence(
            self.gram + torch.outer(pusher, pusher),
            self.cross + torch.outer(object_motion, pusher),
            self.energy + object_motion.square().sum(),
            self.count + 1,
        )

    def clear(self):
        return self.empty(self.gram)

    def relation(self):
        inverse = torch.linalg.pinv(self.gram, hermitian=True)
        transfer = self.cross @ inverse
        support = self.gram @ inverse
        unexplained = (self.energy - (transfer * self.cross).sum()).clamp_min(0.)
        return transfer, support, unexplained


@dataclass(frozen=True)
class FaceSlipEvidence:
    """A contact face and the observed slip relation since arriving on it."""

    face: torch.Tensor
    pusher: torch.Tensor
    angle: torch.Tensor
    tangent_position: torch.Tensor
    drive_energy: torch.Tensor
    cross: torch.Tensor
    slip_energy: torch.Tensor
    transitions: torch.Tensor

    @classmethod
    def start(cls, face, pusher, angle, tangent_position):
        zero = torch.zeros_like(angle)
        return cls(face, pusher, angle, tangent_position, zero, zero, zero, torch.zeros_like(face))

    def observe(self, face, pusher, angle, tangent_position, tangents, radius):
        """Use the next actually arrived pose. A new face begins with empty statistics."""
        same_face = (face >= 0) & (face == self.face)
        t = tangents[face.clamp_min(0)]
        cos, sin = self.angle.cos(), self.angle.sin()
        world_tangent = torch.stack((cos*t[..., 0]-sin*t[..., 1], sin*t[..., 0]+cos*t[..., 1]), dim=-1)
        drive = ((pusher - self.pusher) * world_tangent).sum(-1)
        delta = angle - self.angle
        slip = tangent_position - self.tangent_position + radius * torch.atan2(delta.sin(), delta.cos())
        return FaceSlipEvidence(
            face, pusher, angle, tangent_position,
            torch.where(same_face, self.drive_energy + drive.square(), 0.),
            torch.where(same_face, self.cross + drive*slip, 0.),
            torch.where(same_face, self.slip_energy + slip.square(), 0.),
            torch.where(same_face, self.transitions + 1, 0),
        )

    def estimate(self):
        identified = (self.face >= 0) & (self.drive_energy > torch.finfo(self.angle.dtype).eps)
        scale = torch.where(identified, self.drive_energy, 1.)
        slope = torch.where(identified, self.cross / scale, 0.)
        residual = (self.slip_energy - slope*self.cross).clamp_min(0.)
        return slope, identified, residual


def project_swept_cone(displacement, axis, half_angle):
    """Project onto the cone about unit axis, with angular half-width below pi/2."""
    tangent = torch.stack((-axis[..., 1], axis[..., 0]), dim=-1)
    forward = (displacement * axis).sum(-1)
    sideways = (displacement * tangent).sum(-1)
    cos, sin = half_angle.cos(), half_angle.sin()
    inside = (forward >= 0.) & (sideways.abs()*cos <= forward*sin)
    length = (forward*cos + sideways.abs()*sin).clamp_min(0.)
    edge = cos[..., None]*axis + (sin*sideways.sign())[..., None]*tangent
    return torch.where(inside[..., None], displacement, length[..., None]*edge)


class ContactTransferPort:
    """Close object action contrasts at the existing LeWM affine prediction head."""

    def __init__(self, host, readout, reference_action, *, physical_offset=None):
        self.host = host
        self.head = host.pred_proj.net[-1]
        if host.training or not isinstance(self.head, torch.nn.Linear):
            raise ValueError("A LeWM eval-mode model with a final affine head is required")
        if readout.shape != (6, self.head.out_features) or reference_action.ndim != 1:
            raise ValueError("Readout needs pusher xy, object xy and two angle rows")
        self.reference_action = reference_action
        self.readout = readout.double()
        self.physical_offset = physical_offset
        self.head_readout = self.readout @ self.head.weight.double()
        basis, triangular = torch.linalg.qr(self.head_readout.T)
        if int(torch.linalg.matrix_rank(triangular)) != 6:
            raise ValueError("The frozen native head cannot independently change the object readout")
        self.lift = basis @ torch.linalg.solve_triangular(
            triangular.T, torch.eye(6, dtype=basis.dtype, device=basis.device)[:, 2:4], upper=False
        )

    def _forecast(self, windows, requests, edit):
        def replace_last(_, args):
            q = args[0].reshape(*requests.shape[:-1], -1)
            changed = q.clone()
            changed[..., -1, :] = edit(q)
            return (changed.reshape_as(args[0]),)

        with ExitStack() as hooks:
            hooks.callback(self.head.register_forward_pre_hook(replace_last).remove)
            output = self.host.predict(windows, requests.flatten(0, 1))
        return output.reshape(*requests.shape[:-1], -1)

    def predict(self, states, actions, relation, *, at_reference, trace=False):
        """Predict [roots,candidates,history,latent] from one aligned window per root."""
        roots, candidates, history, width = actions.shape
        if states.shape[:2] != (roots, history) or at_reference.shape != (roots, candidates):
            raise ValueError("State windows and raw reference flags must align with candidate actions")
        prefix = actions[:, :1, :-1].expand(roots, candidates, history - 1, width)
        if not torch.equal(actions[:, :, :-1], prefix):
            raise ValueError("Candidate actions must share the executed prefix at each root")
        reference = torch.cat((actions[:, 0, :-1], self.reference_action.expand(roots, 1, width)), dim=1)
        requests = torch.cat((reference[:, None], actions), dim=1)
        windows = states.repeat_interleave(candidates + 1, dim=0)
        transfer, support, _ = relation
        diagnostics = {}

        def edit(q):
            last = q[:, 1:, -1]
            effect = (last.double() - q[:, :1, -1].double()) @ self.head_readout.T
            pusher, object_motion, _ = effect.split(2, dim=-1)
            unobserved = pusher - pusher @ support.T
            roundoff = 128 * torch.finfo(q.dtype).eps * pusher.norm(dim=-1)
            apply = (unobserved.norm(dim=-1) <= roundoff) & (pusher.norm(dim=-1) > 0.) & ~at_reference
            requested = pusher @ transfer.T - object_motion
            correction = torch.where(apply[..., None], requested, 0.)
            adjusted = last + (correction @ self.lift.T).to(q)
            if trace:
                diagnostics.update(effect=effect, support=apply, correction=correction,
                                   native=F.linear(q[:, 1:], self.head.weight, self.head.bias),
                                   reference=F.linear(q[:, 0, -1], self.head.weight, self.head.bias))
            return torch.cat((q[:, :1, -1], adjusted), dim=1)

        output = self._forecast(windows, requests, edit)
        return output[:, 1:], diagnostics

    def predict_object_pose(self, states, actions, adjust):
        """Apply an object translation from the absolute native pose, with no reference query."""
        if self.physical_offset is None or states.shape[:2] != (actions.shape[0], actions.shape[2]):
            raise ValueError("Absolute adjustment needs calibrated offsets and aligned native windows")
        offset = self.physical_offset
        if self.head.bias is not None:
            offset = offset + self.readout @ self.head.bias.double()

        def edit(q):
            last = q[..., -1, :]
            pose = last.double() @ self.head_readout.T + offset
            shift = adjust(pose)
            return last + (shift @ self.lift.T).to(last)

        return self._forecast(states.repeat_interleave(actions.shape[1], dim=0), actions, edit)


def condition_log_weights(log_weights, hypothesis_mse, temperature):
    """Condition only on an arrived factual transition (generalized likelihood)."""
    if temperature <= 0 or log_weights.shape != hypothesis_mse.shape:
        raise ValueError("positive temperature and matching [B,K] scores required")
    updated = log_weights - hypothesis_mse / (2 * temperature)
    return updated - updated.logsumexp(dim=-1, keepdim=True)


class LegacyGainRegulator:
    def __init__(self, model, neutral_action_embedding, mode="anchored"):
        if mode not in ("anchored", "unanchored"):
            raise ValueError("mode must be anchored or unanchored")
        self.model = model
        self.transformer = model.predictor.predictor.transformer
        self.block = self.transformer.layers[0]
        if model.training:
            raise ValueError("The frozen host must be in eval mode")
        neutral = neutral_action_embedding
        if neutral.ndim != 1:
            raise ValueError("neutral embedding must be [action_embedding_dim]")
        self.neutral = neutral.detach()
        self.mode = mode
        self.calls = 0
        self.reference_block_calls = 0
        self.sample_tokens = 0

    def predict(self, z, actions, gain):
        """Native predict with a scalar or channel-wise last-token intervention.

        z/actions: [B,T,192], gain: [B] or [B,block_width]. For T>1, past
        actions and past token outputs remain unchanged. Raw physical zero
        must be normalized and embedded by the host before construction.
        No no_grad decorator: frozen weights still transmit gain gradients.
        """
        if z.ndim != 3 or actions.ndim != 3 or z.shape[:2] != actions.shape[:2]:
            raise ValueError("states and action embeddings must share [B,T]")
        if gain.ndim not in (1, 2) or gain.shape[0] != z.shape[0]:
            raise ValueError("gain must be [B] or [B,block_width]")
        scalar = gain[:, None] if gain.ndim == 1 else gain
        original = self.block.forward
        neutral = self.transformer.cond_proj(self.neutral[None, None])

        def regulated(x, c):
            native = original(x, c)
            if scalar.shape[-1] not in (1, native.shape[-1]):
                raise ValueError("gain width does not match the native block")
            if self.mode == "anchored":
                reference_conditions = torch.cat(
                    (c[:, :-1], neutral.expand(len(c), 1, -1)), dim=1
                )
                reference = original(x, reference_conditions)
                response = native[:, -1] - reference[:, -1]
                self.reference_block_calls += 1
            else:
                response = native[:, -1] - x[:, -1]
            last = native[:, -1] + scalar * response
            return torch.cat((native[:, :-1], last[:, None]), dim=1)

        self.calls += 1
        self.sample_tokens += z.shape[0] * z.shape[1]
        with patch.object(self.block, "forward", regulated):
            return self.model.predict(z, actions)


@dataclass(frozen=True)
class FeedbackMemory:
    """Per-head fulfilled query/value records, each [B, heads, records, width]."""

    queries: torch.Tensor
    values: torch.Tensor

    @property
    def size(self):
        return self.queries.shape[2]

    def read(self, query):
        """Read [B,heads,width]; the extra zero logit is a zero-value record."""
        if query.ndim != 3 or self.queries.shape != self.values.shape:
            raise ValueError("query [B,H,D] and matching memory tensors required")
        if (self.queries.shape[:2] != query.shape[:2]
                or self.queries.shape[-1] != query.shape[-1]):
            raise ValueError("query and memory head dimensions differ")
        if self.size == 0:
            return torch.zeros_like(query)
        logits = torch.einsum("bhd,bhnd->bhn", query, self.queries)
        logits = logits / query.shape[-1] ** 0.5
        logits = torch.cat((torch.zeros_like(logits[..., :1]), logits), dim=-1)
        weights = logits.softmax(dim=-1)[..., 1:]
        return torch.einsum("bhn,bhnd->bhd", weights, self.values)

    def append(self, query, value, capacity):
        """Return a new state; calling predictions never mutates public memory."""
        if capacity < 1 or query.shape != value.shape:
            raise ValueError("positive capacity and matching query/value required")
        queries = torch.cat((self.queries, query[:, :, None]), dim=2)
        values = torch.cat((self.values, value[:, :, None]), dim=2)
        return FeedbackMemory(queries[:, :, -capacity:], values[:, :, -capacity:])


@dataclass(frozen=True)
class SourceAttention:
    """Native quantities for one issued last-token query."""

    output: torch.Tensor
    log_partition: torch.Tensor
    shift: torch.Tensor
    scale: torch.Tensor
    slot: int


@dataclass(frozen=True)
class IssuedPrediction:
    """One pending last-token prediction; it is not yet factual memory."""

    latent: torch.Tensor
    query: torch.Tensor
    readout: torch.Tensor
    source: SourceAttention | None = None


@dataclass(frozen=True)
class ConsequenceAnswer:
    key: torch.Tensor
    value: torch.Tensor
    share: torch.Tensor
    answer: torch.Tensor


def source_answer_difference(source_output, actual, predicted):
    """Content change and competition change of two augmented answers."""
    content = actual.share * (actual.value - predicted.value)
    competition = (actual.share - predicted.share) * (predicted.value - source_output)
    return content, competition


class NativeResponseRegulator:
    """Reuse a native attention query as the address of its delayed feedback.

    Only the last token receives feedback, before the original attention
    output projection and action gate. The neutral embedding must come from
    a physical zero action through the official normalizer and action encoder.
    Capacity 64 is an explicit FIFO space bound, not learned forgetting.
    """

    def __init__(self, model, neutral_action_embedding, capacity=64, *, feedback="canonical"):
        if model.training:
            raise ValueError("Host must be in eval mode, including during learning")
        if neutral_action_embedding.ndim != 1 or capacity < 1:
            raise ValueError("neutral embedding [D] and positive capacity required")
        if feedback not in ("canonical", "source"):
            raise ValueError("feedback must be canonical or source")
        self.model = model
        self.predictor = model.predictor.predictor
        self.transformer = self.predictor.transformer
        self.block = self.transformer.layers[0]
        self.attention = self.block.attn
        self.neutral = neutral_action_embedding.detach()
        self.capacity = capacity
        self.feedback = feedback
        self.heads = self.attention.heads
        self.inner_dim = self.attention.to_qkv.out_features // 3
        self.head_width = self.inner_dim // self.heads
        self.calls = 0
        self.feedback_feature_calls = 0
        self.source_capture_calls = 0

    def empty_memory(self, batch_size):
        reference = self.attention.to_qkv.weight
        empty = reference.new_empty(batch_size, self.heads, 0, self.head_width)
        return FeedbackMemory(empty, empty.clone())

    def feedback_features(self, latent):
        """Native canonical value features, not an inverse of the predictor.

        Evaluate the same feature map separately on actual and predicted
        successors. Its nonlinear normalizations forbid replacing this by
        a linear map applied directly to a latent residual.
        """
        if latent.ndim != 2:
            raise ValueError("successor latent must be [B,D]")
        self.feedback_feature_calls += 1
        c = self.transformer.cond_proj(self.neutral[None])
        shift, scale, _, _, _, _ = self.block.adaLN_modulation(c).chunk(6, dim=-1)
        u = self._consequence_input(latent, 0, shift, scale)
        value = self._project_consequence(u, 2 * self.inner_dim)
        return value.reshape(len(latent), self.heads, self.head_width)

    def _consequence_input(self, latent, slot, shift, scale):
        x = self.transformer.input_proj(latent + self.predictor.pos_embedding[:, slot])
        return self.attention.norm(self.block.norm1(x) * (1 + scale) + shift)

    def _project_consequence(self, u, offset):
        weight = self.attention.to_qkv.weight[offset:]
        bias = self.attention.to_qkv.bias
        return F.linear(u, weight, None if bias is None else bias[offset:])

    def consequence_answer(self, issued, latent):
        """Answer the issued query with one consequence K/V added to its context."""
        source = issued.source
        if source is None:
            raise ValueError("Consequence readback requires retained source attention")
        if latent.shape != issued.latent.shape:
            raise ValueError("Consequence must have the issued latent shape [B,D]")
        self.feedback_feature_calls += 1
        u = self._consequence_input(latent, source.slot, source.shift, source.scale)
        key, value = self._project_consequence(u, self.inner_dim).chunk(2, dim=-1)
        key = key.reshape_as(issued.query)
        value = value.reshape_as(issued.query)
        logit = (issued.query * key).sum(dim=-1) / self.head_width ** 0.5
        share = torch.sigmoid(logit - source.log_partition)[..., None]
        answer = source.output + share * (value - source.output)
        return ConsequenceAnswer(key, value, share, answer)

    def predict(self, z, actions, memory, *, retain_source=True):
        """Read-only prediction; retain issuance statistics for factual fulfilment."""
        if z.ndim != 3 or actions.ndim != 3 or z.shape[:2] != actions.shape[:2]:
            raise ValueError("native histories must share [B,T]")
        if self.model.training:
            raise ValueError("Host eval mode is required for persistent coordinates")
        captured = {}
        save_source = self.feedback == "source" and retain_source

        def capture_modulation(module, inputs, output):
            shift, scale, *_ = output[:, -1].chunk(6, dim=-1)
            captured["shift"] = shift.clone()
            captured["scale"] = scale.clone()

        def capture_query(module, inputs, output):
            query = output[..., :self.inner_dim][:, -1]
            query = query.reshape(len(z), self.heads, self.head_width)
            if save_source:
                query = query.clone()
                keys = output[..., self.inner_dim:2 * self.inner_dim].unflatten(
                    -1, (self.heads, self.head_width))
                logits = torch.einsum("bhd,bthd->bht", query, keys) / self.head_width ** 0.5
                captured["log_partition"] = logits.logsumexp(dim=-1)
            captured["query"] = query
            captured["readout"] = memory.read(query)

        def inject_feedback(module, inputs):
            head_output = inputs[0]
            if save_source:
                captured["output"] = head_output[:, -1].reshape_as(captured["query"]).clone()
            readout = captured["readout"].reshape(len(z), 1, self.inner_dim)
            return (torch.cat((head_output[:, :-1],
                               head_output[:, -1:] + readout), dim=1),)

        with ExitStack() as hooks:
            if save_source:
                hooks.callback(self.block.adaLN_modulation.register_forward_hook(capture_modulation).remove)
            hooks.callback(self.attention.to_qkv.register_forward_hook(capture_query).remove)
            hooks.callback(self.attention.to_out.register_forward_pre_hook(inject_feedback).remove)
            prediction = self.model.predict(z, actions)[:, -1]
        self.calls += 1
        source = None
        if save_source:
            self.source_capture_calls += 1
            source = SourceAttention(captured["output"], captured["log_partition"],
                                     captured["shift"], captured["scale"], z.shape[1] - 1)
        return IssuedPrediction(prediction, captured["query"], captured["readout"], source)

    def observe(self, memory, issued, actual_next):
        """Fulfil the executed prediction once its real successor has arrived.

        Use the readout from issuance, even if other writes occurred meanwhile.
        The source rule differentiates the full numeric recurrence with a
        stopped factual label. The canonical reference keeps its old derivative.
        """
        if actual_next.shape != issued.latent.shape:
            raise ValueError("actual successor and issued latent shapes differ")
        if self.feedback == "source":
            actual = self.consequence_answer(issued, actual_next.detach())
            predicted = self.consequence_answer(issued, issued.latent)
            content, competition = source_answer_difference(issued.source.output, actual, predicted)
            innovation = content + competition
        else:
            innovation = (self.feedback_features(actual_next)
                          - self.feedback_features(issued.latent.detach()))
        target = issued.readout + innovation
        return memory.append(issued.query, target, self.capacity)

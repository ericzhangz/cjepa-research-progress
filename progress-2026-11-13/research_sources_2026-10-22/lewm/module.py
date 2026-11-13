import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from typing import NamedTuple

def validate_first_block_memory(first_block_memory, memory_site, reference):
    """Validate an explicit per-example memory in native hidden coordinates."""
    if (first_block_memory is None) != (memory_site is None):
        raise ValueError('first_block_memory and memory_site must be supplied together')
    if first_block_memory is None:
        return
    if memory_site not in ('condition', 'input'):
        raise ValueError('memory_site must be condition or input')
    if (reference.ndim != 3 or reference.shape[0] < 1 or reference.shape[1] < 1
            or reference.shape[2] != 192):
        raise ValueError('Native memory requires nonempty [batch,time,192] hidden states')
    if (not isinstance(first_block_memory, torch.Tensor)
            or first_block_memory.shape != (reference.shape[0], 192)):
        raise ValueError('first_block_memory must have shape [batch,192]')
    if (not first_block_memory.is_floating_point()
            or first_block_memory.dtype != reference.dtype
            or first_block_memory.device != reference.device):
        raise ValueError('Native memory must share the hidden-state floating dtype and device')
    if not torch.isfinite(first_block_memory).all():
        raise ValueError('first_block_memory contains nonfinite input')


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, *, response_inputs=False):
        if response_inputs:
            preactivation = self.net[1](self.net[0](x))
            x = preactivation
            for index in range(2, len(self.net)):
                x = self.net[index](x)
            return x, preactivation
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, *, response_inputs=False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        normalized = self.norm2(x)
        mlp_input = modulate(normalized, shift_mlp, scale_mlp)
        if response_inputs:
            mlp, preactivation = self.mlp(mlp_input, response_inputs=True)
            h = x[:, -1] + gate_mlp[:, -1] * mlp[:, -1]
            return h, normalized[:, -1], c[:, -1], gate_mlp[:, -1], preactivation[:, -1]
        x = x + gate_mlp * self.mlp(mlp_input)
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None, first_block_memory=None, memory_site=None, *, response_inputs=False):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        validate_first_block_memory(first_block_memory, memory_site, x)
        if first_block_memory is not None:
            if not self.layers:
                raise ValueError('Native memory requires a first transformer block')
            if memory_site == 'condition' and (
                    isinstance(self.layers[0], Block) or c is None or c.shape != x.shape
                    or c.dtype != x.dtype or c.device != x.device):
                raise ValueError('Condition memory requires matching native conditional states')
            if memory_site == 'input':
                x = x + first_block_memory[:, None]

        for index, block in enumerate(self.layers):
            condition = c
            if first_block_memory is not None and memory_site == 'condition' and index == 0:
                condition = c + first_block_memory[:, None]
            if response_inputs and index == len(self.layers) - 1:
                return block(x, condition, response_inputs=True)
            x = block(x) if isinstance(block, Block) else block(x, condition)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c, first_block_memory=None, memory_site=None, *, response_inputs=False):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        if response_inputs:
            if first_block_memory is not None or memory_site is not None:
                raise ValueError("Response inputs cannot be combined with the first-block memory")
            return self.transformer(x, c, response_inputs=True)
        if first_block_memory is None and memory_site is None:
            x = self.transformer(x, c)
        else:
            x = self.transformer(x, c, first_block_memory=first_block_memory,
                                 memory_site=memory_site)
        return x


class IssuedResponse(NamedTuple):
    log_weights: torch.Tensor
    means: torch.Tensor
    reset_logit: torch.Tensor


class NativeResponseMixture(nn.Module):
    """One native GELU coordinate and three fixed offsets, as in §16."""

    def __init__(self, transformer, head, latent_scale=None, sigma_bounds=None):
        super().__init__()
        if not transformer.layers or not isinstance(transformer.layers[-1], ConditionalBlock):
            raise ValueError("A final native ConditionalBlock is required")
        if not isinstance(transformer.output_proj, (nn.Identity, nn.Linear)):
            raise ValueError("The native output projection must be affine")
        first, batch_norm, activation, last = head.net
        if not (isinstance(first, nn.Linear) and isinstance(batch_norm, nn.BatchNorm1d)
                and batch_norm.track_running_stats and isinstance(activation, nn.GELU)
                and isinstance(last, nn.Linear)):
            raise ValueError("Expected a native Linear/BatchNorm/GELU/Linear prediction head")
        reference = first.weight
        if (latent_scale is None) != (sigma_bounds is None):
            raise ValueError("Gaussian evidence requires both latent scales and sigma bounds")
        scale = bounds = None
        if latent_scale is not None:
            scale = torch.as_tensor(latent_scale, device=reference.device, dtype=reference.dtype)
            bounds = torch.as_tensor(sigma_bounds, device=reference.device, dtype=reference.dtype)
            if (scale.shape != (last.out_features,) or not torch.isfinite(scale).all()
                    or (scale <= 0).any()):
                raise ValueError("Latent scale must contain one finite positive value per output")
            if (bounds.shape != (2,) or not torch.isfinite(bounds).all()
                    or not 0 < bounds[0] < bounds[1]):
                raise ValueError("Sigma bounds must be ordered positive offline calibration values")
            scale = scale.detach().clone()
            bounds = bounds.log().detach().clone()
        width = transformer.norm.normalized_shape[0]
        self.readout = nn.Linear(2 * width, 2, device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            self.readout.weight.zero_()
            self.readout.bias.copy_(reference.new_tensor([0.01, -6.0]))
        self.log_sigma_coordinate = None if bounds is None else nn.Parameter(reference.new_zeros(()))
        self.register_buffer("latent_scale", scale)
        self.register_buffer("log_sigma_bounds", bounds)
        self.register_buffer("log_prior", reference.new_tensor([0.25, 0.5, 0.25]).log())
        self.register_buffer("offsets", reference.new_tensor([-1.0, 0.0, 1.0]))

    def initial_state(self, batch_size):
        return self.log_prior.expand(batch_size, -1).clone()

    def prepare_state(self, log_weights, batch_size):
        if log_weights.shape != (batch_size, 3):
            raise ValueError("Expected one [3] log probability vector per request")
        log_weights = log_weights.to(self.log_prior)
        normalizer = log_weights.logsumexp(dim=-1, keepdim=True)
        if not torch.isfinite(normalizer).all():
            raise ValueError("Each response state must have finite, nonzero total mass")
        return log_weights - normalizer

    def advance(self, log_weights, reset_logit):
        keep = F.logsigmoid(-reset_logit)[:, None] + log_weights
        reset = F.logsigmoid(reset_logit)[:, None] + self.log_prior
        return torch.logaddexp(keep, reset)

    def forward(self, inputs, transformer, head, log_weights, *, factual=False):
        h, normalized, condition, gate, preactivation = inputs
        first, batch_norm, activation, last = head.net
        if transformer.training or head.training or batch_norm.training:
            raise ValueError("Use evaluation-mode native modules to keep dropout and BN fixed")

        b, reset_logit = self.readout(torch.cat((normalized, condition), dim=-1)).unbind(-1)
        mlp = transformer.layers[-1].mlp.net
        u = preactivation[:, :1]
        offsets = mlp[2](u + b[:, None] * self.offsets) - mlp[2](u)
        direction = gate * mlp[4].weight[:, 0]
        centered_h = h - h.mean(dim=-1, keepdim=True)
        centered_v = direction - direction.mean(dim=-1, keepdim=True)
        # Accumulate the three scalar coefficients in double precision before
        # the subtraction implicit in the quadratic variance formula.
        h64, v64 = centered_h.double(), centered_v.double()
        delta = offsets.double()
        variance = (h64.square().mean(-1, keepdim=True)
                    + 2 * delta * (h64 * v64).mean(-1, keepdim=True)
                    + delta.square() * v64.square().mean(-1, keepdim=True))
        scale = (variance.clamp_min(0) + transformer.norm.eps).sqrt().to(h)
        projection = transformer.output_proj
        directions = torch.stack((centered_h, centered_v), dim=1) * transformer.norm.weight
        if isinstance(projection, nn.Linear):
            directions = F.linear(directions, projection.weight)
        gain = (batch_norm.running_var + batch_norm.eps).rsqrt()
        if batch_norm.affine:
            gain = gain * batch_norm.weight
        base, change = (F.linear(directions, first.weight) * gain).unbind(1)
        bias = batch_norm(first(projection(transformer.norm.bias[None])))
        components = activation(bias + (base[:, None] + offsets[..., None] * change[:, None])
                                / scale[..., None])
        prediction = last((log_weights.exp()[..., None] * components).sum(dim=1))
        issue = IssuedResponse(log_weights, last(components), reset_logit) if factual else None
        return prediction, self.advance(log_weights, reset_logit), issue

    def observe(self, issued, arrived):
        """Apply the actual arrival to its issued prediction, then advance the mode."""
        if self.latent_scale is None:
            raise ValueError("Use condition() with the configured observation likelihood")
        if arrived.shape != issued.means[:, 0].shape:
            raise ValueError("The arrived encoding must match the issued batch and latent width")
        log_sigma = torch.lerp(*self.log_sigma_bounds.unbind(), self.log_sigma_coordinate.sigmoid())
        residual = (arrived[:, None] - issued.means) / self.latent_scale
        log_likelihood = -0.5 * residual.square().sum(-1) * (-2 * log_sigma).exp()
        posterior, advanced, log_evidence = self.condition(issued, log_likelihood)
        nll = (-log_evidence / arrived.size(-1) + log_sigma
               + self.latent_scale.log().mean() + 0.9189385332046727)
        return posterior, advanced, nll

    def condition(self, issued, log_likelihood):
        """Use likelihoods from the declared observation channel for this arrival."""
        if log_likelihood.shape != issued.log_weights.shape:
            raise ValueError("Expected one observation log likelihood per issued mode")
        log_joint = issued.log_weights + log_likelihood
        log_evidence = log_joint.logsumexp(-1, keepdim=True)
        posterior = log_joint - log_evidence
        return posterior, self.advance(posterior, issued.reset_logit), log_evidence.squeeze(-1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LiPO / listwise preference objectives with optional integrated DPOP-style anchoring.

Implemented objectives
----------------------
- lipo_pl_*     : list-MLE / Plackett-Luce objective (Eq. 7 in the LiPO paper)
- lipo_lambda_* : LiPO-λ / LambdaLoss objective (Eq. 8 in the LiPO paper)

Integrated anchor
-----------------
This module supports an integrated positive anchor similar in spirit to DPOP-BT:
each model score

    s_i = beta * (logπ_i - logπref_i)

can be replaced by the anchored score

    s_tilde_i = s_i - anchor_lambda * a_i * max(0, logπref_i - logπ_i)

where a_i are anchor weights (e.g. top1 / uniform / rank / custom tensor).

Important
---------
If you use the integrated anchor (anchor_lambda > 0), do NOT also add the old
external regularizer on top of the loss. That would double-count the anchor.

Reference:
LiPO: Listwise Preference Optimization through Learning-to-Rank
https://arxiv.org/abs/2402.01878
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _zeros_scalar_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=x.device, dtype=x.dtype)


def _as_1d_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={tuple(x.shape)}")
    return x


def _predicted_ranks(scores: torch.Tensor) -> torch.Tensor:
    """
    Return 1-based ranks induced by `scores`, highest score = rank 1.
    """
    scores = _as_1d_tensor(scores, "scores")
    n = scores.numel()
    if n == 0:
        return torch.empty(0, device=scores.device, dtype=torch.long)

    try:
        order = torch.argsort(scores, descending=True, stable=True)
    except TypeError:
        order = torch.argsort(scores, descending=True)

    ranks = torch.empty(n, device=scores.device, dtype=torch.long)
    ranks[order] = torch.arange(1, n + 1, device=scores.device, dtype=torch.long)
    return ranks


def _score_vector(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    pi_logprobs = _as_1d_tensor(pi_logprobs, "pi_logprobs")
    ref_logprobs = _as_1d_tensor(ref_logprobs, "ref_logprobs")

    if pi_logprobs.shape != ref_logprobs.shape:
        raise ValueError(
            f"pi_logprobs and ref_logprobs must have same shape, "
            f"got {tuple(pi_logprobs.shape)} vs {tuple(ref_logprobs.shape)}"
        )
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")

    return beta * (pi_logprobs - ref_logprobs)


def _label_pair_mask(labels: torch.Tensor) -> torch.Tensor:
    """
    mask[i, j] = True iff label_i > label_j
    """
    labels = _as_1d_tensor(labels, "labels")
    return labels[:, None] > labels[None, :]


def _dcg_lambdaweight_matrix(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the LambdaLoss DCG weights matrix:

      Δ_{i,j} = |G_i - G_j| * |1/D(τ(i)) - 1/D(τ(j))|

    where:
      G_i = 2^{label_i} - 1
      D(rank) = log(1 + rank)
      τ(i) = predicted rank position induced by current scores

    Returns a dense [L, L] matrix.
    """
    scores = _as_1d_tensor(scores, "scores")
    labels = _as_1d_tensor(labels, "labels")
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and labels must have same shape, "
            f"got {tuple(scores.shape)} vs {tuple(labels.shape)}"
        )

    ranks = _predicted_ranks(scores)  # 1-based
    ranks_f = ranks.to(dtype=scores.dtype)

    gains = torch.pow(2.0, labels) - 1.0
    discounts = torch.log1p(ranks_f)  # log(1 + rank)
    inv_discounts = 1.0 / discounts

    gain_diffs = (gains[:, None] - gains[None, :]).abs()
    discount_diffs = (inv_discounts[:, None] - inv_discounts[None, :]).abs()

    return gain_diffs * discount_diffs


# ---------------------------------------------------------------------
# Integrated DPOP-style anchor helpers
# ---------------------------------------------------------------------

def _anchor_weights_vector(
    length: int,
    device: torch.device,
    dtype: torch.dtype,
    weights="top1",
) -> torch.Tensor:
    """
    Return anchor weights a_i used in:

      s_tilde_i = s_i - anchor_lambda * a_i * max(0, logπref_i - logπ_i)

    Supported:
      - None / "none" : no anchor
      - "top1"        : anchor only the first item
      - "uniform"     : equal weight on all items
      - "rank"        : normalized 1/log2(rank+1)-style decay over list positions
      - tensor [L]    : explicit user-provided nonnegative weights
    """
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")

    if length == 0:
        return torch.empty(0, device=device, dtype=dtype)

    if torch.is_tensor(weights):
        w = _as_1d_tensor(weights, "anchor_weights").to(device=device, dtype=dtype)
        if w.numel() != length:
            raise ValueError(
                f"anchor_weights tensor must have length={length}, got {w.numel()}"
            )
        if torch.any(w < 0):
            raise ValueError("anchor_weights tensor must be nonnegative")
        return w

    if weights is None or weights == "none":
        return torch.zeros(length, device=device, dtype=dtype)

    if weights == "top1":
        w = torch.zeros(length, device=device, dtype=dtype)
        w[0] = 1.0
        return w

    if weights == "uniform":
        return torch.ones(length, device=device, dtype=dtype) / length

    if weights == "rank":
        raw = torch.tensor(
            [1.0 / math.log2(i + 2) for i in range(length)],
            device=device,
            dtype=dtype,
        )
        return raw / raw.sum()

    raise ValueError(
        f"Unknown anchor weight scheme: {weights!r}. "
        f"Choose None, 'none', 'top1', 'uniform', 'rank', or a tensor."
    )


def _anchored_score_vector(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    beta: float,
    anchor_lambda: float = 0.0,
    anchor_weights="top1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      s_tilde      : anchored scores
      slope_ratio  : elementwise factor such that if coeff_i denotes dL/ds_tilde_i,
                     then

                         dL / d logπ_i = beta * (coeff_i * slope_ratio_i)

    since

      s_tilde_i = beta * (logπ_i - logπref_i)
                  - anchor_lambda * a_i * max(0, logπref_i - logπ_i)

    and therefore

      d s_tilde_i / d logπ_i
      = beta + anchor_lambda * a_i * 1[logπ_i < logπref_i]

    so

      slope_ratio_i = (d s_tilde_i / d logπ_i) / beta
                    = 1 + (anchor_lambda / beta) * a_i * 1[violation].
    """
    pi_logprobs = _as_1d_tensor(pi_logprobs, "pi_logprobs")
    ref_logprobs = _as_1d_tensor(ref_logprobs, "ref_logprobs")

    if pi_logprobs.shape != ref_logprobs.shape:
        raise ValueError(
            f"pi_logprobs and ref_logprobs must have same shape, "
            f"got {tuple(pi_logprobs.shape)} vs {tuple(ref_logprobs.shape)}"
        )
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if anchor_lambda < 0.0:
        raise ValueError(f"anchor_lambda must be >= 0, got {anchor_lambda}")

    s = _score_vector(pi_logprobs, ref_logprobs, beta)
    L = s.numel()

    if L == 0 or anchor_lambda == 0.0:
        return s, torch.ones_like(s)

    a = _anchor_weights_vector(
        length=L,
        device=s.device,
        dtype=s.dtype,
        weights=anchor_weights,
    )

    if torch.count_nonzero(a).item() == 0:
        return s, torch.ones_like(s)

    per_pos = torch.clamp(ref_logprobs - pi_logprobs, min=0.0)
    violated = (pi_logprobs < ref_logprobs).to(dtype=s.dtype)

    s_tilde = s - anchor_lambda * a * per_pos
    slope_ratio = 1.0 + (anchor_lambda / beta) * a * violated

    return s_tilde, slope_ratio


# ---------------------------------------------------------------------
# List-MLE / Plackett-Luce
# ---------------------------------------------------------------------

def lipo_pl_loss(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    beta: float,
    reduction: str = "mean",
    anchor_lambda: float = 0.0,
    anchor_weights="top1",
) -> torch.Tensor:
    """
    Eq. (7): list-MLE / Plackett-Luce loss on an already ordered list.

    With integrated anchor:
      s_tilde_i = s_i - anchor_lambda * a_i * max(0, logπref_i - logπ_i)
    """
    r, _ = _anchored_score_vector(
        pi_logprobs=pi_logprobs,
        ref_logprobs=ref_logprobs,
        beta=beta,
        anchor_lambda=anchor_lambda,
        anchor_weights=anchor_weights,
    )

    L = r.numel()
    if L < 2:
        return _zeros_scalar_like(r)

    loss = _zeros_scalar_like(r)
    for i in range(L - 1):
        denom = torch.logsumexp(r[i:], dim=0)
        loss = loss - r[i] + denom

    if reduction == "mean":
        loss = loss / (L - 1)
    elif reduction != "sum":
        raise ValueError(f"Unknown reduction={reduction!r}")

    return loss


def lipo_pl_gradient_coefficients(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    beta: float,
    reduction: str = "mean",
    anchor_lambda: float = 0.0,
    anchor_weights="top1",
) -> torch.Tensor:
    """
    Coefficients c such that:

      d L_pl / d logπ_k = beta * c_k

    This remains true even with integrated anchoring because c_k includes the
    local chain-rule factor induced by s_tilde_k.
    """
    r, slope_ratio = _anchored_score_vector(
        pi_logprobs=pi_logprobs,
        ref_logprobs=ref_logprobs,
        beta=beta,
        anchor_lambda=anchor_lambda,
        anchor_weights=anchor_weights,
    )

    L = r.numel()
    coeffs = torch.zeros_like(r)

    if L < 2:
        return coeffs

    for k in range(L):
        max_i = min(k, L - 2)
        for i in range(max_i + 1):
            subset = r[i:]
            probs = torch.softmax(subset - subset.max(), dim=0)
            coeffs[k] += probs[k - i]

        if k < L - 1:
            coeffs[k] -= 1.0

    if reduction == "mean":
        coeffs = coeffs / (L - 1)
    elif reduction != "sum":
        raise ValueError(f"Unknown reduction={reduction!r}")

    coeffs = coeffs * slope_ratio
    return coeffs


# ---------------------------------------------------------------------
# True LiPO-λ: LambdaLoss
# ---------------------------------------------------------------------

def lipo_lambda_loss(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    reduction: str = "mean",
    anchor_lambda: float = 0.0,
    anchor_weights="top1",
) -> torch.Tensor:
    """
    Eq. (8): true LiPO-λ / LambdaLoss objective.

    L = sum_{label_i > label_j} Δ_{i,j} * log(1 + exp(-(s_i - s_j)))

    With integrated anchor:
      s_tilde_i = s_i - anchor_lambda * a_i * max(0, logπref_i - logπ_i)

    All pairwise score differences are computed from s_tilde.
    The DCG Lambda weights are also computed from the rank induced by s_tilde.
    """
    s, _ = _anchored_score_vector(
        pi_logprobs=pi_logprobs,
        ref_logprobs=ref_logprobs,
        beta=beta,
        anchor_lambda=anchor_lambda,
        anchor_weights=anchor_weights,
    )

    labels = _as_1d_tensor(labels, "labels").to(device=s.device, dtype=s.dtype)

    if labels.shape != s.shape:
        raise ValueError(
            f"labels and scores must have same shape, "
            f"got {tuple(labels.shape)} vs {tuple(s.shape)}"
        )

    L = s.numel()
    if L < 2:
        return _zeros_scalar_like(s)

    pair_mask = _label_pair_mask(labels)
    pair_count = int(pair_mask.sum().item())
    if pair_count == 0:
        return _zeros_scalar_like(s)

    lambda_w = _dcg_lambdaweight_matrix(s, labels)
    score_diffs = s[:, None] - s[None, :]
    pair_losses = F.softplus(-score_diffs)  # log(1 + exp(-(s_i - s_j)))

    loss = (lambda_w * pair_losses * pair_mask.to(s.dtype)).sum()

    if reduction == "mean":
        loss = loss / pair_count
    elif reduction != "sum":
        raise ValueError(f"Unknown reduction={reduction!r}")

    return loss


def lipo_lambda_gradient_coefficients(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    reduction: str = "mean",
    anchor_lambda: float = 0.0,
    anchor_weights="top1",
) -> torch.Tensor:
    """
    Coefficients c such that:

      d L_lambda / d logπ_k = beta * c_k

    Even with integrated anchor, this remains true because c_k includes the
    local chain-rule factor induced by s_tilde_k.

    The Lambda weights use the rank permutation induced by the anchored scores.
    In practice, for a streaming backward implementation, you should still call
    this on detached score values if you want the rank-induced weights frozen
    within the step.
    """
    s, slope_ratio = _anchored_score_vector(
        pi_logprobs=pi_logprobs,
        ref_logprobs=ref_logprobs,
        beta=beta,
        anchor_lambda=anchor_lambda,
        anchor_weights=anchor_weights,
    )

    labels = _as_1d_tensor(labels, "labels").to(device=s.device, dtype=s.dtype)

    if labels.shape != s.shape:
        raise ValueError(
            f"labels and scores must have same shape, "
            f"got {tuple(labels.shape)} vs {tuple(s.shape)}"
        )

    L = s.numel()
    coeffs = torch.zeros_like(s)

    if L < 2:
        return coeffs

    pair_mask = _label_pair_mask(labels)
    pair_count = int(pair_mask.sum().item())
    if pair_count == 0:
        return coeffs

    mask_f = pair_mask.to(dtype=s.dtype)
    lambda_w = _dcg_lambdaweight_matrix(s, labels)

    score_diffs = s[:, None] - s[None, :]
    sig_neg = torch.sigmoid(-score_diffs)  # σ(-(s_i - s_j))

    # For each valid pair (i, j) with label_i > label_j:
    # d/ds_i softplus(-(s_i - s_j)) = -σ(-(s_i - s_j))
    # d/ds_j softplus(-(s_i - s_j)) = +σ(-(s_i - s_j))
    weighted = lambda_w * sig_neg * mask_f

    coeffs += -weighted.sum(dim=1)  # winner side
    coeffs +=  weighted.sum(dim=0)  # loser side

    if reduction == "mean":
        coeffs = coeffs / pair_count
    elif reduction != "sum":
        raise ValueError(f"Unknown reduction={reduction!r}")

    coeffs = coeffs * slope_ratio
    return coeffs


# ---------------------------------------------------------------------
# Legacy optional external regularizer (for ablations only)
# ---------------------------------------------------------------------

def lipo_dpop_penalty(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    lam: float = 50.0,
    weights: str = "top1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy external DPOP-style regularization penalty.

    This is kept only for ablations / comparisons against the integrated anchor.

    L_penalty = lam * sum_k w_k * max(0, logπ_ref_k - logπ_k)

    Returns:
      penalty_value : scalar tensor
      penalty_grad  : shape [L], direct gradient wrt logπ_k

    IMPORTANT:
      Do not combine this penalty with the integrated anchor unless you
      intentionally want both effects at once.
    """
    pi_logprobs = _as_1d_tensor(pi_logprobs, "pi_logprobs")
    ref_logprobs = _as_1d_tensor(ref_logprobs, "ref_logprobs")

    if pi_logprobs.shape != ref_logprobs.shape:
        raise ValueError(
            f"pi_logprobs and ref_logprobs must have same shape, "
            f"got {tuple(pi_logprobs.shape)} vs {tuple(ref_logprobs.shape)}"
        )

    L = pi_logprobs.shape[0]
    device, dtype = pi_logprobs.device, pi_logprobs.dtype

    if L == 0 or lam == 0.0:
        zero = _zeros_scalar_like(pi_logprobs)
        return zero, torch.zeros_like(pi_logprobs)

    per_pos = torch.clamp(ref_logprobs - pi_logprobs, min=0.0)

    if weights == "top1":
        w = torch.zeros(L, device=device, dtype=dtype)
        w[0] = 1.0
    elif weights == "uniform":
        w = torch.ones(L, device=device, dtype=dtype) / L
    elif weights == "rank":
        raw = torch.tensor(
            [1.0 / math.log2(i + 2) for i in range(L)],
            device=device, dtype=dtype,
        )
        w = raw / raw.sum()
    else:
        raise ValueError(
            f"Unknown weights scheme: {weights!r}. "
            f"Choose 'top1', 'uniform', or 'rank'."
        )

    penalty_value = lam * (w * per_pos).sum()
    violated = (pi_logprobs < ref_logprobs).to(dtype)
    penalty_grad = -lam * w * violated

    return penalty_value, penalty_grad

# Unperfect output penalty:
def _non_top_weights_vector(
    length: int,
    device: torch.device,
    dtype: torch.dtype,
    weights="uniform",
) -> torch.Tensor:
    """
    Build weights for non-top outputs only.

    Always:
      w[0] = 0

    Supported:
      - None / "none" : all zeros
      - "uniform"     : equal normalized weights over positions 1..L-1
      - "rank"        : rank-decay weights over positions 1..L-1
      - tensor [L]    : custom nonnegative weights; position 0 is forced to 0
    """
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")

    if length == 0:
        return torch.empty(0, device=device, dtype=dtype)

    if length == 1:
        return torch.zeros(1, device=device, dtype=dtype)

    if torch.is_tensor(weights):
        w = _as_1d_tensor(weights, "non_top_weights").to(device=device, dtype=dtype)

        if w.numel() != length:
            raise ValueError(
                f"non_top_weights tensor must have length={length}, got {w.numel()}"
            )

        if torch.any(w < 0):
            raise ValueError("non_top_weights tensor must be nonnegative")

        w = w.clone()
        w[0] = 0.0

        total = w.sum()
        if total > 0:
            w = w / total

        return w

    if weights is None or weights == "none":
        return torch.zeros(length, device=device, dtype=dtype)

    if weights == "uniform":
        w = torch.zeros(length, device=device, dtype=dtype)
        w[1:] = 1.0 / (length - 1)
        return w

    if weights == "rank":
        raw = torch.tensor(
            [0.0] + [1.0 / math.log2(i + 2) for i in range(1, length)],
            device=device,
            dtype=dtype,
        )

        total = raw.sum()
        if total > 0:
            raw = raw / total

        return raw

    raise ValueError(
        f"Unknown non-top weight scheme: {weights!r}. "
        f"Choose None, 'none', 'uniform', 'rank', or a tensor."
    )


def lipo_non_top_increase_penalty(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    lam_non_top: float = 1.0,
    weights: str = "uniform",
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Complementary non-top penalty.

    This is not a replacement for DPOP.

    It penalizes absolute increases of non-top output probabilities relative
    to the frozen reference model.

    Formula:
      L_non_top = lam_non_top * sum_{j >= 1} w_j * max(
          0,
          logπ_j - logπref_j - margin
      )

    Properties:
      - top1 receives exactly zero coefficient
      - non-top outputs receive positive coefficients when they become more
        probable than under the reference model

    Returns:
      penalty_value : scalar tensor
      penalty_grad  : shape [L], direct gradient wrt logπ_j
    """
    pi_logprobs = _as_1d_tensor(pi_logprobs, "pi_logprobs")
    ref_logprobs = _as_1d_tensor(ref_logprobs, "ref_logprobs")

    if pi_logprobs.shape != ref_logprobs.shape:
        raise ValueError(
            f"pi_logprobs and ref_logprobs must have same shape, "
            f"got {tuple(pi_logprobs.shape)} vs {tuple(ref_logprobs.shape)}"
        )

    if lam_non_top < 0.0:
        raise ValueError(f"lam_non_top must be >= 0, got {lam_non_top}")

    L = pi_logprobs.numel()
    device, dtype = pi_logprobs.device, pi_logprobs.dtype

    if L == 0 or L == 1 or lam_non_top == 0.0:
        zero = _zeros_scalar_like(pi_logprobs)
        return zero, torch.zeros_like(pi_logprobs)

    w = _non_top_weights_vector(
        length=L,
        device=device,
        dtype=dtype,
        weights=weights,
    )

    increase = torch.clamp(
        pi_logprobs - ref_logprobs - margin,
        min=0.0,
    )

    penalty_value = lam_non_top * (w * increase).sum()

    active = (pi_logprobs > ref_logprobs + margin).to(dtype)
    penalty_grad = lam_non_top * w * active

    # Absolute safety: top1 must never be suppressed by this term.
    penalty_grad[0] = 0.0

    return penalty_value, penalty_grad

def lipo_asymmetric_dpop_penalty(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    lam_top: float = 50.0,
    lam_non_top: float = 1.0,
    non_top_weights: str = "uniform",
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Asymmetric DPOP-style penalty.

    Reuses the existing lipo_dpop_penalty for top1 preservation,
    and adds a complementary non-top increase penalty.

    Top1:
      lam_top * max(0, logπref_0 - logπ_0)

    Non-top:
      lam_non_top * sum_{j >= 1} w_j * max(0, logπ_j - logπref_j - margin)
    """
    loss_top, grad_top = lipo_dpop_penalty(
        pi_logprobs=pi_logprobs,
        ref_logprobs=ref_logprobs,
        lam=lam_top,
        weights="top1",
    )

    loss_non_top, grad_non_top = lipo_non_top_increase_penalty(
        pi_logprobs=pi_logprobs,
        ref_logprobs=ref_logprobs,
        lam_non_top=lam_non_top,
        weights=non_top_weights,
        margin=margin,
    )

    return loss_top + loss_non_top, grad_top + grad_non_top


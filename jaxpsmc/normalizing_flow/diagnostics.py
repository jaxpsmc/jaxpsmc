"""Device-side flow diagnostics. All particle operations are deterministic."""

import jax
import jax.numpy as jnp

from ..particles_jax import compute_logw_and_logz_jax


def coordinates(x):
    return x


def summary(observe, x, weights):
    """Weighted means, 5th, 50th and 95th empirical percentiles per observable."""
    valid = weights > 0
    first = jnp.argmax(weights)
    values = observe(jnp.where(valid[:, None], x, x[first]))
    w = jnp.where(valid, weights, 0)
    p = w / w.sum()
    order = jnp.argsort(values, axis=0)
    sorted_values = jnp.take_along_axis(values, order, axis=0)
    cdf = jnp.cumsum(p[order], axis=0)
    q = jax.vmap(
        lambda col: jnp.searchsorted(col, jnp.array([0.05, 0.5, 0.95])),
        in_axes=1,
        out_axes=1,
    )(cdf)
    quantiles = jnp.take_along_axis(sorted_values, jnp.minimum(q, len(w) - 1), axis=0)
    result = jnp.concatenate(((p @ values)[None, :], quantiles))
    finite = jnp.all(~valid[:, None] | jnp.isfinite(values), axis=0)
    return jnp.where((w.sum() > 0) & finite, result, jnp.nan)


def history_summary(observe, state, beta):
    lw, _, mask = compute_logw_and_logz_jax(state, beta_final=beta, normalize=True)
    x = state.x.reshape(-1, state.x.shape[-1])
    return summary(observe, x, jnp.where(mask, jnp.exp(lw), 0))


def region_counts(observe, before, after):
    """Net endpoint transitions for each region over the complete mutation sweep."""
    if not getattr(observe, "component_responsibilities", False):
        return jnp.empty((2, 0), dtype=jnp.int32)
    old, new = observe(before), observe(after)
    labels = jnp.arange(old.shape[1])
    origin = jnp.argmax(old, axis=1)[:, None] == labels
    destination = jnp.argmax(new, axis=1)[:, None] == labels
    return jnp.stack(
        (
            jnp.sum(~origin & destination, axis=0, dtype=jnp.int32),
            jnp.sum(origin & ~destination, axis=0, dtype=jnp.int32),
        )
    )


def round_trip(x, recovered):
    return jnp.max(jnp.abs(x - recovered) / (1 + jnp.abs(x)), axis=-1)


def fit_stats(old, new, particles, weights, rows, count, best_epoch, epochs):
    """Use the actual split, including its exact particle weights."""
    slot = jnp.arange(len(rows))
    split = count // 2
    train = jnp.where(slot < split, weights[rows], 0)
    valid = jnp.where((slot >= split) & (slot < count), weights[rows], 0)

    def weight_stats(w):
        p = w / w.sum()
        return jnp.array([1 / (p @ p), p.max()])

    # Stream the exact holdout. Large persistent buffers often have few valid rows.
    size = count - split
    width = min(256, (len(rows) + 1) // 2)
    offsets = jnp.arange(width)

    def nll(z, ld):
        return (
            0.5 * (jnp.mean(z * z, axis=-1) + jnp.log(2 * jnp.pi))
            - ld / particles.shape[1]
        )

    def score(i, total):
        slots = i * width + offsets
        indices = rows[split + jnp.minimum(slots, size - 1)]
        x = particles[indices]
        previous = jax.lax.cond(
            old.fitted,
            lambda: nll(*old.flow.transform_and_log_det(x)),
            lambda: nll(x, 0),
        )
        fitted = nll(*new.transform_and_log_det(x))
        w = jnp.where(slots < size, weights[indices], 0)
        return total + jnp.sum(
            jnp.where(w > 0, w * jnp.stack((previous, fitted)), 0), axis=1
        )

    dtype = jnp.result_type(weights, particles)
    scores = (
        jax.lax.fori_loop(0, (size + width - 1) // width, score, jnp.zeros(2, dtype))
        / valid.sum()
    )
    slots = jnp.arange(min(32, width))
    x = particles[rows[split + jnp.minimum(slots, size - 1)]]
    z, det = new.transform_and_log_det(x)
    back, inv = new.inverse_and_log_det(z)
    mask = slots < size
    err = round_trip(x, back)
    jac = jnp.abs(det + inv)
    bad = ~jnp.isfinite(err + jac)
    return jnp.concatenate(
        (
            jnp.array([split, count - split]),
            weight_stats(train),
            weight_stats(valid),
            scores,
            jnp.array(
                [
                    best_epoch,
                    epochs,
                    mask.sum(),
                    (mask & bad).sum(),
                    jnp.nanpercentile(jnp.where(mask, err, jnp.nan), 99),
                    jnp.max(jnp.where(mask, err, 0)),
                    jnp.max(jnp.where(mask, jac, 0)),
                ]
            ),
        )
    )


FIT_FIELDS = (
    "train_rows",
    "valid_rows",
    "train_ess",
    "train_max_weight",
    "valid_ess",
    "valid_max_weight",
    "previous_nll",
    "fitted_nll",
    "best_epoch",
    "stop_epoch",
    "checked",
    "nonfinite",
    "roundtrip_p99",
    "roundtrip_max",
    "logdet_max",
)
MUT_FIELDS = (
    "attempts",
    "accepted",
    "exact_rows",
    "surrogate_rows",
    "stage1_passes",
    "nonfinite_map",
    "jump_sq",
    "audited",
    "audit_nonfinite",
    "ratio_max",
    "da_ratio_max",
    "inverse_roundtrip_max",
    "inverse_logdet_max",
)


def empty_fit(dtype):
    return jnp.full(len(FIT_FIELDS), jnp.nan, dtype)


def mutation_stats(
    previous,
    accepted,
    eligible,
    preaccepted,
    da,
    stage_ratio,
    u,
    proposal,
    theta,
    proposed_theta,
    target,
    proposed_target,
    reference,
    kernel_ratio,
    flow,
    condition,
):
    """Rebuild the theta-space target with forward flow Jacobians on up to 8 walkers."""
    n = min(8, len(u))

    def evaluate(x, z, target):
        mapped, forward = jax.vmap(
            lambda row: flow.bijection.transform_and_log_det(row, condition)
        )(x[:n])
        return target[:n] - forward - reference(z[:n]), mapped, forward

    old, _, _ = evaluate(u, theta, target)
    new, mapped, forward = evaluate(proposal, proposed_theta, proposed_target)
    _, inverse = jax.vmap(
        lambda row: flow.bijection.inverse_and_log_det(row, condition)
    )(proposed_theta[:n])
    expected = new - old
    # A proposal outside target support legitimately has a minus-infinite log ratio.
    supported = (
        eligible[:n] & jnp.isfinite(proposed_target[:n]) & jnp.isfinite(target[:n])
    )
    residual = jnp.abs(expected - kernel_ratio[:n])
    da_error = jnp.where(da, jnp.abs(stage_ratio[:n] - expected), 0)
    bad = ~jnp.isfinite(residual + da_error)
    displacement = jnp.sum(jnp.square(proposal - u), axis=1)
    invalid = ~(
        jnp.isfinite(proposal).all(axis=1) & jnp.isfinite(proposed_theta).all(axis=1)
    )
    totals = jnp.array(
        [
            len(u),
            accepted.sum(),
            eligible.sum(),
            da * eligible.sum(),
            preaccepted.sum(),
            invalid.sum(),
            jnp.where(accepted, displacement, 0).sum(),
            supported.sum(),
            (supported & bad).sum(),
        ],
        dtype=u.dtype,
    )
    maxima = jnp.stack(
        (
            jnp.max(jnp.where(supported, residual, 0)),
            jnp.max(jnp.where(supported, da_error, 0)),
            round_trip(proposed_theta[:n], mapped).max(),
            jnp.abs(inverse + forward).max(),
        )
    )
    return jnp.concatenate((previous[:9] + totals, jnp.maximum(previous[9:], maxima)))


def stage_stats(observe, state, prev_beta, rw, rs, moved, mutation, fit):
    weights = rw["weights"]
    uniform = jnp.ones_like(moved["logl"])
    populations = jnp.stack(
        (
            history_summary(observe, state, prev_beta),
            history_summary(observe, state, rw["beta"]),
            summary(observe, rw["x"], weights),
            summary(observe, rs["x"], uniform),
            summary(observe, moved["x"], uniform),
        )
    )
    historical = rw["idx"] < (state.t - 1) * state.x.shape[1]
    return {
        "beta": rw["beta"],
        "population": populations,
        "mutation": mutation,
        "fit": fit,
        "region_counts": region_counts(observe, rs["x"], moved["x"]),
        "history_weight": jnp.sum(jnp.where(historical, weights, 0)),
    }


def empty_stage(observe, x):
    n = observe(x[:1]).shape[-1]
    return {
        "beta": jnp.asarray(jnp.nan, x.dtype),
        "population": jnp.full((5, 4, n), jnp.nan, x.dtype),
        "mutation": jnp.full(len(MUT_FIELDS), jnp.nan, x.dtype),
        "fit": empty_fit(x.dtype),
        "region_counts": jnp.zeros(
            (2, n if getattr(observe, "component_responsibilities", False) else 0),
            jnp.int32,
        ),
        "history_weight": jnp.asarray(jnp.nan, x.dtype),
    }


class FlowFitError(FloatingPointError):
    """The sampler stopped before mutating with a failed fit."""

    def __init__(self, trace):
        super().__init__("Adaptive flow refresh failed before mutation")
        self.trace = trace


def failed_stage(observe, x, beta, fit):
    record = empty_stage(observe, x)
    return dict(record, beta=beta, fit=fit)

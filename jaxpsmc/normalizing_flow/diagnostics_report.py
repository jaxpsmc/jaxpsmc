"""Write one text summary and two diagnostic figures after a completed run."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .diagnostics import FIT_FIELDS, MUT_FIELDS, history_summary, summary


def write_flow_diagnostics(
    directory,
    trace,
    state,
    posterior,
    returned,
    observe,
    names,
    *,
    flow,
    kernel,
    seed,
    mode,
    delayed_acceptance,
    reference=None,
):
    """reference, when supplied, contains final reference means in observable order."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    trace = jax.device_get(trace)
    active = np.isfinite(trace["beta"])
    beta = trace["beta"][active]
    population = trace["population"][active]
    fits = dict(zip(FIT_FIELDS, trace["fit"][active].T, strict=True))
    mutation = dict(zip(MUT_FIELDS, trace["mutation"][active].T, strict=True))

    def uniform(x):
        return summary(observe, x, jnp.ones(len(x), x.dtype))

    final = np.asarray(
        jnp.stack(
            (
                history_summary(observe, state, 1.0),
                summary(observe, posterior.samples, posterior.weights),
                uniform(posterior.samples_resampled),
                uniform(jnp.asarray(returned)),
            )
        )
    )
    if final.shape[-1] != len(names):
        raise ValueError("Give one name per observable column.")
    if reference is not None and np.shape(reference) != (len(names),):
        raise ValueError("Give one reference mean per observable.")
    lines = [
        f"flow={flow}  kernel={kernel}  mode={mode}  seed={seed}  dtype={state.x.dtype}",
        f"completed mutation stages={len(beta)}  DA={delayed_acceptance}  posterior resample status={int(posterior.resample_status)}",
    ]
    if len(beta):
        lines.append(
            f"last mutation beta={beta[-1]:.9g}  final importance target beta=1"
        )
    p = posterior.weights
    lines.append(
        f"retained weight ESS={p.sum() ** 2 / jnp.sum(p * p):.6g}  "
        f"maximum retained weight={p.max():.6g}  returned rows={len(returned)}"
    )

    def vector(v):
        return np.array2string(
            np.asarray(v), precision=5, separator=", ", max_line_width=110
        )

    if len(beta):

        def total(field):
            return np.sum(mutation[field], dtype=np.float64)

        exact = total("exact_rows")
        jump = total("jump_sq") / exact if exact else np.nan
        lines += [
            (
                f"accepted/attempted={total('accepted'):.0f}/{total('attempts'):.0f}  "
                f"full likelihood rows={exact:.0f}  surrogate rows={total('surrogate_rows'):.0f}  "
                f"accepted squared jump in scaled u per full likelihood row={jump:.6g}"
            )
        ]
        if delayed_acceptance:
            lines.append(
                f"stage-one passes={total('stage1_passes'):.0f}  final accepts={total('accepted'):.0f}"
            )
        lines.append(
            f"supported proposals audited={total('audited'):.0f}  nonfinite residuals={total('audit_nonfinite'):.0f}  "
            f"nonfinite proposal coordinates={total('nonfinite_map'):.0f}"
        )
        for field in (
            "ratio_max",
            "da_ratio_max",
            "inverse_roundtrip_max",
            "inverse_logdet_max",
        ):
            if field == "da_ratio_max" and not delayed_acceptance:
                continue
            vals = mutation[field]
            i = np.argmax(np.where(np.isfinite(vals), vals, np.inf))
            lines.append(f"{field}={vals[i]:.5g}  stage={i + 1}  beta={beta[i]:.5g}")
        fitted = np.flatnonzero(np.isfinite(fits["train_rows"]))
        if fitted.size:
            # Show the fit with the greatest deterioration on its own held-out rows.
            delta = fits["fitted_nll"] - fits["previous_nll"]
            i = fitted[
                np.argmax(np.where(np.isfinite(delta[fitted]), delta[fitted], np.inf))
            ]
            lines.append(
                f"largest holdout NLL change  stage={i + 1}  beta={beta[i]:.5g}  "
                f"historical fitting weight={trace['history_weight'][active][i]:.5g}"
            )
            lines.extend(
                "  " + "  ".join(f"{k}={fits[k][i]:.6g}" for k in group)
                for group in (FIT_FIELDS[:6], FIT_FIELDS[6:10], FIT_FIELDS[10:])
            )
        failure = mutation["nonfinite_map"] + mutation["audit_nonfinite"] > 0
        failure |= np.nan_to_num(fits["nonfinite"]) > 0
        failures = np.flatnonzero(failure)
        if failures.size:
            first = failures[0]
            lines.append(
                f"First observed nonfinite diagnostic: stage={first + 1}  beta={beta[first]:.5g}"
            )
        if trace["region_counts"].shape[-1]:
            counts = trace["region_counts"][active].sum(axis=0)
            lines.append(
                f"Dominant-component changes between mutation endpoints: "
                f"entries={vector(counts[0])}  exits={vector(counts[1])}"
            )
        if reference is not None and trace["region_counts"].shape[-1]:
            delta = np.abs(np.diff(population[:, :, 0, :], axis=1))
            i, checkpoint, col = np.unravel_index(np.argmax(delta), delta.shape)
            operation = ("reweight", "retain", "resample", "mutate")[checkpoint]
            lines.append(
                f"largest observed component-mass change  stage={i + 1}  beta={beta[i]:.5g}  "
                f"{operation}  {names[col]}={population[i, checkpoint, 0, col]:.5g} -> "
                f"{population[i, checkpoint + 1, 0, col]:.5g}"
            )
    lines.append(
        "Final observable means in order: untrimmed / retained / all resampled / returned"
    )
    for i, name in enumerate(names):
        ref = "" if reference is None else f"  reference mean={reference[i]:.5g}"
        lines.append(f"  {name}  {vector(final[:, 0, i])}{ref}")
    if reference is None:
        lines.append(
            "Independent posterior reference unavailable; coverage unverified."
        )
    (directory / "flow_diagnostics.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    with PdfPages(directory / "flow_diagnostics.pdf") as pdf:
        # Paginate observables so large models retain readable axes.
        for start in range(0, len(names), 4):
            group = names[start : start + 4]
            fig = Figure(figsize=(9, 2.6 * len(group) + 0.8), layout="constrained")
            axes = fig.subplots(len(group), 1, squeeze=False)[:, 0]
            for i, (ax, name) in enumerate(zip(axes, group), start=start):
                for j, label in enumerate(
                    (
                        "history at previous beta",
                        "reweighted history",
                        "retained",
                        "resampled",
                        "mutated",
                    )
                ):
                    ax.plot(
                        np.arange(1, len(beta) + 1),
                        population[:, j, 0, i],
                        marker=".",
                        label=label,
                    )
                if reference is None:
                    ax.fill_between(
                        np.arange(1, len(beta) + 1),
                        population[:, -1, 1, i],
                        population[:, -1, 3, i],
                        alpha=0.15,
                        label="mutated 5–95%",
                    )
                    ax.plot(
                        np.arange(1, len(beta) + 1),
                        population[:, -1, 2, i],
                        linestyle=":",
                        label="mutated median",
                    )
                if reference is not None:
                    ax.scatter(
                        len(beta) + 1, reference[i], marker="*", label="final reference"
                    )
                ax.scatter(len(beta) + 1, final[-1, 0, i], marker="x", label="returned")
                ax.set(
                    xlabel="SMC mutation stage", ylabel=name, title="Observable mean"
                )
            axes[0].legend(fontsize=7, ncol=3)
            pdf.savefig(fig)
        fig = Figure(figsize=(9, 9), layout="constrained")
        axes = fig.subplots(4, 1, sharex=True)
        stage = np.arange(1, len(beta) + 1)
        axes[0].plot(stage, beta, label="beta")
        axes[0].plot(
            stage, trace["history_weight"][active], label="historical fitting weight"
        )
        axes[1].plot(
            stage,
            fits["previous_nll"] - fits["fitted_nll"],
            ".-",
            label="holdout gain (nats/dim)",
        )
        for field in ("ratio_max", "inverse_roundtrip_max", "inverse_logdet_max"):
            axes[2].plot(stage, mutation[field], ".-", label=field)
        axes[2].set_yscale("symlog", linthresh=np.finfo(state.x.dtype).eps)
        exact = mutation["exact_rows"]
        axes[3].plot(
            stage,
            np.divide(
                mutation["jump_sq"],
                exact,
                out=np.full_like(exact, np.nan),
                where=exact > 0,
            ),
            ".-",
            label="squared jump / full likelihood row",
        )
        for ax in axes:
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("SMC mutation stage")
        pdf.savefig(fig)


def write_failed_fit(directory, trace, flow, seed):
    """Record the failed refresh even though posterior selection cannot run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    beta = np.asarray(trace["beta"])
    # Successful stages always store a finite number of attempted proposals.
    stage = int(np.sum(np.isfinite(np.asarray(trace["mutation"])[:, 0])))
    metrics = np.asarray(trace["fit"][stage])
    lines = [
        f"flow={flow}  seed={seed}",
        f"Fit failed before mutation: stage={stage + 1}  beta={beta[stage]:.9g}",
        "NaN fields are unavailable or nonfinite.",
    ]
    lines.extend(
        f"{name}={value:.6g}" for name, value in zip(FIT_FIELDS, metrics, strict=True)
    )
    (directory / "flow_diagnostics.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

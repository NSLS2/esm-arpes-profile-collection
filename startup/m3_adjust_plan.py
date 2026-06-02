"""
Bluesky plan implementation of the M3 mirror pitch hill-climb.

This is the refactor of ``m3_adjust_core.m3_hill_climb`` into a proper
Bluesky plan: it ``yield from`` plan stubs instead of calling ``RE(...)``
internally, uses ``bps.rd``/``bps.mv``/``bps.sleep`` for all device I/O,
deduplicates the six identical 10-sample blocks into a ``_sample()``
helper plan, and wraps the body in ``bpp.finalize_wrapper`` so the
diagnostic axis is retracted on success, on the give-up path, **and**
on exception or RunEngine interruption.

Behavioral notes vs. the original ``m3_adjust.py:113-226``:

* **Give-up restore is fixed.** The original script issues
  ``RE(mv(mirror, M3_Ry_0))`` then falls through to code that
  immediately moves the mirror back to the last attempted position.
  Here the give-up path issues a single ``bps.mv`` to ``m3_0`` and
  returns from the body, so the mirror ends up at ``m3_0`` as
  intended.
* **Asymmetric extra step preserved.** When the direction-search
  determines ``direction == -1`` the post-search step uses
  ``2 * direction * step`` rather than ``direction * step`` (mirrors
  lines 188-192 of the original).
* **Two-loop structure preserved.** Direction search and climb remain
  separate while-loops so the algorithm maps cleanly back to the
  original.
* **CSV format unchanged.** Same five columns:
  ``eng, pgm_energy, pgm_focus, Au, M3_Ry``.
"""

import csv

import numpy as np

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp


def _sample(signal, n, delay):
    """Read ``signal`` ``n`` times spaced by ``delay``.

    Returns ``(mean, std)``. Plan-message generator.
    """
    vals = np.empty(n)
    for i in range(n):
        yield from bps.sleep(delay)
        vals[i] = yield from bps.rd(signal)
    return float(np.mean(vals)), float(np.std(vals))


def _step_and_sample(mirror, target, signal, settle, n, delay):
    """Move ``mirror`` to ``target``, settle, then sample.

    Returns ``(mean, std)``.
    """
    yield from bps.mv(mirror, target)
    yield from bps.sleep(settle)
    avg, std = yield from _sample(signal, n, delay)
    return avg, std


def m3_adjust(
    *,
    mirror,
    signal,
    diag,
    diag_in=-6,
    diag_out=2,
    step=5e-5,
    n_samples=10,
    sample_delay=0.1,
    settle_time=3.0,
    max_insignificant=5,
    csv_path=None,
    eng=None,
    pgm_energy=None,
    pgm_focus=None,
):
    """Hill-climb ``mirror`` to maximize ``signal``.

    Parameters
    ----------
    mirror : ophyd motor-like (e.g. ``M3.Ry``)
    signal : ophyd Signal-like (e.g. ``qem08.current1.mean_value``)
    diag   : ophyd motor-like (e.g. ``M4AUdiag.trans``)
    diag_in, diag_out : float
        Diagnostic insert/retract positions.
    step : float
        Mirror step size.
    n_samples, sample_delay : int, float
        Per-measurement read count and inter-read delay.
    settle_time : float
        Sleep after each mirror move before sampling.
    max_insignificant : int
        Number of insignificant direction-search steps before giving up.
    csv_path : str or None
        If set, append one row at completion (success or give-up).
    eng, pgm_energy, pgm_focus : float or None
        Pass-through values for the CSV row.

    Returns
    -------
    (float, float)
        ``(final_mirror_position, final_signal_average)``. The signal
        average is the last ``Au1_avg`` value computed by the algorithm
        (matching the value written to ``csv_path``'s 4th column).
    """

    # final_pos and final_au are populated by _body() and consumed by the
    # CSV-write tail after finalize_wrapper completes.
    final = {"pos": None, "au": None}

    def _retract_diag():
        yield from bps.mv(diag, diag_out)

    def _body():
        # --- backlash unwind ---
        m3 = yield from bps.rd(mirror)
        yield from bps.mv(mirror, m3 - 2 * step)
        yield from bps.mv(mirror, m3 + 2 * step)
        yield from bps.sleep(settle_time)
        m3_0 = yield from bps.rd(mirror)
        m3 = m3_0

        # --- insert diag, baseline sample ---
        yield from bps.mv(diag, diag_in)
        au0_avg, au0_std = yield from _sample(signal, n_samples, sample_delay)

        # --- first +step probe ---
        m3 = m3 + step
        au1_avg, au1_std = yield from _step_and_sample(
            mirror, m3, signal, settle_time, n_samples, sample_delay
        )

        # --- direction search loop ---
        direction = 0.0
        dir_found = False
        insignificant = 0
        gave_up = False

        while not dir_found:
            if abs(au1_avg - au0_avg) > (au0_std + au1_std) / 2:
                direction = +1.0 if (au1_avg - au0_avg) > 0 else -1.0
                dir_found = True
            else:
                m3 = m3 + step
                au0_avg, au0_std = au1_avg, au1_std
                # NOTE: original sleeps BEFORE the move inside this branch
                # (line 160-161). Preserved literally.
                yield from bps.sleep(settle_time)
                yield from bps.mv(mirror, m3)
                au1_avg, au1_std = yield from _sample(
                    signal, n_samples, sample_delay
                )
                insignificant += 1
                if insignificant == max_insignificant:
                    # FIX vs. original: genuine restore + early return.
                    yield from bps.mv(mirror, m3_0)
                    gave_up = True
                    dir_found = True

        if gave_up:
            final["pos"] = yield from bps.rd(mirror)
            final["au"] = au1_avg
            return  # diag retract happens in finalize_wrapper

        # --- seed sample after direction is known ---
        # Mirrors lines 182-185 of the original: re-uses Au1_values as the
        # new Au0 baseline at the current position.
        au0_avg, au0_std = yield from _sample(signal, n_samples, sample_delay)

        # --- asymmetric extra step (lines 188-192) ---
        if direction == 1.0:
            m3 = m3 + direction * step
        else:
            m3 = m3 + 2 * direction * step
        au1_avg, au1_std = yield from _step_and_sample(
            mirror, m3, signal, settle_time, n_samples, sample_delay
        )

        # --- climb loop ---
        max_found = False
        while not max_found:
            if abs(au1_avg - au0_avg) > (au0_std + au1_std) / 2:
                if (au1_avg - au0_avg) > 0:
                    m3 = m3 + direction * step
                    au0_avg, au0_std = au1_avg, au1_std
                    au1_avg, au1_std = yield from _step_and_sample(
                        mirror, m3, signal, settle_time, n_samples, sample_delay
                    )
                else:
                    max_found = True
                    m3 = m3 - direction * step
                    yield from bps.mv(mirror, m3)
            else:
                max_found = True

        final["pos"] = yield from bps.rd(mirror)
        final["au"] = au1_avg

    yield from bpp.finalize_wrapper(_body(), _retract_diag())

    # --- CSV log: append one row on completion (success or give-up) ---
    if csv_path is not None and final["pos"] is not None:
        with open(csv_path, mode="a") as f:
            csv.writer(f, delimiter=",").writerow([
                eng,
                None if pgm_energy is None else float("{:.2f}".format(pgm_energy)),
                None if pgm_focus is None else float("{:.2f}".format(pgm_focus)),
                final["au"],
                float("{:.5f}".format(final["pos"])),
            ])

    return final["pos"], final["au"]


# ---------------------------------------------------------------------------
# Centroid variant: ``tune_centroid``-based alternative.
# ---------------------------------------------------------------------------


def m3_adjust_centroid(
    *,
    mirror,
    signal,
    diag,
    diag_in=-6,
    diag_out=2,
    tune_range=5e-4,
    min_step=5e-5,
    num_points=10,
    step_factor=3.0,
    snake=False,
    n_samples=10,
    sample_delay=0.1,
    settle_time=3.0,
    csv_path=None,
    eng=None,
    pgm_energy=None,
    pgm_focus=None,
    signal_field=None,
):
    """Centroid-based alternative to :func:`m3_adjust`.

    Delegates the peak-finding to :func:`bluesky.plans.tune_centroid` --
    a successive-refinement centroid search that scans a bracketed range
    around the current mirror position, recenters on the signal
    centroid, shrinks the range by ``step_factor``, and repeats until
    ``min_step`` is reached. See the ``tune_centroid`` documentation for
    convergence semantics.

    This plan reuses the same M3-specific scaffolding as :func:`m3_adjust`:

    * Backlash unwind (mirror approaches from the positive direction)
      before the scan begins.
    * Diagnostic ``diag`` inserted to ``diag_in`` before the scan and
      retracted to ``diag_out`` on completion or exception via
      ``bpp.finalize_wrapper``.
    * One CSV row appended on completion with the same five columns as
      :func:`m3_adjust`: ``eng, pgm_energy, pgm_focus, Au, M3_Ry``.
    * Same return contract: ``(final_mirror_position, final_signal_avg)``.

    Parameters
    ----------
    mirror : ophyd motor-like (e.g. ``M3.Ry``)
    signal : ophyd Readable (e.g. ``qem08.current1.mean_value``)
    diag   : ophyd motor-like (e.g. ``M4AUdiag.trans``)
    diag_in, diag_out : float
        Diagnostic insert/retract positions.
    tune_range : float
        Half-range of the initial centroid scan. ``tune_centroid`` is
        called with ``start = m3_0 - tune_range`` and
        ``stop = m3_0 + tune_range``, where ``m3_0`` is the mirror
        position after the backlash unwind.
    min_step : float
        Smallest step size for the centroid refinement.
    num_points : int
        Points per traversal in each centroid pass.
    step_factor : float
        Range-shrink factor between successive centroid passes
        (``step_factor > 1.0``).
    snake : bool
        Passed through to ``tune_centroid``. Default ``False`` matches
        the M3 backlash preference of always approaching from the same
        direction.
    n_samples, sample_delay : int, float
        After the centroid search converges, the plan takes one final
        ``n_samples``-read measurement to populate the CSV row's ``Au``
        column.
    settle_time : float
        Sleep after the backlash unwind before sampling.
    csv_path : str or None
        If set, append one row at completion (success or exception-free
        early exit).
    eng, pgm_energy, pgm_focus : float or None
        Pass-through values for the CSV row.
    signal_field : str or None
        Name of the data field on ``signal`` to maximize. Defaults to
        ``signal.name`` which is correct for simple ophyd ``Signal``
        objects but may need to be specified explicitly for compound
        devices (e.g. ``"qem08_current1_mean_value"``).

    Returns
    -------
    (float, float)
        ``(final_mirror_position, final_signal_average)``.
    """

    if signal_field is None:
        signal_field = signal.name

    final = {"pos": None, "au": None}

    def _retract_diag():
        yield from bps.mv(diag, diag_out)

    def _body():
        # --- backlash unwind ---
        m3 = yield from bps.rd(mirror)
        yield from bps.mv(mirror, m3 - 2 * min_step)
        yield from bps.mv(mirror, m3 + 2 * min_step)
        yield from bps.sleep(settle_time)
        m3_0 = yield from bps.rd(mirror)

        # --- insert diag ---
        yield from bps.mv(diag, diag_in)

        # --- centroid search ---
        # tune_centroid requires start < stop.
        start = m3_0 - tune_range
        stop = m3_0 + tune_range
        yield from bp.tune_centroid(
            [signal],
            signal_field,
            mirror,
            start,
            stop,
            min_step,
            num=num_points,
            step_factor=step_factor,
            snake=snake,
        )

        # --- final sample for the CSV row ---
        au_avg, _au_std = yield from _sample(signal, n_samples, sample_delay)
        final["pos"] = yield from bps.rd(mirror)
        final["au"] = au_avg

    yield from bpp.finalize_wrapper(_body(), _retract_diag())

    # --- CSV log: append one row on completion ---
    if csv_path is not None and final["pos"] is not None:
        with open(csv_path, mode="a") as f:
            csv.writer(f, delimiter=",").writerow([
                eng,
                None if pgm_energy is None else float("{:.2f}".format(pgm_energy)),
                None if pgm_focus is None else float("{:.2f}".format(pgm_focus)),
                final["au"],
                float("{:.5f}".format(final["pos"])),
            ])

    return final["pos"], final["au"]

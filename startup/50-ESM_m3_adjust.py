"""
Bluesky plan implementation of the M3 motor pitch hill-climb.

This is the refactor of ``m3_adjust_core.m3_hill_climb`` into a proper
Bluesky plan: it ``yield from`` plan stubs instead of calling ``RE(...)``
internally, uses ``bps.rd``/``bps.mv``/``bps.sleep`` for all device I/O,
deduplicates the six identical 10-sample blocks into a ``_sample()``
helper plan, and wraps the body in ``bpp.finalize_wrapper`` so the
diagnostic axis is retracted on success, on the give-up path, **and**
on exception or RunEngine interruption.

Behavioral notes vs. the original ``m3_adjust.py:113-226``:

* **Give-up restore is fixed.** The original script issues
  ``RE(mv(motor, M3_Ry_0))`` then falls through to code that
  immediately moves the motor back to the last attempted position.
  Here the give-up path issues a single ``bps.mv`` to ``m3_0`` and
  returns from the body, so the motor ends up at ``m3_0`` as
  intended.
* **Asymmetric extra step preserved.** When the direction-search
  determines ``direction == -1`` the post-search step uses
  ``2 * direction * step`` rather than ``direction * step`` (motors
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


def _step_and_sample(motor, target, signal, settle, n, delay):
    """Move ``motor`` to ``target``, settle, then sample.

    Returns ``(mean, std)``.
    """
    yield from bps.mv(motor, target)
    yield from bps.sleep(settle)
    avg, std = yield from _sample(signal, n, delay)
    return avg, std


def m3_adjust(
    *,
    motor=M3.Ry,
    signal=qem08.current1.mean_value,
    diag=M4AUdiag.trans,
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
    """Hill-climb ``motor`` to maximize ``signal``.

    Parameters
    ----------
    motor : ophyd motor-like (e.g. ``M3.Ry``)
    signal : ophyd Signal-like (e.g. ``qem08.current1.mean_value``)
    diag   : ophyd motor-like (e.g. ``M4AUdiag.trans``)
    diag_in, diag_out : float
        Diagnostic insert/retract positions.
    step : float
        motor step size.
    n_samples, sample_delay : int, float
        Per-measurement read count and inter-read delay.
    settle_time : float
        Sleep after each motor move before sampling.
    max_insignificant : int
        Number of insignificant direction-search steps before giving up.
    csv_path : str or None
        If set, append one row at completion (success or give-up).
    eng, pgm_energy, pgm_focus : float or None
        Pass-through values for the CSV row.

    Returns
    -------
    (float, float)
        ``(final_motor_position, final_signal_average)``. The signal
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
        m3 = yield from bps.rd(motor)
        print("M3_Ry start (pre-backlash) = {}".format(m3))
        yield from bps.mv(motor, m3 - 2 * step)
        yield from bps.mv(motor, m3 + 2 * step)
        yield from bps.sleep(settle_time)
        m3_0 = yield from bps.rd(motor)
        print("M3_Ry_0 (after backlash unwind) = {}".format(m3_0))
        m3 = m3_0

        # --- insert diag, baseline sample ---
        yield from bps.mv(diag, diag_in)
        au0_avg, au0_std = yield from _sample(signal, n_samples, sample_delay)

        # --- first +step probe ---
        m3 = m3 + step
        au1_avg, au1_std = yield from _step_and_sample(
            motor, m3, signal, settle_time, n_samples, sample_delay
        )
        print("M3_Ry after first +step probe = {}".format((yield from bps.rd(motor))))

        # --- direction search loop ---
        direction = 0.0
        dir_found = False
        insignificant = 0
        gave_up = False

        while not dir_found:
            threshold = (au0_std + au1_std) / 2
            print(
                "direction-search: M3_Ry={M3_Ry}  Au0_avg={Au0_avg} +/- {Au0_std}  "
                "Au1_avg={Au1_avg} +/- {Au1_std}  diff={diff}  threshold={threshold}".format(
                    M3_Ry=(yield from bps.rd(motor)),
                    Au0_avg=au0_avg,
                    Au0_std=au0_std,
                    Au1_avg=au1_avg,
                    Au1_std=au1_std,
                    diff=(au1_avg - au0_avg),
                    threshold=threshold,
                )
            )
            if abs(au1_avg - au0_avg) > threshold:
                direction = +1.0 if (au1_avg - au0_avg) > 0 else -1.0
                dir_found = True
            else:
                m3 = m3 + step
                au0_avg, au0_std = au1_avg, au1_std
                # NOTE: original sleeps BEFORE the move inside this branch
                # (line 160-161). Preserved literally.
                yield from bps.sleep(settle_time)
                yield from bps.mv(motor, m3)
                print(
                    "M3_Ry after insignificant-step move = {}".format(
                        (yield from bps.rd(motor))
                    )
                )
                au1_avg, au1_std = yield from _sample(signal, n_samples, sample_delay)
                insignificant += 1
                print(
                    "one more insignificant step during direction search, tot: ",
                    insignificant,
                )
                if insignificant == max_insignificant:
                    # FIX vs. original: genuine restore + early return.
                    yield from bps.mv(motor, m3_0)
                    print("could not adjust M3")
                    gave_up = True
                    dir_found = True

        if gave_up:
            final["pos"] = yield from bps.rd(motor)
            final["au"] = au1_avg
            return  # diag retract happens in finalize_wrapper

        print("determined direction: ", direction)

        # --- seed sample after direction is known ---
        # motors lines 182-185 of the original: re-uses Au1_values as the
        # new Au0 baseline at the current position.
        au0_avg, au0_std = yield from _sample(signal, n_samples, sample_delay)

        # --- asymmetric extra step (lines 188-192) ---
        if direction == 1.0:
            m3 = m3 + direction * step
        else:
            m3 = m3 + 2 * direction * step
        au1_avg, au1_std = yield from _step_and_sample(
            motor, m3, signal, settle_time, n_samples, sample_delay
        )
        print(
            "M3_Ry after extra step (direction known) = {}".format(
                (yield from bps.rd(motor))
            )
        )
        print("extra step in the direction of increased signal")
        print(
            "climb-loop seed: M3_Ry={M3_Ry}  Au0_avg={Au0_avg} +/- {Au0_std}  "
            "Au1_avg={Au1_avg} +/- {Au1_std}  diff={diff}  threshold={threshold}".format(
                M3_Ry=(yield from bps.rd(motor)),
                Au0_avg=au0_avg,
                Au0_std=au0_std,
                Au1_avg=au1_avg,
                Au1_std=au1_std,
                diff=(au1_avg - au0_avg),
                threshold=(au0_std + au1_std) / 2,
            )
        )

        # --- climb loop ---
        max_found = False
        while not max_found:
            threshold = (au0_std + au1_std) / 2
            if abs(au1_avg - au0_avg) > threshold:
                print(
                    "climb-loop: M3_Ry={M3_Ry}  Au0_avg={Au0_avg} +/- {Au0_std}  "
                    "Au1_avg={Au1_avg} +/- {Au1_std}  diff={diff}  threshold={threshold}".format(
                        M3_Ry=(yield from bps.rd(motor)),
                        Au0_avg=au0_avg,
                        Au0_std=au0_std,
                        Au1_avg=au1_avg,
                        Au1_std=au1_std,
                        diff=(au1_avg - au0_avg),
                        threshold=threshold,
                    )
                )
                if (au1_avg - au0_avg) > 0:
                    print("significant, positive step, continue")
                    m3 = m3 + direction * step
                    au0_avg, au0_std = au1_avg, au1_std
                    au1_avg, au1_std = yield from _step_and_sample(
                        motor, m3, signal, settle_time, n_samples, sample_delay
                    )
                else:
                    print("significant, negative step, reached max: step back")
                    max_found = True
                    m3 = m3 - direction * step
                    yield from bps.mv(motor, m3)
            else:
                print("insignificant, do nothing, go out")
                max_found = True

        final["pos"] = yield from bps.rd(motor)
        final["au"] = au1_avg

    yield from bpp.finalize_wrapper(_body(), _retract_diag())

    # --- CSV log: append one row on completion (success or give-up) ---
    if csv_path is not None and final["pos"] is not None:
        with open(csv_path, mode="a") as f:
            csv.writer(f, delimiter=",").writerow(
                [
                    eng,
                    None if pgm_energy is None else float("{:.2f}".format(pgm_energy)),
                    None if pgm_focus is None else float("{:.2f}".format(pgm_focus)),
                    final["au"],
                    float("{:.5f}".format(final["pos"])),
                ]
            )

    return final["pos"], final["au"]


# ---------------------------------------------------------------------------
# Centroid variant: ``tune_centroid``-based alternative.
# ---------------------------------------------------------------------------


def m3_adjust_centroid(
    *,
    motor=M3.Ry,
    signal=qem08.current1.mean_value,
    diag=M4AUdiag.trans,
    diag_in=-6,
    diag_out=2,
    tune_range=5e-4,
    min_step=5e-5,
    num_points=10,
    step_factor=3.0,
    n_samples=10,
    sample_delay=0.1,
    csv_path=None,
    eng=None,
    pgm_energy=None,
    pgm_focus=None,
):
    """Centroid-based alternative to :func:`m3_adjust`.

    Inserts the diagnostic, delegates peak-finding to
    :func:`bluesky.plans.tune_centroid` with ``snake=False`` so the
    centroid scan always sweeps left-to-right, then takes a final
    ``n_samples`` measurement at the centroid for the CSV row. Diag
    retract is wrapped in ``bpp.finalize_wrapper`` so it runs on
    success, give-up, or exception.

    Parameters
    ----------
    motor : ophyd motor-like (e.g. ``M3.Ry``)
    signal : ophyd Readable (e.g. ``qem08.current1.mean_value``)
    diag   : ophyd motor-like (e.g. ``M4AUdiag.trans``)
    diag_in, diag_out : float
        Diagnostic insert/retract positions.
    tune_range : float
        Half-range of the initial centroid scan. ``tune_centroid`` is
        called with ``start = m3_initial - tune_range`` and
        ``stop = m3_initial + tune_range``, where ``m3_initial`` is
        the motor position when the plan is invoked.
    min_step : float
        Smallest step size for the centroid refinement.
    num_points : int
        Points per traversal in each centroid pass.
    step_factor : float
        Range-shrink factor between successive centroid passes
        (``step_factor > 1.0``).
    n_samples, sample_delay : int, float
        After the centroid search converges, the plan takes one final
        ``n_samples``-read measurement to populate the CSV row's ``Au``
        column.
    csv_path : str or None
        If set, append one row at completion (success or exception-free
        early exit).
    eng, pgm_energy, pgm_focus : float or None
        Pass-through values for the CSV row.

    Returns
    -------
    (float, float)
        ``(final_motor_position, final_signal_average)``.
    """

    signal_field = signal.name

    final = {"pos": None, "au": None}

    def _retract_diag():
        yield from bps.mv(diag, diag_out)

    def _body():
        # --- read initial motor position (no backlash unwind; see
        # docstring for why) ---
        m3_initial = yield from bps.rd(motor)
        print("M3_Ry initial = {}".format(m3_initial))

        # --- insert diag ---
        yield from bps.mv(diag, diag_in)

        # --- centroid search ---
        # tune_centroid requires start < stop.
        # snake=False is REQUIRED: it makes the final mv direction
        # deterministic (always a negative-direction approach from the
        # end of the ascending sweep), so the optical system arrives at
        # the centroid in a reproducible mechanical state. snake=True
        # would make the final approach direction depend on the parity
        # of refinement passes, breaking reproducibility.
        start = m3_initial - tune_range
        stop = m3_initial + tune_range
        print(
            "tune_centroid: bracket=[{}, {}]  min_step={}  num={}".format(
                start, stop, min_step, num_points
            )
        )
        yield from bp.tune_centroid(
            [signal],
            signal_field,
            motor,
            start,
            stop,
            min_step,
            num=num_points,
            step_factor=step_factor,
            snake=False,
        )

        # --- final sample for the CSV row ---
        au_avg, _au_std = yield from _sample(signal, n_samples, sample_delay)
        final["pos"] = yield from bps.rd(motor)
        final["au"] = au_avg
        print("final: M3_Ry={}  Au_avg={}".format(final["pos"], final["au"]))

    yield from bpp.finalize_wrapper(_body(), _retract_diag())

    # --- CSV log: append one row on completion ---
    if csv_path is not None and final["pos"] is not None:
        with open(csv_path, mode="a") as f:
            csv.writer(f, delimiter=",").writerow(
                [
                    eng,
                    None if pgm_energy is None else float("{:.2f}".format(pgm_energy)),
                    None if pgm_focus is None else float("{:.2f}".format(pgm_focus)),
                    final["au"],
                    float("{:.5f}".format(final["pos"])),
                ]
            )

    return final["pos"], final["au"]

"""
Extracted core of the M3 mirror pitch hill-climb adjustment.

This module is a near-verbatim extraction of lines 113-226 of
``m3_adjust.py`` (the Qt run-button script used during XAS measurements
at the ESM ARPES endstation). The extraction is mechanical:

* Hard-coded ophyd device references (``M3.Ry``, ``M4AUdiag.trans``,
  ``qem08.current1.mean_value``) are replaced by parameters.
* ``eval('M3.Ry.user_readback.get()')`` becomes ``mirror.position``.
* ``sleep(3)`` and ``time.sleep(0.1)`` become ``time.sleep(settle_time)``
  and ``time.sleep(sample_delay)``.
* CSV logging is preserved but guarded by ``csv_path is not None`` so
  tests can opt in via ``tmp_path``.
* The widget guard (``widgetValue(ui().ckb_M3_adj)``) is removed; the
  caller decides whether to invoke this function.

The control flow, asymmetric "extra step" at lines 188-192, give-up
behaviour, and ordering of operations are preserved literally so that
characterization tests against this module describe the existing
behaviour of the production script.
"""

import csv
import time

import numpy as np
from bluesky.plan_stubs import mv


def m3_hill_climb(
    *,
    RE,
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
    RE : bluesky.RunEngine
        The run engine used to execute ``mv`` plans.
    mirror : ophyd motor-like
        Mirror axis to optimize (e.g. ``M3.Ry`` or ``ophyd.sim.SynAxis``).
        Must expose ``.position``.
    signal : ophyd Signal-like
        Detector signal to maximize. Must expose ``.get()``.
    diag : ophyd motor-like
        Diagnostic insertion axis (e.g. ``M4AUdiag.trans``).
    diag_in, diag_out : float
        Positions to insert / retract ``diag``.
    step : float
        Mirror step size.
    n_samples : int
        Number of ``signal.get()`` reads per measurement.
    sample_delay : float
        Sleep between consecutive sample reads.
    settle_time : float
        Sleep after each mirror move before sampling.
    max_insignificant : int
        Number of insignificant direction-search steps before giving up.
    csv_path : str or None
        If set, append one row at the end of a successful adjustment.
    eng, pgm_energy, pgm_focus : float or None
        Values passed straight through to the CSV row, mirroring the
        original log format ``[eng, pgm_energy, pgm_focus, Au, M3_Ry]``.

    Returns
    -------
    float
        The final ``mirror.position`` after the adjustment.
    """

    M3_Ry = mirror.position
    RE(mv(mirror, M3_Ry - 2 * step))
    RE(mv(mirror, M3_Ry + 2 * step))  # always arrive from the positive direction (backlash)
    time.sleep(settle_time)
    M3_Ry_0 = mirror.position
    print(mirror.position)
    M3_Ry = M3_Ry_0
    RE(mv(diag, diag_in))
    Au0_values = np.array([0.0] * n_samples)
    Au0_avg, Au0_sigma = 0.0, 0.0
    Au1_values = np.array([0.0] * n_samples)
    Au1_avg, Au1_sigma = 0.0, 0.0

    for i in range(n_samples):
        time.sleep(sample_delay)
        Au0_values[i] = signal.get()
    Au0_avg, Au0_std = np.mean(Au0_values), np.std(Au0_values)

    M3_Ry = M3_Ry + step
    RE(mv(mirror, M3_Ry))
    time.sleep(settle_time)
    print(mirror.position)
    for i in range(n_samples):
        time.sleep(sample_delay)
        Au1_values[i] = signal.get()
    Au1_avg, Au1_std = np.mean(Au1_values), np.std(Au1_values)

    dir_found = False
    max_found = False
    insignificant_step = 0
    direction = 0.0

    while not dir_found:  # loop to determine direction
        print('av0 = {}, av1 = {}, diff = {}'.format(Au0_avg, Au1_avg, (Au1_avg - Au0_avg)))
        if abs((Au1_avg - Au0_avg)) > (Au0_std + Au1_std) / 2:  # significant change
            if (Au1_avg - Au0_avg) > 0:
                direction = +1.0
            else:
                direction = -1.0
            dir_found = True

        else:  # not significant step
            M3_Ry = M3_Ry + step
            time.sleep(settle_time)
            RE(mv(mirror, M3_Ry))
            print(mirror.position)
            Au0_avg, Au0_std = Au1_avg, Au1_std
            for i in range(n_samples):
                time.sleep(sample_delay)
                Au1_values[i] = signal.get()
            Au1_avg, Au1_std = np.mean(Au1_values), np.std(Au1_values)
            insignificant_step += 1
            print('one more insignificant step during direction search, tot: ', insignificant_step)

        if insignificant_step == max_insignificant:
            RE(mv(mirror, M3_Ry_0))
            print('could not adjust M3')
            dir_found = True
            max_found = True
            # return

    print('determined direction: ', direction)
    insignificant_step = 0

    for i in range(n_samples):
        time.sleep(sample_delay)
        Au1_values[i] = signal.get()
    Au0_avg, Au0_std = np.mean(Au1_values), np.std(Au1_values)

    # now that dir of increase is known, move one time in that direction
    if direction == 1.0:
        M3_Ry = M3_Ry + direction * step
    else:
        M3_Ry = M3_Ry + 2 * direction * step
    RE(mv(mirror, M3_Ry))
    time.sleep(settle_time)
    print(mirror.position)

    # Au0_avg, Au0_std = Au1_avg, Au1_std
    for i in range(n_samples):
        time.sleep(sample_delay)
        Au1_values[i] = signal.get()
    Au1_avg, Au1_std = np.mean(Au1_values), np.std(Au1_values)
    print("extra step in the direction of increased signal")
    print('av1 = {}, av0 = {}, diff = {}'.format(Au1_avg, Au0_avg, (Au1_avg - Au0_avg)))
    while not max_found:
        if abs((Au1_avg - Au0_avg)) > (Au0_std + Au1_std) / 2:  # significant change
            print('av1 = {}, av0 = {}, diff = {}'.format(Au1_avg, Au0_avg, (Au1_avg - Au0_avg)))
            if (Au1_avg - Au0_avg) > 0:  # same direction
                print('significant, positive step, continue')
                M3_Ry = M3_Ry + direction * step
                RE(mv(mirror, M3_Ry))
                time.sleep(settle_time)
                Au0_avg, Au0_std = Au1_avg, Au1_std
                for i in range(n_samples):
                    time.sleep(sample_delay)
                    Au1_values[i] = signal.get()
                Au1_avg, Au1_std = np.mean(Au1_values), np.std(Au1_values)
            else:
                print('significant, negative step, reached max: step back')
                max_found = True
                M3_Ry = M3_Ry - direction * step
                RE(mv(mirror, M3_Ry))
        else:
            print('insignificant, do nothing, go out')
            max_found = True

    # after while-loop, pull out the diagnostic
    RE(mv(diag, diag_out))

    if csv_path is not None:
        with open(csv_path, mode='a') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow([
                eng,
                None if pgm_energy is None else eval('{:.2f}'.format(pgm_energy)),
                None if pgm_focus is None else eval('{:.2f}'.format(pgm_focus)),
                Au1_avg,
                eval('{:.5f}'.format(mirror.position)),
            ])

    return mirror.position

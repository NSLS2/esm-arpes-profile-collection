"""Step 4b (MOVES an EPU, optional). Run inside bsui:

    %run -i gateway_tests/4b_epu_move.py
    st = timed_set(EPU57.gap, GAP + 0.1)   # GAP = gap printed below
    st                                     # after RBV stops changing: done=True?
    st = timed_set(EPU57.gap, GAP)         # move back
    st = timed_set(EPU57.gap, EPU57.gap.setpoint.get())   # zero-length move

Do not press Ctrl-C or close bsui while timed_set is running.
"""
import time

print("EPU57 gap:", EPU57.gap.readback.get(), " EPU105 gap:", EPU105.gap.readback.get())


def timed_set(gap, value):
    if not gap.safe_to_actuate():
        raise RuntimeError("The other EPU gap is not above 219; not moving.")
    t0 = time.monotonic()
    st = gap.set(value)
    print(f"set() returned after {time.monotonic() - t0:.1f} s")
    return st

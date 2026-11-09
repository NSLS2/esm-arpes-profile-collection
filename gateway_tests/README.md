# Gateway put-callback tests (temporary)

## Issue
ophyd's `PVPositioner` writes its setpoint with a CA put callback (`wait=True`). Through the accelerator CA gateway this:
- crashes the gateway if the writing client disconnects mid-put while another client has the PV open;
- holds every client's reads of that PV until the put finishes, which shows up as `ReadTimeoutError` in bsui/bsstudio.

This hits `FEslit` gap/center (`FE:C21A…`) and `EPU57`/`EPU105` gap/phase (`SR:C21-ID…`).

## Changes on this branch (`startup/10-machine.py`, `startup/11-undulator.py`)
- `PVPositionerNoPutCallback`: setpoint and actuate are plain puts. The move finishes when `done` reaches `done_value`.
- FE slit: done = `DMOV` (unchanged).
- EPU: done changed from `Cmd:Start-Cmd == 0` to `Sts:Moving-Sts == 0`, as SIX uses for the same type of EPU PVs. `Start-Cmd` is a `bo` with `HIGH=0` that has read "Start" for days with the EPU idle, so it doesn't look like a done flag. Still kept as the actuate.
- EPU stop is a plain put. It previously raised `AttributeError`.

## Run order: what each step tells us
| Script | Moves? | Tells us |
|---|---|---|
| `1_baseline.sh` | no | slit ORIG value and normal read time |
| `2_watch_slit.sh` + `2_stall_put_callback.sh NEW` | slit −0.1 | reads stall during a put callback (the bug reproduced) |
| `3_plain_put.sh ORIG` | slit back | with a plain put, reads don't stall; `DMOV` 0→1 brackets the motion (slit done is valid) |
| `4a_epu_readonly.sh 57\|105` | no | `Start-Cmd` 0/1 and labels, `SPMG`, `Moving-Sts` 0 = No, `DMOV` not `DISABLE` |
| `4b_watch_epu.sh` + `4b_epu_move.py` (in bsui) | EPU +0.1 and back | see below |

`4b` with the **deployed** profile:
- `set()` takes about the move time: writing the setpoint alone moves the EPU, so `actuate` can go.
- `set()` returns at once: `Start-Cmd` starts the move, so keep `actuate`.
- `Moving-Sts` 1 during motion and 0 once `RBV` settles: the done change on this branch is right.
- `Start-Cmd` never returns to 0, or does so before `RBV` settles: the old done signal was wrong.
- Zero-length move: if `Moving-Sts` doesn't toggle, a move to the current gap will never finish on this branch.

`4b` with **this branch**: `set()` returns at once, and `st` should become done when `Moving-Sts` returns to 0.

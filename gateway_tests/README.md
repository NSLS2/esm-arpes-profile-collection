# Gateway put-callback tests (temporary)

## Issue
ophyd's `PVPositioner` writes its setpoint with a CA put callback (`wait=True`). Through the accelerator CA gateway this:
- crashes the gateway if the writing client disconnects mid-put while another client has the PV open;
- holds every client's reads of that PV until the put finishes, which shows up as `ReadTimeoutError` in bsui/bsstudio.

This hits `FEslit` gap/center (`FE:C21A…`) and `EPU57`/`EPU105` gap/phase (`SR:C21-ID…`).

## Changes on this branch (`startup/10-machine.py`, `startup/11-undulator.py`)
- `PVPositionerNoPutCallback`: setpoint and actuate are plain puts. The move finishes when `done` reaches `done_value`.
- FE slit: done = `DMOV` (unchanged).
- EPU: actuate and done both use `Cmd:Start-Cmd`; done is numeric `0`, as on `main`. The comparison trace below rules out `Sts:Moving-Sts` and motor `DMOV` as whole-request completion signals.
- EPU stop is a plain put. It previously raised `AttributeError`.

## Run order: what each step tells us
| Script | Moves? | Tells us |
|---|---|---|
| `1_baseline.sh` | no | slit ORIG value and normal read time |
| `2_watch_slit.sh` + `2_stall_put_callback.sh NEW` | slit −0.1 | reads stall during a put callback (the bug reproduced) |
| `3_plain_put.sh ORIG` | slit back | with a plain put, reads don't stall; `DMOV` 0→1 brackets the motion (slit done is valid) |
| `4a_epu_readonly.sh 57\|105` | no | `Start-Cmd` 0/1 and labels, `SPMG`, `Moving-Sts` 0 = No, `DMOV` not `DISABLE` |
| `4b_watch_epu.sh` + `4b_epu_move.py` (in bsui) | EPU +0.1 and back | see below |

## EPU105 result

The comparison trace shows:
- `Sts:Moving-Sts` stayed at `0` through both real and zero-length requests. A `PVPositioner` using it never observes the required not-done → done transition, so its `MoveStatus` hangs.
- `Cmd:Start-Cmd` transitioned `0` → `1` → `0` for every request with both versions. It returned to `0` only after the final `DMOV == 1`, including zero-length requests.
- `DMOV` can complete one cycle before `Start-Cmd` is asserted and another after it is asserted, so it does not bracket the full EPU request.

Therefore EPU completion remains `Cmd:Start-Cmd == 0`; this branch only removes CA put callbacks from setpoint, actuate, and stop writes.

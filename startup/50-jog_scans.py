import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import uuid
from bluesky.utils import short_uid

def setup_metadata(exposure):
    """ Handles the desired exposure and sets up metadata for the scan

    Args:
        exposure (float): The desired exposure time for the scan.

    Returns:
        dict: A dictionary containing the scan metadata.
    """
    acq_time = 0
    computed_exposure = exposure
    num_frame = 0

    sp ={
        "time_per_frame": acq_time,
        "requested_exposure": exposure,
        "computed_exposure": computed_exposure,
        "num_frames": num_frame,
        "type": "ct",
        "uid": str(uuid.uuid4()),
        "plan_name": "ct",
    }

    return sp

def rocking_ct(dets, exposure, slow_motor, sstart, sstop, fast_motor, fstart, fstop, *, num=1, md=None, slow_steps=15):
    """Take a count while "rocking" the y-position"""
    _md = md or {}

    # TODO push the exposure out to the dets, ask Elio if this is needed for quadem
    sp_md = yield from setup_metadata(exposure)
    _md.update(sp_md)
    _md["plan_name"] = "jog"

    _md["jog_md"] = {"start": sstart, "stop": sstop, "motor": slow_motor.name}

    @bpp.reset_positions_decorator([fast_motor.velocity])
    def per_shot(dets):
        nonlocal fstart, fstop
        yield from bps.mv(fast_motor, fstart, slow_motor, sstop)  # got to initial position
        yield from bps.mv(fast_motor.velocity, abs(fstop - fstart) / exposure, timeout=1)  # set velocity
        gp = short_uid("rocker")
        gp2 = short_uid("dets")
        sts = yield from bps.trigger(dets, group=gp2)  # trigger the detectors
        while not sts.done:
            fstart, fstop = fstop, fstart
            gp3 = short_uid("fast_rocker")
            yield from bps.abs_set(fast_motor, sstop, group=gp3)  # set motor to move towards end
            yield from bps.wait(gp3)
        yield from bps.create()
        yield from bps.read(dets)
        yield from bps.save()
        yield from bps.wait(group=gp)

    return (yield from bps.count(dets, md=_md,
                                    per_shot=per_shot if fstart != fstop else bps.trigger_and_read,
                                    num=num))
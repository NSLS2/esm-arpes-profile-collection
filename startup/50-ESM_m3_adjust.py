"""IPython startup loader for the m3_adjust Bluesky plan.

The numbered filename ensures the plan is available in the interactive
namespace after device modules (10-machine, 20-motors, 30-detectors)
have loaded. The actual implementation lives in ``m3_adjust_plan.py``
so it is importable under a normal Python identifier from tests.
"""

from m3_adjust_plan import m3_adjust, m3_adjust_centroid  # noqa: F401

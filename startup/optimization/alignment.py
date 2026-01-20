import time as ttime
import numpy as np
from tiled.client.container import Container


class ElectrometerEvaluation:
    def __init__(self, tiled_client: Container) -> None:
        self._tiled_client = tiled_client

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        """
        Read the electrometer current from Tiled.
        """

        ttime.sleep(3.0)

        outcomes = []
        run = self._tiled_client[uid]
        electrometer_current = run["primary/xqem01_current1_mean_value"].read()
        suggestion_ids = [suggestion["_id"] for suggestion in run.metadata["start"]["blop_suggestions"]]
        for idx, sid in enumerate(suggestion_ids):
            outcome = {
                "_id": sid,
                "xqem01_current": electrometer_current[idx],
            }
            outcomes.append(outcome)

        return outcomes

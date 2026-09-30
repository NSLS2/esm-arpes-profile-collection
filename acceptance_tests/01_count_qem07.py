from bluesky.plans import count
from bluesky.callbacks import LiveTable

RE(count([qem07], num=5), LiveTable(qem07.hints["fields"]))

import os

import nslsii
from ophyd.signal import EpicsSignalBase


# Set timeout for all EpicsSignalBase objects
EpicsSignalBase.set_defaults(connection_timeout=10)

# Configure RE + other services
nslsii.configure_base(get_ipython().user_ns,
                      'arpes',
                      publish_documents_with_kafka=True,
                      redis_prefix="arpes-",
                      redis_url="info.esm.nsls2.bnl.gov")

# Set ipython startup dir variable (used in some modules):
PROFILE_STARTUP_PATH = os.path.abspath(get_ipython().profile_dir.startup_dir)

def proposal_path_template():
    """Return a template string for the proposal path with {cycle} and {data_session} placeholders."""
    return "/nsls2/data/esm/proposals/{cycle}/{data_session}"

def proposal_path(cycle, data_session):
    return proposal_path_template().format(cycle=cycle, data_session=data_session)
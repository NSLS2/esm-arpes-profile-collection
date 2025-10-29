import os
import time

import nslsii
from ophyd.signal import EpicsSignalBase

from IPython import get_ipython
from IPython.terminal.prompts import Prompts, Token
from tiled.client import from_profile
from databroker import Broker


# Set timeout for all EpicsSignalBase objects
EpicsSignalBase.set_defaults(connection_timeout=10)

# Configure a Tiled writing client
tiled_writing_client = from_profile("nsls2", api_key=os.environ["TILED_BLUESKY_WRITING_API_KEY_ESM"])["arpes"]["raw"]

class TiledInserter:
    name = 'arpes'

    def insert(self, name, doc):
        tiled_writing_client.post_document(name, doc)

tiled_inserter = TiledInserter()

# Configure RE + other services
nslsii.configure_base(get_ipython().user_ns,
                      tiled_inserter,
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


# Initialize (user-authenticated) Tiled reading client
print("Initializing Tiled reading client...\nMake sure you check for duo push.")
tiled_reading_client = from_profile("nsls2", username=None)["arpes"]["raw"]

db = Broker(tiled_reading_client)  # Keep for backcompatibility with older code that uses databroker

# Set up custom prompt to show proposal ID
class ProposalIDPrompt(Prompts):
    def in_prompt_tokens(self, cli=None):
        return [
            (
                Token.Prompt,
                f"{RE.md.get('data_session', 'N/A')} [",
            ),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, "]: "),
        ]

ip = get_ipython()
ip.prompts = ProposalIDPrompt(ip)

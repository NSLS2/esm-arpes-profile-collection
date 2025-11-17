import numpy
import os

from bluesky_tiled_plugins import TiledWriter
from bluesky.callbacks.buffer import BufferingWrapper
from tiled.client import from_uri

# Define document-specific patches to be applied before sending them to TiledWriter

def patch_descriptor(doc):
    # Ensure dtype_str has the proper numpy format (to pass the EventModel validator)
    for key, val in doc["data_keys"].items():
        if "dtype_str" in val:
            val["dtype_str"] = numpy.dtype(val["dtype_str"]).str
        val["shape"] = tuple(map(lambda x: max(x, 0), val.get("shape", [])))

    return doc

def patch_resource(doc):

    kwargs = doc.get("resource_kwargs", {})

    # Fix the resource path
    root = doc.get("root", "")
    if not doc["resource_path"].startswith(root):
        doc["resource_path"] = os.path.join(root, doc["resource_path"])
    doc["root"] = ""

    if doc.get("spec") in ["AD_HDF5"]:
        kwargs.update({"dataset": 'entry/instrument/detector/data'})
    elif doc.get("spec") in ["AD_TIFF"]:
        kwargs["join_method"] = "stack"
    elif doc.get("spec") in ["A1_HDF5"]:
        kwargs.update({"dataset": 'entry/instrument/analyzer/data'})

    return doc

# Initialize the Tiled client and the TiledWriter
api_key = os.environ.get("TILED_BLUESKY_WRITING_API_KEY_ARPES")
tiled_writing_client_sql = from_uri("https://tiled.nsls2.bnl.gov", api_key=api_key)['arpes']['migration']
tw = TiledWriter(client = tiled_writing_client_sql,
                 backup_directory="/tmp/tiled_backup",
                 patches = {"descriptor": patch_descriptor,
                            "resource": patch_resource},
                 spec_to_mimetype= {
                     "AD_HDF5": "application/x-hdf5",
                     "A1_HDF5": "application/x-hdf5",
                     "AD_TIFF": "multipart/related;type=image/tiff",
                 })

# Thread-safe wrapper for TiledWriter
tw = BufferingWrapper(tw)

# Subscribe the TiledWriter
RE.md["tiled_access_tags"] = (RE.md["data_session"],)
RE.subscribe(tw)

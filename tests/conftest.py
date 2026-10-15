import os

from pinch.pipeline import load_env

load_env()  # picks up HF_HOME etc. from .env so model downloads land where configured
os.environ["PINCH_CANDIDATES"] = "1"  # tests don't load the CLIP expression judge unless they opt in
os.environ["PINCH_MEMORY"] = "off"  # never touch the real character memory; tests opt in with a temp store

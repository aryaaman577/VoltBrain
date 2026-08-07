import runpy
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(BASE_DIR, "app.py"), run_name="__main__")

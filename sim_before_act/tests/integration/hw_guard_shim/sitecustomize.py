"""TEST GUARD: patch cv2.VideoCapture (camera open) in every Python process that inherits this directory on PYTHONPATH."""
import importlib.abc
import importlib.util
import os
import sys


def _guarded_videocapture(*a, **k):
    with open(os.environ["HW_GUARD_LOG"], "a") as f:
        f.write(f"cv2.VideoCapture{a}\n")
    raise RuntimeError("offline path tried to open a camera")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "cv2" or "HW_GUARD_LOG" not in os.environ:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec("cv2")
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig = loader.exec_module

        class _Loader(importlib.abc.Loader):
            def create_module(self, s):
                return loader.create_module(s)

            def exec_module(self, module):
                orig(module)
                m = sys.modules.get("cv2", module)
                m.VideoCapture = _guarded_videocapture

        spec.loader = _Loader()
        return spec


if "HW_GUARD_LOG" in os.environ:
    sys.meta_path.insert(0, _Finder())

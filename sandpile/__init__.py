__title__ = 'sandpile'

from ._version import __version__


# Passthrough no-op profile decorator
import __builtin__
try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile


from .sandpile import *
from .compare_cases import *
from .cost import *
from . import cluster

"""
Enables librosa's on-disk cache for expensive, input-independent computations
(most importantly: building the CQT filter bank).

IMPORTANT: librosa reads the LIBROSA_CACHE_DIR environment variable exactly
once, the moment `librosa` itself is first imported in a process. Setting the
variable *after* that import has already happened has no effect.

That's why this is its own tiny module with zero imports of its own: every
other module in this project that does `import librosa` imports this module
first (as its very first line). That guarantees the env vars below are set
before `librosa` is ever loaded, no matter which script is the entry point or
what order things get imported in.
"""
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(os.path.dirname(_THIS_DIR), ".librosa_cache")

os.environ.setdefault("LIBROSA_CACHE_DIR", _CACHE_DIR)
# Cache level controls which internal librosa functions get cached, as a
# threshold -- everything tagged at this level or below gets cached.
os.environ.setdefault("LIBROSA_CACHE_LEVEL", "10")

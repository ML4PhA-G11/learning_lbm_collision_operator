"""Backend selector for Keras 3.

On CPUs without AVX (e.g. Intel Pentium 4415U), the PyPI TensorFlow wheel
crashes with SIGILL at import. This module probes the CPU before any Keras
import and falls back to ``KERAS_BACKEND=torch`` in that case. On AVX-capable
hardware the default (tensorflow) is left untouched, so existing
``tensorflow.keras``-style code keeps working with no behaviour change.

Downstream code should import what it needs from here:

    from kerascompat import K, ops, BACKEND

instead of reaching into ``tensorflow.keras`` or ``tf.*`` directly.
"""
import os
import platform


def _cpu_has_avx() -> bool:
    """Return True if the running CPU advertises AVX.

    Reads ``/proc/cpuinfo`` on Linux. On non-Linux platforms returns True
    (assume modern CPU; let TF try and surface its own error if not).
    """
    if platform.system() != "Linux":
        return True
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return "avx" in line.split()
    except OSError:
        return True
    return True


if not _cpu_has_avx():
    os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import backend as K
from keras import ops

BACKEND = keras.backend.backend()

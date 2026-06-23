# Vendored from https://github.com/DBraun/audiotree (v1.0.0), MIT-licensed
# (see ../LICENSE). This is a *minimal* copy: only the ``AudioTree`` pytree
# container and its array helpers needed by magenta_rt inference. The data
# pipeline pieces (``TreeWriter`` / ``sources`` / ``transforms`` / loudness /
# file IO) used only by ``magenta_rt.sft`` are intentionally omitted, which is
# why this copy needs no ``librosa`` / ``loudness`` / ``jaxloudnorm``.
#
# The ``magenta_rt._vendor`` hook puts this package on ``sys.path`` as the
# top-level ``audiotree`` *only when the real package is not installed*, so
# ``from audiotree import AudioTree`` resolves here for a plain inference
# install and to the full PyPI/git package once that is installed (which always
# wins). Keep ``from audiotree import AudioTree`` working in both cases.
__version__ = "1.0.0+vendored.magenta_rt"

from .core import AudioTree

__all__ = ["AudioTree"]

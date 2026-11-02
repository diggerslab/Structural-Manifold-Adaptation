"""Structural Manifold Adaptation: portable S2 data allocation."""
from .data import make_windows
from .sampler import SMASampler
from .structure import StructuralMetric, build_graph

__version__ = '0.1.0'
__all__ = ['make_windows', 'SMASampler', 'StructuralMetric', 'build_graph']

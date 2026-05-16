"""
M3FusionTrack: All-Weather Object Tracking with Multi-modal Multi-frequency
Foundation Model and Adaptive Gated Fusion.

This package contains a simplified reference implementation of the model
described in the paper. It is intended for research and teaching purposes,
not as a production-grade tracker.
"""

from .models.m3fusiontrack import M3FusionTrack, build_m3fusiontrack
from .trackers.tracker import M3Tracker

__version__ = "0.1.0"
__all__ = ["M3FusionTrack", "build_m3fusiontrack", "M3Tracker"]

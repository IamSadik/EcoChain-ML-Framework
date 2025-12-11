"""
EcoChain-ML: A Hybrid Framework for Energy-Efficient Machine Learning 
Model Verification Using Lightweight Blockchain

This package implements an energy-aware ML inference framework with 
blockchain-based verification for sustainable computing.
"""

__version__ = "1.0.0"
__author__ = "EcoChain-ML Research Team"

from . import scheduler
from . import blockchain
from . import inference
from . import monitoring
from . import simulator

__all__ = [
    'scheduler',
    'blockchain',
    'inference',
    'monitoring',
    'simulator'
]

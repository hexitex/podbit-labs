"""
NN-Lab Runner — trains neural networks from declarative experiment specs.

Fixed menus, no arbitrary code. Everything is selected from known architectures,
datasets, optimizers, and measurements.
"""

from runner.experiment import run_experiment

__all__ = ['run_experiment']

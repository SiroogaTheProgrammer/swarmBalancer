"""swarm.arena - the drone test bench: RECON, DEFENCE and OFFENCE tests, and training both swarms.

Vectorised numpy simulations (many episodes at once), hand-written baseline brains, the trainable "mesh"
brain and uploadable policy packages. ``python -m swarm.arena --help``; docs/ARENA.md is the manual.
"""

from .brains import (BUILTINS, DEFAULTS, ROLES, Brain, BrainError, MixedBrain, NeuralBrain, PythonBrain,
                     check_actions, list_brains, load_brain)
from .clash import AttackerObs, ClashEnv, DefenderObs
from .config import ClashConfig, ConfigError, ReconConfig, TrainConfig, from_dict, to_dict
from .core import K, KD, M, MSG
from .recon import ReconEnv, ReconObs
from .run import TESTS, play_clash, play_recon, run_test

__all__ = [
    "BUILTINS", "DEFAULTS", "ROLES", "Brain", "BrainError", "MixedBrain", "NeuralBrain", "PythonBrain",
    "check_actions", "list_brains", "load_brain", "AttackerObs", "ClashEnv", "DefenderObs", "ClashConfig",
    "ConfigError", "ReconConfig", "TrainConfig", "from_dict", "to_dict", "K", "KD", "M", "MSG", "ReconEnv",
    "ReconObs", "TESTS", "play_clash", "play_recon", "run_test",
]

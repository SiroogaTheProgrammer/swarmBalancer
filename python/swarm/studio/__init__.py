"""swarm.studio - the drone swarm test bench: RECON, DEFENCE and OFFENCE tests in 3D, training both swarms.

    python -m swarm.studio          (or: python dev.py studio)

A local web application: a standard-library HTTP server (:mod:`swarm.studio.server`) serves a WebGL user
interface (``static/``) and a JSON API. Tests and training run in worker processes
(:mod:`swarm.studio.worker`) with the same arena code as ``python -m swarm.arena``, so the numbers on
screen are the command line's numbers. docs/STUDIO.md is the guide, docs/ARENA.md the manual.
"""

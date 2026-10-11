"""swarm.studio - the 3D tactical console: plan scenarios, pick or upload brains and tactics, watch the swarm.

    python -m swarm.studio          (or: python dev.py studio)

A local web application: a standard-library HTTP server (:mod:`swarm.studio.server`) serves a WebGL user
interface (``static/``) and a JSON API. Simulations, validations and builds run in worker processes
(:mod:`swarm.studio.worker`) with the same simulator, bench and SDK as the command line, so the numbers
in the console are the battery's numbers. docs/STUDIO.md is the field guide.
"""

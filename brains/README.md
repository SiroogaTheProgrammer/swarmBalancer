# brains/ - your package library

Packages uploaded in the studio's ARMORY, installed with `python -m swarm.sdk install`, or created with
`python -m swarm.sdk new NAME --type perception-python|perception-swm|tactic` land here, one folder each.
The studio and `python -m swarm.sdk list` pick them up automatically.

This folder is git-ignored (except this file): keep the source of your brains and tactics in your own
repository and upload / install it here. How to write them: [docs/SDK.md](../docs/SDK.md).

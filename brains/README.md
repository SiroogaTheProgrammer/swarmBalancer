# brains/ - your package library

Brains you LOAD in the studio or save from its TRAIN tab, packages installed with
`python -m swarm.sdk install`, and skeletons created with
`python -m swarm.sdk new NAME --type policy-defender|policy-attacker|policy-recon|perception-python|perception-swm|tactic`
land here, one folder each. The studio's brain lists and `python -m swarm.sdk list` pick them up
automatically.

This folder is git-ignored (except this file): keep the source of your brains and tactics in your own
repository and load / install it here. How to write them: [docs/ARENA.md](../docs/ARENA.md) (drone brains
for the tests) and [docs/SDK.md](../docs/SDK.md) (packages, perception brains, tactics).

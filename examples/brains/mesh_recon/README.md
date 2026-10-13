# Mesh recon (trained)

A neural **recon** brain for the arena's RECON test. One small network (40 inputs -> 16 ReLU -> 10
outputs, 826 weights) that every drone runs. It weighs five steering urges (nearest unseen ground, back to
the lead, towards friends, away from friends too close, towards windows still missing detail) and decides
the speed, the camera, its flying height and two radio channels.

It started as plain "frontier" search at full-detail height (what an untrained network does) and was
evolved from scratch for 3000 generations on the default mission - 12 drones, a 3 km square, the base in
the middle - 39 minutes on 8 CPU cores. This is the best generation on the training's fixed test
(generation 2782). It learned to hold the full-detail height, to keep the whole swarm in the radio chain
to the lead while it spreads out, and to switch cameras on only over ground it would still improve:

| RECON, 12 episodes, seed 21 | surveyed | 90 % after | reported | in contact | camera on | energy | score |
|---|---|---|---|---|---|---|---|
| **Mesh recon (trained)** | 95.8 % | 510 s | 96.1 % | 98.9 % | 41 % | 30 J | **1.07** |
| Lawnmower | **99.9 %** | **482 s** | **100 %** | **100 %** | 100 % | 67 J | 1.00 |
| Relay sweep (load-aware) | 83.1 % | 600 s (1 run in 12) | 83.6 % | 80.6 % | **27 %** | **20 J** | 0.77 |
| Frontier (high) | 73.0 % | - | 58.3 % | 53.2 % | 100 % | 72 J | 0.30 |
| Flock | 31.8 % | - | 33.3 % | 100 % | 100 % | 72 J | 0.13 |
| *the previous version (120 generations)* | 93.4 % | 458 s | 92.8 % | 75.2 % | 73 % | 53 J | 0.82 |

It beats the best hand-written recon swarm on the overall score: within 4 points of the lawnmower's
coverage and reporting and nearly always in contact, for less than half its compute and radio energy.
These 12 situations are not the ones it was selected on, so the margin is not luck.

```powershell
python -m swarm.arena test recon --recon lib:mesh_recon
# reproduce (about 40 minutes on 8 cores; the brain is recon_best.npz):
python -m swarm.arena train --mode recon --generations 3000 --seed 1 --out out/arena/recon
```

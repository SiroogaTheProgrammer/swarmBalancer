# Mesh wall (trained)

A neural **defender** brain for the arena's DEFENCE test: one small network (35 inputs per option row ->
16 ReLU -> 6 outputs, 678 weights) that every interceptor runs on what it senses and what its friends
radio - their sightings and their claims. It scores each of the attackers it knows about and "hold
station", and takes the best; the "hold" row also says where to wait - its guard post.

It learned two things on its own:

* **Sharing out targets.** Nobody told it how to use the claims its friends radio. It shares out targets
  about as well as the hand-written auction - 1.3 interceptors per attacker it chases, where an
  uncoordinated wall sends almost three.
* **Where to wait.** Given the choice, it moved the whole wall forward: every interceptor waits at the
  forward limit, half way to where the attackers start, stacked deepest at the two flanks and spread over
  three heights from the floor to the ceiling. A swarm creeping in under the terrain mask is then seen
  1.2 km before it could be from the line, and an attacker that slips through is chased down from behind
  on its way to the line.

| 12 episodes, seed 21 | vs the 5 built-in attacks: intercepted | interceptors per target | vs Mesh swarm (trained) | vs the previous swarm (edge attack) |
|---|---|---|---|---|
| **Mesh wall (trained)** | 94.4 % | 1.32 | **88 %** | **98 %** |
| Pickets + auction, hand-written | **95.9 %** | 1.25 | 62 % | 91 % |
| Smart wall (auction), hand-written | 93.5 % | 1.28 | 48 % | 35 % |
| Greedy, no coordination | 71.4 % | 2.75 | 35 % | 33 % |
| Zones | 60.3 % | 1.11 | 30 % | 9 % |
| previous version (could only hold the line) | 92.8 % | 1.29 | 48 % | 35 % |

How it was made: the previous version came from 2000 generations of co-evolution, when walls could only
hold the line - and no wall stopped more than 35 % of that run's swarm. Once guard posts were added, it was
trained on against that swarm for 3000 more generations: 36 minutes on 8 cores with the native ARM64
Python. Within 5 generations it found forward defence, and head to head the share of the swarm
intercepted jumped from 34 % to 92 %. Over the run it averaged 93-96 %; the swarm's best counter got 28 %
through (generation 1402), and in the last 1000 generations never more than 10 %. This is the run's best
wall on the fixed skill test, from generation 647. The final wall (generation 3000) holds the final swarm
better (94 %) but has lost a little against the hand-written attacks (93.5 %).

It relies on the depth it is given. With *Forward limit* (`guard_depth`) at 0.3 it stops 79 % of the
trained swarm and 89 % of the built-in attacks; kept on the line (0.1), 61 % and 81 %.

```powershell
python -m swarm.arena test defence --defender lib:mesh_defender --attacker lib:mesh_attacker
python -m swarm.arena export lib:mesh_defender --role defender --out deploy/mesh_defender.swm   # C++ engine, MCUs too
# train both on (about 12 minutes per 1000 generations on 8 cores):
python -m swarm.arena train --generations 1000 --start-defender lib:mesh_defender --start-attacker lib:mesh_attacker --out out/arena/clash
```

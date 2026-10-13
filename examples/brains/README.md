# Example brain, policy & tactic packages

Each folder is an SDK package (see [docs/SDK.md](../../docs/SDK.md)). The studio's brain lists and the
BRAINS drawer show the arena policies; `python -m swarm.sdk list` shows all of them.

| package | type | try it |
|---|---|---|
| [mesh_defender](mesh_defender) | arena policy, trained network (defender) | an interceptor wall co-evolved with the swarm below. Nobody told it how: it shares out targets like the hand-written smart wall (1.32 interceptors per target vs greedy's 2.75) and, free to choose where to wait, moved the whole wall forward - it stops 88 % of the trained swarm, where the smart wall stops 48 %. `python -m swarm.arena test defence --defender lib:mesh_defender --attacker lib:mesh_attacker` |
| [mesh_attacker](mesh_attacker) | arena policy, trained network (attacker) | the swarm it trained against: it goes to ground out of sensor range and creeps in on one flank under the terrain mask - 37 % through a picket wall, 50 % through the hand-written smart wall, where the best built-in attacks get 5.6 % and 9.5 %. `python -m swarm.arena test offence --attacker lib:mesh_attacker --defender pickets` |
| [mesh_recon](mesh_recon) | arena policy, trained network (recon) | a recon swarm evolved for 3000 generations: it beats the hand-written lawnmower overall - nearly the same coverage, 99 % in radio contact, for under half the sensing energy. `python -m swarm.arena test recon --recon lib:mesh_recon` |
| [pincer_attack](pincer_attack) | arena policy, Python (attacker) | the example to copy: a radio-coordinated two-flank attack under the terrain mask in about 50 lines of numpy that breaks a zone defence (60 % through). `python -m swarm.sdk validate examples/brains/pincer_attack` |
| [edge_detector](edge_detector) | perception brain, Python | classical computer vision, no network: thresholds + blob extents, ~4 operations per pixel, 98 % on the simulator's world. `python -m swarm.sdk export examples/brains/edge_detector --out deploy/edge` makes it a device-runtime factory for Linux companions. |
| [micro_cnn](micro_cnn) | perception brain, `.swm` | a ~20 KB int8 CNN sized for Cortex-M4 drones. `python examples/brains/micro_cnn/train.py` trains and compiles `model.swm`, which the C++ engine runs on every device. |
| [load_aware_stripe](load_aware_stripe) | coordination tactic | the striped design with frames dealt by member speed and heartbeat backpressure; on the mixed s11 swarm: busiest CPU 3 % instead of 100 %, no rejected frames. |

```powershell
python -m swarm.studio                                   # DEFENCE tab: pick "Mesh wall (trained)", RUN TEST
python -m swarm.sdk validate examples/brains/load_aware_stripe
```

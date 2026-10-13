# Mesh swarm (trained)

A neural **attacker** brain for the arena's OFFENCE test: one small network (53 inputs -> 16 ReLU -> 5
outputs: heading, speed, change of height and two radio channels, 949 weights) that every attacker runs on
what it and its friends see of the interceptors and what its friends radio.

It co-evolved with the [mesh wall](../mesh_defender). It started from the previous version: a swarm that
had found a massed low-level attack at the very edge of the front and got 65 % through the hand-written
smart wall. Then walls were allowed to post interceptors forward, and against them that attack collapsed
(1-8 % through). In 3000 more generations the swarm found another way in:

* for the first half minute, still beyond the interceptors' sensors, the whole swarm drops to the floor
  and slides towards one flank;
* then it creeps in as a broad line a kilometre wide, at 5 m under the terrain mask, where it is seen only
  500 m away;
* it reaches the line only in the last quarter-minute of the engagement.

Pickets see it too late and are swamped. The trained wall, waiting at the forward limit and stacked at the
flanks, catches most of it.

| got through (12 episodes, seed 21) | vs Smart wall (auction) | vs Pickets + auction | vs Greedy | vs Zones | vs Mesh wall (trained) |
|---|---|---|---|---|---|
| **Mesh swarm (trained)** | 50 % | **37 %** | 63 % | 68 % | **12 %** |
| previous version (edge attack) | **65 %** | 8 % | **67 %** | **91 %** | 1 % |
| best built-in attack | 9.5 % (Terrain hug) | 5.6 % (Decoys + strike) | 41 % (Jink) | 57 % (Fan out) | 9 % (Decoys + strike) |

This is the run's best swarm on the fixed skill test, from generation 2906. The wall learned faster this
time: head to head it intercepted 93-96 % of the swarm on average over the run; the swarm's best counter
got 28 % through (generation 1402).

Watch it in the studio (OFFENCE tab, against Pickets + auction): the swarm goes to ground out of the
pickets' sight, then pours through on one flank.

```powershell
python -m swarm.arena test offence --attacker lib:mesh_attacker --defender pickets
# train both on (about 12 minutes per 1000 generations on 8 cores):
python -m swarm.arena train --generations 1000 --start-defender lib:mesh_defender --start-attacker lib:mesh_attacker --out out/arena/clash
```

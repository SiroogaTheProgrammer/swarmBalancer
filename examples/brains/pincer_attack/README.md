# Pincer (radio-coordinated)

A hand-written **attacker** brain for the arena's OFFENCE test, and the example to copy when you write
your own: [policy.py](policy.py) is about 50 lines of numpy.

The swarm splits in two by each drone's role code. Both halves drop under the terrain mask (interceptors
see them only up close there, but they fly a quarter slower), fly to their own flank of the corridor and
loiter beyond the interceptors' sensor range, radioing "in position". When most of the friends a drone
hears are ready (or someone calls "go", or time runs short) it calls "go" too - the call floods through the
mesh hop by hop - and both flanks strike the ends of the line at the same moment. A drone about to be hit
breaks to the side and dives or climbs away from the interceptor.

It uses all three things the 3-D arena gives an attacker: height (the `altitude` action), the terrain
mask, and the radio. It breaks a zone defence - one sector per interceptor - better than any built-in
attack. A wall whose interceptors share what they see and share out the targets is another matter: the
whole line learns where the flanks are, and closes ranks as interceptors are spent - and pickets posted
forward see the flanks forming long before they strike.

| got through (12 episodes, seed 3) | vs Zones | vs Greedy | vs Smart wall (auction) | vs Pickets + auction |
|---|---|---|---|---|
| **Pincer** | **60 %** | 13 % | 7 % | 5 % |
| best built-in attack | 57 % (Fan out) | 37 % (Jink) | 9 % (Terrain hug) | 5 % (Decoys + strike) |

```powershell
python -m swarm.sdk validate examples/brains/pincer_attack
python -m swarm.arena test offence --attacker lib:pincer_attack --defender zone
```

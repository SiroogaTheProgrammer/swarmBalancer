# Example brain & tactic packages

Each folder is an SDK package (see [docs/SDK.md](../../docs/SDK.md)); the studio's ARMORY lists them.

| package | type | try it |
|---|---|---|
| [edge_detector](edge_detector) | perception brain, Python | classical computer vision, no network: thresholds + blob extents, ~4 operations per pixel, 98 % on the simulator's world. `python -m swarm.sdk export examples/brains/edge_detector --out deploy/edge` makes it a device-runtime factory for Linux companions. |
| [micro_cnn](micro_cnn) | perception brain, `.swm` | a ~20 KB int8 CNN sized for Cortex-M4 drones. `python examples/brains/micro_cnn/train.py` (or COMPILE in the ARMORY) trains and compiles `model.swm`, which the C++ engine runs on every device. |
| [load_aware_stripe](load_aware_stripe) | coordination tactic | the striped design with frames dealt by member speed and heartbeat backpressure; on the mixed s11 swarm: busiest CPU 3 % instead of 100 %, no rejected frames. |

```powershell
python -m swarm.sdk validate examples/brains/load_aware_stripe
python -m swarm.studio      # MISSION PLANNER: pick s11_heterogeneous, tick load_aware, EXECUTE
```

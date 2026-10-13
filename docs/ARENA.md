# The arena - drone swarm test bench

Three tests and a training mode on one simulator (`python/swarm/arena`), with two front ends: the 3D
studio (`python -m swarm.studio`, see [STUDIO.md](STUDIO.md)) and the command line
(`python -m swarm.arena`). Both run the same code, so they report the same numbers.

| test | the brain under test flies | it wins by | headline numbers |
|---|---|---|---|
| **RECON** | a recon swarm | surveying the whole area in detail fast, reporting what it finds, staying in radio contact, keeping the sensing load low | area surveyed, time to 90 %, targets reported, in radio contact, mean altitude, camera on, CPU, energy |
| **DEFENCE** | the interceptor wall | intercepting the incoming swarm without piling several interceptors onto one attacker | intercepted, got through, interceptors per target, time to intercept |
| **OFFENCE** | the attacking swarm | getting past the wall | got through, shot down, still flying, time under the terrain mask, mean altitude |
| **TRAIN** | both swarms (or one, or a recon swarm) | - | learning curves: who is winning head to head, who learns faster |

Every test plays all its episodes at once (8 by default). It then replays **exactly the same situations**
(same seed, so the same arrivals or hidden targets) with every built-in brain of the role under test.
The comparison table is therefore like for like. Episode 0 is recorded for the 3D replay.

```powershell
python -m swarm.arena test defence --defender auction --attacker decoy
python -m swarm.arena test offence --attacker lib:pincer_attack --defender zone --set attackers=20
python -m swarm.arena test recon --recon relay_sweep --episodes 12 --json out/recon.json
python -m swarm.arena train --generations 40 --out out/arena/run1          # both swarms learn
python -m swarm.arena brains                                                # what can be selected
```

## The world

Everything is 3-D: `x` across the front (east), `y` towards the enemy (north), `z` altitude, in metres;
speeds in m/s, times in seconds. Drones are point masses that fly like multirotors: a top horizontal
speed, a separate and slower climb / descent rate, and an acceleration limit. Every brain decides once per
decision interval (`dt`); in between, the autopilots keep flying its last commands (DEFENCE and OFFENCE
simulate flight in steps of at most 0.5 s). A brain whose `macs` take longer than `dt` on its board
(`*_device`, a `DeviceProfile`) decides less often, and the CPU load figure shows what it costs.

### DEFENCE and OFFENCE (the clash)

* **The airspace** is a box `width` wide, from the ground up to `ceiling`. The protected line is the
  vertical plane `y = 0`: an attacker scores by crossing it at any height. Attackers start around
  `y = depth` at mixed heights: `line` spreads them along the front, `cluster` packs them into one group,
  `waves` sends them in two waves. Interceptors guard evenly spaced stations at `0.1 * depth`, alternating
  between a low layer (15 % of the ceiling, kept above the terrain mask) and a high one (55 %): a
  two-layer wall. When an interceptor is spent, the line re-spaces its stations over the rest, so a spent
  interceptor leaves no hole. By default 24 interceptors face 48 attackers over a 3 km front, the
  attackers start 3 km out, and the ceiling is 600 m.
* **Guard posts.** A defender brain may also choose where each interceptor waits (its `station` action):
  anywhere across the front, at any height, from the line out to `guard_depth` x `depth` (by default half
  way to where the attackers start). Interceptors posted forward as pickets see a low, terrain-hugging
  attack a kilometre before the line could, and the radio carries the warning back. A brain that does not
  choose keeps its interceptors in the line. Set `guard_depth` to 0.1 to keep every wall on the line.
* **Flight.** An interceptor flies lead pursuit to the predicted intercept point of the attacker it chose,
  climbing or diving at its climb rate, and re-aims every physics step; when it holds, it returns to its
  guard post. An attacker flies a heading (radians off "straight at the line", + towards +x) at a speed
  (0.2 ... 1 of its top speed), and climbs or descends to an `altitude` when its brain sets one - otherwise
  it keeps its height.
* **The terrain mask.** Below `low_altitude` the ground hides a drone: the other side sees it only within
  a third of `sensor_range`. Terrain following costs a quarter of its top speed. Flying low is the
  stealthy, slow way in.
* **Intercepts.** An interceptor that passes within `capture_radius` of an attacker (3-D) takes it out
  with probability `p_kill`. The pass is checked along the whole physics step, so fast closures cannot
  tunnel through. Each interceptor engages at most one attacker per physics step and has `shots` kills
  before it is spent; a miss can try again.
* **Sensors.** Both sides see enemies within `sensor_range` (3-D distance; less for low flyers, see
  above). An interceptor's tracker shows it the `KD = 8` nearest attackers it knows about, an attacker's
  the `K = 4` nearest interceptors.
* **The mesh radio.** `comm_range` metres between friends (3-D). Each message on each link is lost with
  probability `comm_loss`; what a drone hears reflects its friends' last decision. Every drone radios its
  **sightings**, so it knows about every enemy that a friend it hears can see: the whole wall learns
  of an attack that only a few interceptors can see. With `relay` above 1 (*Sighting relay*, Advanced)
  every drone also repeats the tracks it heard, one hop per decision, so a sighting crosses up to `relay`
  hops of the mesh - more warning for drones far from the action, for more radio traffic. Interceptors
  radio their **claim** (which attacker they are going for and how soon they get there, time-stamped so it
  is aged on arrival) and repeat the claims they heard, so claims reach two hops. Attackers radio "I am
  being chased". Both sides add 2 learned channels. Every decision costs a 24-byte header plus the payload
  (10 bytes per sighting, 4 per claim).
* **The end.** A run ends when no attacker is left in the air, or at `duration`.

| number | meaning |
|---|---|
| intercepted / got through / still flying | shares of the attacking swarm (they add up to 100 %) |
| interceptors per target | while attackers are being chased: interceptors locked on per chased attacker (1.0 = perfect sharing) |
| time to intercept | from the first time any interceptor saw an attacker to its kill |
| flying under the terrain mask / mean altitude | share of the attackers' flying time below `low_altitude`; their mean height |
| misses | passes within the capture radius that failed the hit roll |
| CPU load per drone | brain MACs x decisions per second / the board's MAC/s (`DeviceProfile`) |
| radio per drone | kbit/s each drone sends |
| advance | how far each attacker got towards the line, 0 ... 1 (used to score attackers in training) |

### RECON

* **Mission.** A square of side `area` (3 km by default), surveyed by a swarm (12 drones) that lifts off
  from a base in its middle (x east, y north). Drone 0 is the **lead**: it carries the long-range link to
  the operator. Drones fly between `min_altitude` and `max_altitude`.
* **Height is the trade-off.** A camera sees the ground within `altitude * tan(camera_fov / 2)` - a
  100 m radius from 100 m with the default 90 degree lens. The detail it resolves is full up to
  `detail_altitude` and falls to nothing at three times that height. Every cell of the shared
  `grid x grid` coverage map keeps the best detail any camera saw it in (0 ... 1).
* **Surveying and finding.** "Area surveyed" is the mean detail over all cells. Each of the `targets`
  hidden objects needs its own amount of detail to be recognised - some stand out, some hardly do - and is
  **found** when a camera passes over it resolving at least that much. Flying high finds the obvious
  targets fast; the rest need a closer look.
* **Reporting.** A report reaches the operator once its finder is in the lead's radio chain: directly or
  over friends, hop by hop (3-D distances). A drone outside the chain keeps its reports until it is back
  in contact (store and forward).
* **Loads.** A running camera costs the detector (`perception_macs` per frame at `camera_fps`) on the
  drone's board, `device`. Add the swarm brain's MACs, the radio (a state broadcast every decision plus
  16 bytes per report) and the flying (distance, acceleration). Compute and radio energy come from the
  board's `DeviceProfile` (nJ per MAC and per bit).
* **The end.** At `duration`, or as soon as the whole area is surveyed in full detail and every target
  is reported.

| number | meaning |
|---|---|
| area surveyed / time to 90 % | final mean detail over the area / when it first reached 90 % |
| seen | share of the area any camera saw at all, in any detail |
| targets reported / report delay | share of targets whose report reached the operator / mean wait after the find |
| in radio contact | mean share of drones in the lead's radio chain |
| mean altitude | the swarm's mean height over the mission |
| camera on / CPU load | camera duty cycle / (detector + brain) MACs per second over the board's MAC/s |
| radio, energy, flown | kbit/s per drone, compute + radio joules per drone, km per drone |

## Brains

A brain plays one **role** (`recon`, `defender`, `attacker`) for every drone of its side and every
episode at once. Each step it gets that role's observation and returns actions.

| built-in | role | idea | radio |
|---|---|---|---|
| `greedy` | defender | each interceptor goes for the attacker it can reach first; neighbours pile on | - |
| `auction` | defender | **smart wall**: claims are radioed, a target a friend reaches sooner is dropped, the most urgent free one is taken (with a fixed personal preference, so interceptors facing the same crowd do not all bid for the same one) - one interceptor per attacker, the rest stay as reserve | claims |
| `pickets` | defender | the auction, with every third interceptor posted 35 % of the way out as a picket: a low attack is seen a kilometre earlier | claims |
| `zone` | defender | one sector of the line per interceptor; no radio needed, but a crowd in one sector overwhelms it | - |
| `rush` | attacker | straight at the line, full speed, at the height it arrived at | - |
| `spread` | attacker | fan out across the front and through the whole height band | - |
| `evade` | attacker | break hard sideways, and dive or climb away, when an interceptor is about to hit | - |
| `sneak` | attacker | everyone dives under the terrain mask: seen late, but a quarter slower | - |
| `decoy` | attacker | a third fly high and ahead as decoys and drag interceptors up and sideways; the rest hug the terrain and wait until the radio says the interceptors are committed, then sprint | "go" relayed drone to drone |
| `lawnmower` | recon | pre-planned line-abreast sweep at full-detail height, cameras always on | - |
| `flock` | recon | boids on a shared route: always in contact, overlapping cameras | - |
| `frontier` | recon | from high up (wide view, about 70 % detail) to the nearest unseen ground, away from neighbours: sees everything fast, scatters | - |
| `relay_sweep` | recon | frontier at full-detail height on a radio leash, rejoin the lead when cut off, camera only over ground it would improve | leash |

The **library** holds your packages (`brains/`, git-ignored) and the examples in
[examples/brains](../examples/brains):

* `mesh_defender` and `mesh_attacker` co-evolved over two runs. In the first (2000 generations, walls
  could only hold the line) the wall learned to share out targets as well as the auction, and the swarm a
  massed low-level attack at the edge of the front that got 65 % through the auction wall - no wall
  stopped more than 35 % of it. The line was the problem: hidden under the terrain mask, the attack left
  the interceptors 17 s of warning, so only the nine nearest could ever reach it. Once walls could choose
  their guard posts, 3000 more generations (36 minutes on 8 cores) changed the winner. Within 5
  generations the wall moved forward - all of it, to the forward limit, stacked at the flanks - and
  intercepted 92 % of the swarm instead of 34 %. The swarm answered by going to ground out of sensor range
  and creeping in on one flank: 37 % through a picket wall, but only 12 % through the trained one.
* `mesh_recon` is a recon swarm evolved for 3000 generations: it beats the hand-written lawnmower on the
  overall score - nearly the same coverage and radio contact for under half the sensing energy.
* `pincer_attack` is a hand-written, radio-coordinated attack and the example to copy.

### The mesh network

Every trained brain has the same shape: `features -> Dense(hidden) -> ReLU -> Dense(outputs)`. The weights
are shared by every drone of the side; each drone runs the network on its own observation. Its inputs
are listed by name in `swarm.arena.brains.FEATURE_NAMES`, and `python -m swarm.sdk export` prints them.

* **Defender:** the network scores each option - every tracked attacker, then "hold station" - from a row
  of that option's features plus the interceptor's context, and the interceptor takes the best. So "go for a
  free, urgent attacker" is one rule however many it tracks. An untrained network picks the nearest
  attacker, like `greedy`. The "hold" row also gives the radio channels and the guard post: a shift from
  the interceptor's station in the line (up to a quarter of the front sideways, half the depth forwards
  or back, half the ceiling up or down). Among its inputs is a fixed per-drone code (`role`), so one
  network can post some interceptors forward and keep the rest in the line. Networks saved before guard
  posts load with zero weights there: they decide exactly as before.
* **Attacker:** outputs a heading, a speed, a change of height (up to 150 m either way) and two radio
  channels. Untrained, it flies straight at the line at the height it arrived at.
* **Recon:** outputs weights for five steering urges (nearest unseen ground, back to the lead, towards
  friends, away from close friends, towards windows still missing detail), a speed, the camera switch, a
  flying height (a third to three times `detail_altitude`) and two radio channels. Untrained, it flies
  to the nearest gap at full-detail height, like `relay_sweep` without the leash.

`python -m swarm.arena export lib:mesh_defender --role defender --out wall.swm` (or `swarm.sdk export`)
compiles it for the C++ engine, which runs on every board including Cortex-M MCUs. It costs a few thousand
MACs per decision, a negligible share of any of the boards.

## Writing a brain

A Python **policy**: a class with `role` and `act(obs)`. `python -m swarm.sdk new my_wall --type
policy-defender` (or `policy-attacker`, `policy-recon`) writes one that works, to edit:

```python
import numpy as np

class MyWall:
    role = "defender"            # recon | defender | attacker
    title = "My wall"            # optional: shown in the studio
    summary = "..."              # optional
    macs = 300                   # optional: compute per drone per decision, for the CPU figure
    comms = True                 # optional: it coordinates over the radio

    def reset(self, batch, seed):    # optional: before each run; keep per-episode memory here
        pass

    def act(self, obs):              # every array is [B episodes, N drones, ...]
        taken = obs.claim_eta < obs.claim_reference()        # a quicker friend has claimed it
        cost = np.where(obs.slot_mask & ~taken, obs.slot_eta, np.inf)
        return {"target": np.where(np.isfinite(cost.min(-1)), cost.argmin(-1), -1)}
```

`act` works on whole arrays: every drone of every episode at once. Use `np.where` instead of `if` and
avoid Python loops over drones. What `act` returns is checked: missing or unknown keys, wrong shapes,
NaN, or targets out of range stop the run with a message naming your brain.

| role | return | meaning |
|---|---|---|
| defender | `{"target": int}` | slot `0..KD-1` of the observation to go for, `-1` = hold station |
| attacker | `{"heading": float, "speed": float}` | radians off straight-at-the-line (+ towards +x), 0 ... 1 of top speed |
| recon | `{"direction": [.., 2], "speed": float, "camera": bool}` | horizontal direction, any length (zero = hover; a 3-D one steers by its horizontal part), 0 ... 1, camera on/off |
| attacker, recon | `"altitude": float` (optional) | metres to climb or descend to (kept inside the airspace); without it a drone keeps its height |
| defender | `"station": [.., 3]` (optional) | the guard post each interceptor waits at when it holds (x, y, altitude; kept between the line and `guard_depth`); without it, its station in the line (`obs.station`) |
| any | `"msg": [.., 2]` (optional) | values in -1 ... 1 your friends hear next step as `obs.msg` (mean of what each heard) |

Observation fields (`obs.cfg` is the test's settings, `obs.t` the time, `obs.index` each drone's number).
Positions and velocities are 3-D: `[..., 0]` x, `[..., 1]` y, `[..., 2]` altitude.

| defender (`DefenderObs`) | |
|---|---|
| `alive`, `pos`, `vel`, `station` | own state; `station` is its station in the line [B, N, 3] (it moves when the line re-spaces) - where it waits unless the brain returns a `station` of its own |
| `slot_mask`, `slot_id` | which of the KD slots hold an attacker it knows about - seen itself or by a friend it heard (nearest first) - and its track id |
| `slot_rel`, `slot_vel`, `slot_dist` | attacker position relative to me, its velocity, 3-D distance |
| `slot_eta` | my time to intercept it, climbing included (`inf` if I cannot catch it) |
| `slot_wall`, `slot_wall_time` | its distance and time to the line |
| `claims`, `claim_eta` | friends going for it (claims travel two hops), and the best ETA among them, aged to "from now" (`inf` = unclaimed) |
| `slot_mine`, `my_claim_eta` | it is my current target; the ETA I radioed with my claim, aged the same way |
| `claim_reference()` | the ETA a friend's claim must beat to count as theirs (same verdict on both sides) |
| `friends_n`, `friends_rel`, `msg` | friends heard this step, their mean position relative to me, mean message |
| `role()` | a fixed code in -1 ... 1 per drone, e.g. to post every third interceptor forward |

| attacker (`AttackerObs`) | |
|---|---|
| `alive`, `pos`, `vel` | own state (`pos[..., 1]` is the distance to the line, `pos[..., 2]` the height; below `cfg.low_altitude` I am masked) |
| `slot_mask`, `slot_rel`, `slot_vel`, `slot_dist` | the K nearest interceptors it knows about (seen itself or by a friend it heard) |
| `slot_closing` | +1 = flying straight at me |
| `chased` | an interceptor is closing in on me (radioed to friends automatically) |
| `friends_n`, `friends_rel`, `friends_chased`, `msg` | friends heard, their mean position, share of them chased, mean message |
| `role()` | a fixed code in -1 ... 1 per drone, to give drones different jobs from one brain |

| recon (`ReconObs`) | |
|---|---|
| `pos`, `vel` | own state, metres from the south-west corner of the area (the base is in its middle) |
| `linked`, `lead_rel` | in the lead's radio chain; where the lead was when last heard, relative to me |
| `friend_mask`, `friend_rel`, `friends_n` | the 3 nearest friends heard this step |
| `gap_rel` | horizontal: the nearest ground no camera has seen yet, relative to me (zero once all is seen) |
| `look` | detail still missing in a window towards E, NE, N, NW, W, SW, S, SE (0 ... 1) |
| `uncovered_here` | share of the ground in my camera's view that it would still improve from my height |
| `footprint`, `detail` | ground radius my camera sees, and the detail it resolves (1 = full), from my height |
| `msg`, `role()` | as above |

Then:

```powershell
python -m swarm.sdk validate brains/my_wall     # loads it, runs a test, checks determinism, ranks it vs the built-ins
python -m swarm.arena test defence --defender lib:my_wall
```

Or press **LOAD** next to a brain list in the studio and choose the `.py` file. It is installed, checked
and selected straight away.

Python policies run in the simulator and, with your own glue, on Linux companion computers. For MCU-class
drones, train a mesh network for the role and export it to `.swm`.

## Training

Training improves a mesh network by **evolution strategies** (OpenAI-ES):

1. Each generation tries `population` variants of the current weights: `theta +/- sigma * noise`, in
   antithetic pairs.
2. The variants are ranked (centred ranks: only the order counts).
3. The weights step towards the better ones with Adam.

No gradients flow through the simulator, so any score works - "attackers that got through" included.

A generation plays:

* every defender variant against the current attackers and a sparring partner;
* every attacker variant against the current defenders and a sparring partner;
* the current defenders against every built-in attack, and the current attackers against every built-in
  wall (**skill**: who learns faster) - only for a side that learns;
* the current pair head to head (**who is winning**).

All variants meet identical situations **and identical luck**: every episode has a key, and its arrivals
(or hidden targets), radio losses and hit rolls are drawn from that key alone. Variants on the same
situation share a key, so they meet the same lost messages and the same hit rolls, and their scores
differ by what they did, not by luck (common random numbers). The skill and head-to-head games use the
same fixed situations every generation, so a curve moves only when a brain changes. Sparring partners are
earlier versions of the opponent (a hall of fame, one every 5 generations) and the built-ins, so neither
side can win by over-fitting the other's latest quirk. Over a long run that hall of fame keeps the latest
snapshots plus a uniform sample of the opponent's whole history, so a tactic beaten long ago cannot
quietly come back.

### Speed: every CPU core

A generation is split into chunks that run in parallel worker processes, one per CPU core (`workers`, 0 =
all; `--workers` on the command line, *CPU cores* in the studio). Because each episode depends only on
its key, how a generation is split - and on how many cores - changes the speed, never the result: a run
is identical on 1 core or 16. (The exceptions: a Python policy that draws random numbers of its own may
play differently when the split changes, and a native and an emulated interpreter round floating point
differently, so the same seed gives each its own, equally valid run.) With the defaults on an 8-core
machine (Snapdragon X, native ARM64 Python; an x64 Python running emulated in brackets):

| generation | 1 core | 8 cores |
|---|---|---|
| two swarms (24 interceptors against 48 attackers: 97 episodes) | 5 s (8 s) | 0.7 s (1.2 s) |
| recon swarm (12 drones: 40 episodes) | 2.2 s (3.3 s) | 0.43 s (0.75 s) |

So 1000 generations of co-evolution take about 12 minutes. A generation is cut into chunks of at least
6 episodes; a core without a chunk idles, so on a machine with many cores raise `population` - more
variants per generation for the same time. Runs can be long: up to 20,000 generations; a match is still
recorded every `record_every` generations, but at most about 120 per run. While training runs, Windows is
asked not to sleep because nobody is using the computer (a laptop otherwise drops into Modern Standby
after a few idle minutes and the run pauses); closing the lid or pressing the power button still puts it
to sleep, and the run carries on after wake-up.

**Windows on ARM.** The usual Python on an ARM64 PC (the Microsoft Store one) is an x64 build running
emulated, at about half speed. `python dev.py native-python` sets up python.org's ARM64 build of the same
Python version, with the same numpy, in `.native-python/` (git-ignored; downloaded from NuGet and checked
against the SHA-512 NuGet publishes). `python -m swarm.arena train` and the studio's training then run on
it by themselves; everything else stays on your Python. `python dev.py doctor` shows which one trains,
`SWARM_NATIVE_PYTHON=off` keeps training on yours, and deleting the folder removes it.

Calling `train()` from your own Python script: put the call under `if __name__ == "__main__":` (on
Windows, worker processes start by importing your script), or set `workers=1`.

Scores:

* **defender:** (intercepted + 0.25 x still flying - got through) / attackers.
* **attacker:** got through / attackers + 0.1 x mean advance. The advance term exists because an early
  version rewarded attackers for surviving; they learned to hang back out of reach until the clock ran
  out - a loophole, not a tactic.
* **recon:** 0.5 coverage + 0.5 mean coverage over time + 0.5 reported - 0.8 `w_cohesion` (1 - contact)
  - `w_load` (0.2 camera duty + 0.1 effort).

A side that does not learn stays fixed: any brain of its role - built-in, Python or a saved network - is
the opponent the other side trains against. A side starts from a fresh network or continues training a
saved one.

The output folder holds `defender.npz`, `attacker.npz` or `recon.npz` (the latest weights, rewritten
every generation), `defender_best.npz` / `attacker_best.npz` / `recon_best.npz` (the weights that scored
best on the fixed skill test so far: evolution is noisy, and the last generation is not always the best),
`history.json` (rewritten every few seconds) and `matches/gen_NNNN.json`: every `record_every`
generations a recorded match - the first head-to-head game, or the first recon test game, so always the
same situation. In the studio, **SAVE ... BRAIN** turns the best or the latest weights into a library
package. `python -m swarm.sdk install out/arena/run1/defender_best.npz` does the same from the command
line.

## What the arena is not

Drones are 3-D point masses with separate speed and climb limits: no wind, no collisions between friends,
and the ground is flat - the terrain mask is a height below which sensors see less, not a modelled
landscape. Sensors are perfect inside their range. The radio has independent losses - no
interference, jamming or bandwidth limit. Flight energy is not modelled in joules. The numbers compare
brains on equal terms; they are not predictions for a real airframe. To move a winning brain towards
hardware: the `.swm` export runs in the C++ engine on the target board, and the device runtime
([RUNTIME.md](RUNTIME.md)) carries real radio traffic.

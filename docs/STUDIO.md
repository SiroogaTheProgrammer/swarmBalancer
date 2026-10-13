# SWARM//OPS - the test bench

A local application for testing what drone swarms do: run the RECON, DEFENCE and OFFENCE tests, watch
every run in 3D, load your own brains, and let two swarms train against each other. It runs offline and
shows the same numbers as `python -m swarm.arena` ([ARENA.md](ARENA.md) explains what is measured and how).

```powershell
python -m swarm.studio            # or: python dev.py studio   (VS Code: Terminal > Run Task > studio)
```

The bench opens in a chrome-less app window (Edge/Chrome `--app`) or your default browser. The terminal
prints the link. It carries this session's access token, so open the bench through it.

Options: `--port 8770` (the default; any free port if it is taken), `--no-browser`, `--browser` (a normal
tab). Stop it with Ctrl+C or the power button. It needs the repository's Python environment (numpy) and a
browser with WebGL.

## The screen

```
 SWARM//OPS     RECON   DEFENCE   OFFENCE   TRAIN                      BRAINS   ?   power
 +-------------+--------------------------------------+------------------+
 | setup       |  3D replay of episode 0               |  verdict         |
 | - brains    |  (drag: rotate, right-drag: pan,      |  score cards     |
 |   + LOAD    |   wheel: zoom)                        |  same situations,|
 | - settings  |                                       |  other brains    |
 | - RUN TEST  |  > ======o======  speed  radio intent |  recent runs     |
 +-------------+--------------------------------------+------------------+
```

* **Left:** pick the brain under test and its opponent, change the few settings that matter (more under
  *Advanced*), press **RUN TEST**.
* **Middle:** the first episode replays in 3D. Use **space** to play or pause, the slider to scrub, 1x-32x
  for speed. The *radio* and *intent* switches show or hide the mesh links and who is going for whom.
  Drones are drawn as icons that keep their size on screen across a field kilometres wide; each has a
  thin stalk down to the ground, so you can read its height.
* **Right:** the verdict, the key numbers (green = best of all the brains compared, red = worst) and the
  same situations replayed with every built-in brain of that role. Below that, recent runs: click one to
  open it again.

Settings, brain choices and the open tab are remembered between sessions.

## The tests

**RECON** - what you see:

* the area outline, the full-detail height (green outline) and the highest the swarm may fly, and the base
  in the middle (amber ring);
* surveyed cells shaded by the detail they were seen in - brighter is sharper - and glowing for a moment
  when they improve;
* each drone at its height, with its camera's view cone down to its footprint when the camera is on: the
  higher it flies, the wider the footprint and the fainter (less detailed) the cone;
* the lead drone in amber, and radio links in green when they are part of the chain to the lead;
* targets: faint when hidden, amber once found, green once the report has reached the operator.

The HUD counts the area surveyed (mean detail), targets found and reported, drones in contact, cameras on
and the swarm's mean altitude.

**DEFENCE** - your wall against an incoming swarm. What you see:

* the protected plane, from the ground to the ceiling, and the guard posts of the interceptors (a two-layer
  line that closes ranks when an interceptor is spent - unless the wall's brain posts some of them
  forward as pickets);
* the airspace outline and the terrain mask: a faint amber layer near the ground; an attacker below it
  is drawn dark red - the interceptors see it only up close;
* interceptors in cyan, attackers in red, nosing up or down as they climb or dive;
* amber lines from each interceptor to the attacker it is going for, **magenta when two or more go for the
  same one** - that is wasted effort, and what a smart wall avoids;
* flashes for intercepts (white) and breaches (red, on the protected plane at the height it crossed); the
  ticker lists them with their heights.

**OFFENCE** - the same engagement with the attacking swarm's brain under test. Watch how a coordinated
swarm uses height, the terrain mask and its radio, e.g. the pincer's "ready" and "go" calls.

## LOAD a brain

**LOAD** next to a brain list (or **LOAD BRAIN** in the BRAINS drawer, or drag a file onto it) takes:

* a `.py` file with a class that has `role = "defender"` / `"attacker"` / `"recon"` and `act(obs)` - see
  [ARENA.md](ARENA.md#writing-a-brain), or start from `python -m swarm.sdk new my_wall --type
  policy-defender`;
* `.npz` weights trained in TRAIN or with `python -m swarm.arena train`;
* a package `.zip` (`python -m swarm.sdk pack`).

The brain is installed in `brains/` and selected. Its checks run in the background: it must load, return
valid actions through a short test and be deterministic, and it is ranked against the built-ins. A toast
shows the result; click it for the full report.

The **BRAINS** drawer lists everything, by role, with what it costs per decision. You can:

* **CHECK** a brain again;
* **DOWNLOAD** it as a package;
* read the **SOURCE** of a Python brain;
* **DELETE** your own (examples are read-only);
* jump to **TEST IT**.

## TRAIN

Choose **TWO SWARMS** (the wall and the attackers) or **RECON SWARM**. For each side, set whether it
**learns**:

* a learning side starts from a fresh network or keeps training a saved one;
* a side that does not learn stays fixed, as the opponent the other side trains against.

Set the number of generations (up to 20,000) and press **START TRAINING**. Training simulates on all CPU
cores (*Advanced: CPU cores* to leave some free - the results are the same either way). On an 8-core
machine a generation of two swarms (24 interceptors against 48 attackers) takes about 0.7 s and a recon
generation about 0.4 s, so a thousand generations is a coffee break, and an overnight run is fine (a
training job may run up to 48 hours). While it trains, the computer will not go to sleep from being idle;
closing the lid still sends it to sleep, and training carries on after wake-up. On a Windows on ARM PC
whose Python is the usual emulated x64 build, the panel says so: run `python dev.py native-python` once
and training runs on a native interpreter, about twice as fast (see [ARENA.md](ARENA.md#speed-every-cpu-core)).

The right panel updates every generation:

* **Who is winning:** the share of attackers intercepted and getting through when the current brains meet
  head to head, plus a tug-of-war bar.
* **Who learns faster:** each side's score against the built-in opponents. They play the same fixed
  situations every generation, so a rising line is real progress, not luck. On long runs every curve is
  drawn faintly with its moving average on top. The verdict below says who is ahead, who is learning
  faster, and which generation was best so far.
* **Recorded matches:** every few generations a match between the current brains is recorded and plays in
  the 3D view (**follow new matches**, or **WATCH LATEST MATCH**). It is always the same situation, so you
  see the tactics change, not the scenario.
* **SAVE ... BRAIN** puts a trained network into the library under a name of your choice: the **best** one
  on the skill test (evolution is noisy, so the last generation is not always the best) or the **latest**.
  From then on it is in every brain list.

**STOP & KEEP** ends the run after the current generation and keeps the brains; **ABORT** kills it now.
Past runs are listed below; click one to see its curves and matches again.

## Where things are kept

| path | what |
|---|---|
| `brains/` | brains you loaded or saved (git-ignored) |
| `examples/brains/` | the shipped examples (mesh_defender, mesh_attacker, mesh_recon, pincer_attack, ...) |
| `out/studio/runs/` | test reports (the last 60) |
| `out/studio/training/` | training runs: curves, recorded matches, weights (the last 24) |

## How it works

A standard-library HTTP server on 127.0.0.1 serves the page (`python/swarm/studio/static/`: WebGL, no
libraries) and a JSON API. Brain code never runs inside the server. Each test, training run and check runs
`python -m swarm.studio.worker` in its own process, which can be stopped or killed, and streams progress
back.

Security: the server answers only on 127.0.0.1, every API call needs the per-launch token, the Host and
Origin headers must be this server's (no DNS rebinding, no cross-site requests), uploads are size-checked
and unpacked safely, and a strict Content-Security-Policy applies. Uploaded Python brains are code you
choose to run: they are not sandboxed, just as with any script.

The workload benchmark for shared inference (the s01-s12 battery) is not part of the bench. It stays on the
command line: `python -m swarm.bench`.

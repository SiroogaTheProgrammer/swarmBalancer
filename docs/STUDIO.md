# SWARM//OPS - the 3D studio

A tactical console for the swarm: plan a mission, pick or upload brains and tactics, run it, and watch
in 3D how the drones work together. It runs locally, needs no internet, and shows the same numbers as
`python -m swarm.bench`.

```powershell
python -m swarm.studio            # or: python dev.py studio   (VS Code: Terminal > Run Task > studio)
```

The console opens in a chrome-less app window (Edge/Chrome `--app`) or your default browser. The terminal
prints the link; it carries this session's access token, so open the studio through it. Options:
`--port 8770` (default; any free port if taken), `--no-browser`, `--browser` (normal tab). Stop it with
Ctrl+C or **SHUT DOWN** in the console menu. Requirements: the repository's Python environment (numpy) and
a browser with WebGL. Build the C++ engine (`python dev.py build`) to run `.swm` brains natively.

On the first launch the console simulates `s02_leader_loss` (the command drone dies at t = 10 s);
afterwards it reopens your last operation.

## Views

| key | view | what it is for |
|---|---|---|
| F1 | **OPS** | 3D playback of a run: the swarm, its cameras, radio traffic, CPU load, failures, detections |
| F2 | **MISSION PLANNER** | edit a scenario, validate it, EXECUTE it, save it as a preset |
| F3 | **ARMORY** | perception brains and coordination tactics: upload, create, validate, compile, export for drones |
| F4 | **AFTER-ACTION** | the battery's report for a run - every design against the baseline - and the run archive |
| F5 | **FIELD MANUAL** | this guide, the [SDK](SDK.md) and the rest of `docs/`, rendered in the console |

## OPS - reading the theatre

A run simulates **every design** in the scenario on identical conditions; the strategy buttons (keys 1-9)
switch between them at the same moment in time. The default is your plugin tactic if the run has one,
else `striped`.

| you see | it means |
|---|---|
| cyan airframe / amber airframe with a spinning diamond | swarm member / current leader (by its own view) |
| red X, `OFFLINE` | the member is down (killed by the scenario's fault schedule) |
| member flashing red, dashed red line A -> B | A suspects B: B's heartbeats stopped arriving |
| translucent pyramid and box on the ground | camera on, and its nadir footprint |
| ring around a member, filling up cyan -> amber -> red | its CPU load over the last second; a bright dot = computing now |
| thin cyan lines to the leader | command links |
| moving points | messages: frames (cyan, size ~ bytes), feature maps (violet), detection reports (green), commands (amber); red and fading = lost |
| expanding rings | broadcasts: heartbeats (faint white), commands (amber); a big red/green/amber ring = kill / revive / new leader |
| yellow object on the ground | sighted: a frame with something in it was captured, not yet classified |
| red object, closing diamond, `TRK-0123 VEHICLE 93%` | confirmed track: the leader knows (that moment is the mission's latency) |
| amber / magenta / grey | wrong class / false alarm / missed or never processed (frame dropped) |

**What is real and what is presentation.** Every event - captures, inferences, CPU bookings, each
message's queueing, airtime, latency and loss, suspicions, elections, kills - comes from the deterministic
simulation trace. Two things are presentation only: the **geography** (the simulator has none: route,
formation, terrain and where an object sits inside a footprint are a deterministic stage set) and the
**packet flight time**, which is stretched so millisecond messages stay visible (the inspector shows real
times). The **sensor feed** is the actual frame the camera captured in the simulation, with the brain's
answer and the ground truth.

Panels: **ORBAT** (members, role, camera, CPU), **swarm design** (this design's final metrics as ratios
vs the baseline), **inspector** (click a member or a track), **event log** (click a line to jump there),
live counters at the bottom, minimap, and the **timeline**: bars = channel load (red when saturated),
green ticks = detections reported, red ticks = lost messages, markers = kills (red), revives (green) and
elections (amber).

| control | action |
|---|---|
| drag / right-drag (or Shift-drag) / wheel | orbit / pan / zoom |
| click | select a member or a track |
| Space, Left/Right (Shift: 1 s), Home/End | play-pause, step 0.1 s, start/end |
| `[` `]` | previous / next event |
| 1-9 | switch design |
| V / T / F / L / Esc | track swarm / top-down / follow selection / legend / clear selection |

## MISSION PLANNER

Every field of a battery `Scenario`: the ORBAT (any mix of device profiles - the leader is elected by
capability), sensors, the shared radio link, the brain (oracle, a trained `.swm` from `models/`, or an ARMORY
package), the designs under test (built-ins and plugin tactics; the baseline is always run), fault injection
(kill the leader or a member, revive members), measures of effectiveness (checks) and the **theatre**
(terrain, formation, route, altitude, spacing, speed - presentation only, never changes results). The OPORD
panel summarises the mission and shows the server-side validation live.

* **EXECUTE** simulates it in a worker process (progress over the theatre) and opens it in OPS.
* **SAVE PRESET** writes `scenarios/studio/<name>.py` - a real preset: `python scenarios/studio/<name>.py`
  runs it from a terminal; the 3D settings ride along as a `THEATRE` dict the battery ignores.
* **ADD TO BATTERY** writes `scenarios/sNN_name.py` instead (needs a `sNN_` name, a description and checks):
  it becomes part of `python -m swarm.bench` and of the test suite.
* Presets not created in the studio are never overwritten; only studio presets can be deleted.

## ARMORY

All brains and tactics: the oracle, every `models/**/*.swm`, the built-in designs, the examples in
`examples/brains/` and your packages in `brains/`. Select one for its dossier: validation report, device fit
(per device: time per frame, RAM fit under the device's cap, which runtime can host it), conformance runs
for tactics, source files.

* **Upload** a `.zip` package, a compiled `.swm`, or a single `.py` brain/tactic; it is validated at once.
* **CREATE** scaffolds a package in `brains/` that validates out of the box - then edit it in your editor
  and press VALIDATE.
* **COMPILE** runs a `.swm` package's build script (train + export) with its log streamed live.
* **EXPORT FOR DEVICE** writes the device artefacts to `out/studio/exports/` with the exact next commands.
* **USE IN MISSION** puts the brain or tactic into the MISSION PLANNER.

How to write packages: [SDK.md](SDK.md).

## AFTER-ACTION

The battery's metric tables for the run - mission & speed, compute/radio/energy, survivability against a
fault-free twin run - with ratios against the baseline coloured by whether they are better or worse, the
checks, comparison charts and, if requested, the device stress test. Runs are kept in `out/studio/runs/`
(the newest 40); **REPLAY IN OPS** reopens one.

## Files

| path | contents |
|---|---|
| `scenarios/studio/` | presets saved by the studio (runnable, version them) |
| `brains/` | uploaded / created packages (git-ignored except its README) |
| `examples/brains/` | example packages |
| `out/studio/runs/<id>/` | `run.json` (scenario, metrics, traces), the recorded camera frames |
| `out/studio/validation/`, `exports/`, `presets.json` | validation cache, device exports, preset index |

## Security model

The studio can run code you upload, so it is built to be reachable by you only:

* the server binds to **127.0.0.1**; every API call needs the **per-launch session token** from the launch
  link (kept in the URL fragment, which browsers never send to servers or in referrers);
* the `Host` header must be this server (blocks DNS rebinding) and a browser `Origin`/`Sec-Fetch-Site`
  must be same-origin (blocks other web pages); there is no CORS;
* a strict Content-Security-Policy, no framing, bounded request bodies, static files from one folder;
* **the server never executes package or preset code**: simulations, validation, compiles and the preset
  index run in separate worker processes with time limits. That is isolation for robustness, not a sandbox:
  package code runs with your user's rights, like any script you start. Only upload code you trust.

## How it works

`swarm.studio.server` (standard-library HTTP) serves `static/` (hand-written WebGL, no external libraries)
and a JSON API. EXECUTE starts `python -m swarm.studio.worker`, which loads the scenario's tactic packages,
runs `swarm.bench.run_scenario(..., trace=True)` and stores the result. `trace=True` attaches a
`swarm.sim.trace.TraceRecorder` to each simulation through `Simulator.tracer`: it only observes - it never
schedules events or draws random numbers - so traced metrics are identical to the battery's
(`tests/test_studio.py` checks this). Row layouts of a trace are `swarm.sim.trace.SCHEMA`.

## Troubleshooting

* *ACCESS DENIED* - open the link printed by `python -m swarm.studio` (a new tab needs the token).
* *BRAIN ENGINE: NUMPY* - `python dev.py build`; `python -c "from swarm.brain import native; print(native.diagnosis())"`.
* *MISSION FAILED* - the banner shows the worker's error and log tail; the same scenario runs from a terminal
  with SAVE PRESET + `python scenarios/studio/<name>.py`.
* A package stays *unvalidated* - VALIDATE it in the ARMORY; `.swm` packages validate after COMPILE.
* Slow playback on huge runs - long durations with many drones produce large traces; shorten the duration
  or reduce fps. Traces beyond ~3 M rows are truncated (flagged in the OPS header).

"""The arena test bench: environments, brains, training, the test runner and policy packages."""

from __future__ import annotations

import json

import numpy as np
import pytest

from swarm.arena import (BUILTINS, AttackerObs, BrainError, ClashConfig, ClashEnv, ConfigError, DefenderObs,
                         MixedBrain, NeuralBrain, PythonBrain, ReconConfig, ReconEnv, ReconObs, TrainConfig,
                         check_actions, from_dict, list_brains, load_brain, play_clash, play_recon, run_test)
from swarm.arena.brains import FEATURE_NAMES, OUTPUT_NAMES, OUTPUTS
from swarm.arena.config import schema
from swarm.arena.core import K, KD, decide_every, intercept_time, nearest
from swarm.arena.es import ES, centred_ranks
from swarm.arena.recon import QUALITY_STEP
from swarm.sdk import PackageError

SMALL = ClashConfig(defenders=4, attackers=6, width=800.0, depth=700.0, ceiling=300.0, duration=40.0)
SMALL_SETTINGS = {"defenders": 4, "attackers": 6, "width": 800.0, "depth": 700.0, "ceiling": 300.0, "duration": 40.0}
SMALL_RECON = ReconConfig(drones=3, area=600.0, duration=80.0, targets=5, grid=12)


def clash(defender, attacker, cfg=SMALL, batch=6, seed=3, record=False):
    env = ClashEnv(cfg, batch, seed, record=record)
    d = defender if not isinstance(defender, str) else load_brain(defender, "defender")
    a = attacker if not isinstance(attacker, str) else load_brain(attacker, "attacker")
    return env, play_clash(env, d, a)


# ----------------------------------------------------------------------------- settings and geometry
def test_settings_are_checked():
    assert from_dict(ClashConfig, {"attackers": 20}).attackers == 20
    for bad, needle in [({"attackers": 0}, "attackers"), ({"attackers": 2.5}, "whole"), ({"nope": 1}, "unknown"),
                        ({"spawn": "zigzag"}, "spawn"), ({"defender_device": "toaster"}, "device"),
                        ({"p_kill": float("nan")}, "number"), ({"shots": True}, "number")]:
        with pytest.raises(ConfigError, match=needle):
            from_dict(ClashConfig, bad)
    with pytest.raises(ConfigError, match="even"):
        from_dict(TrainConfig, {"population": 7})
    rows = {r["name"]: r for r in schema(ClashConfig)}
    assert rows["attackers"]["basic"] and rows["attackers"]["min"] == 1
    assert rows["spawn"]["choices"] == ["line", "cluster", "waves"] and "host" not in rows["defender_device"]["choices"]
    with pytest.raises(ConfigError, match="low_altitude"):
        from_dict(ClashConfig, {"ceiling": 60, "low_altitude": 40})      # the terrain mask must leave room above
    with pytest.raises(ConfigError, match="min_altitude"):
        from_dict(ReconConfig, {"min_altitude": 300, "max_altitude": 200})


def test_intercept_time_and_nearest():
    assert intercept_time(np.array([100.0, 0.0]), np.array([10.0, 0.0]), 20.0) == pytest.approx(10.0)
    assert np.isinf(intercept_time(np.array([100.0, 0.0]), np.array([25.0, 0.0]), 20.0))  # faster target
    idx, ok = nearest(np.array([[[3.0, np.inf, 1.0]]]), K)
    assert idx[0, 0, :2].tolist() == [2, 0] and ok[0, 0].tolist() == [True, True, False, False]


# ----------------------------------------------------------------------------- the clash (DEFENCE / OFFENCE)
def test_clash_bookkeeping_is_consistent_and_deterministic():
    env, s = clash("auction", "decoy")
    assert np.allclose(s["intercepted"] + s["leaked"] + s["held_off"], 1.0)
    shots_used = SMALL.shots - env.ammo
    assert (env.ammo >= 0).all() and (shots_used.sum(1) >= env.a_killed.sum(1)).all()
    assert not (env.a_killed & env.a_breached).any()
    _, again = clash("auction", "decoy")
    for k, v in s.items():
        np.testing.assert_array_equal(np.nan_to_num(v), np.nan_to_num(again[k]), err_msg=k)
    assert all(np.isfinite(v).all() for k, v in s.items() if k != "kill_time")


def test_the_smart_wall_spreads_its_interceptors():
    cfg = ClashConfig(duration=100.0)
    _, auction = clash("auction", "rush", cfg, batch=12)
    _, greedy = clash("greedy", "rush", cfg, batch=12)
    assert auction["overkill"].mean() < greedy["overkill"].mean() - 0.3   # fewer interceptors per target
    assert auction["intercepted"].mean() > greedy["intercepted"].mean()


def _placed(cfg, attackers):
    """A clash with the interceptors on their stations and the attackers at the given (x, y, z)."""
    env = ClashEnv(cfg, 1, seed=0)
    env.dp = env.station.copy()
    env.ap = np.array(attackers, float)[None]
    env._pair()
    return env


def test_terrain_mask_hides_low_flyers_and_friends_share_what_they_see():
    cfg = ClashConfig(defenders=4, attackers=3, width=800.0, depth=700.0, ceiling=300.0, comm_loss=0.0)
    hidden_range = cfg.sensor_range / 3
    # a low attacker 620 m out (hidden: seen only within 500 m), the same one high (seen), one far and low
    env = _placed(cfg, [[-300.0, 690.0, 20.0], [-300.0, 690.0, 120.0], [300.0, 1000.0, 10.0]])
    known = env.defender_obs().slot_id[0]
    assert 1 in known and 0 not in known and 2 not in known
    # now the low one comes within 500 m of interceptor 0 only: interceptor 3 cannot see it itself, but
    # hears interceptor 0's sighting over the mesh
    env = _placed(cfg, [[-300.0, 400.0, 20.0], [300.0, 1000.0, 10.0], [-300.0, 1000.0, 10.0]])
    assert env.dist[0, 0, 0] < hidden_range < env.dist[0, 3, 0]
    assert np.linalg.norm(env.dp[0, 0] - env.dp[0, 3]) < cfg.comm_range
    o = env.defender_obs()
    assert 0 in o.slot_id[0, 0] and 0 in o.slot_id[0, 3] and 1 not in o.slot_id[0, 3]


def test_sightings_relay_hop_by_hop():
    # four interceptors in a chain 750 m apart (radio 800 m); only the first sees the low attacker
    def knows(relay, decisions):
        cfg = ClashConfig(defenders=4, attackers=1, width=3000.0, depth=700.0, ceiling=300.0, comm_loss=0.0,
                          relay=relay)
        env = _placed(cfg, [[-1125.0, 470.0, 20.0]])
        for _ in range(decisions):
            o = env.defender_obs()
        return (o.slot_mask & (o.slot_id == 0)).any(-1)[0].tolist()

    assert knows(1, 3) == [True, True, False, False]           # one hop: only the drone that hears the spotter
    assert knows(3, 1) == [True, True, False, False]           # one hop per decision ...
    assert knows(3, 3) == [True, True, True, True]             # ... up to `relay` hops
    assert knows(2, 3) == [True, True, True, False]


def test_brains_may_post_interceptors_forward():
    env = ClashEnv(SMALL, 2, seed=0, record=True)
    o = env.defender_obs()
    want = o.station.copy()
    want[0, :, 1] = 5000.0                                     # too far out: kept at the forward limit
    want[0, 0] = np.nan                                        # NaN: its station in the line
    env.apply_defenders(o, np.full((2, SMALL.defenders), -1), station=want)
    limit = SMALL.guard_depth * SMALL.depth
    posts = env.posts()
    assert np.allclose(posts[0, 1:, 1], limit) and np.allclose(posts[0, 0], env.station[0, 0])
    assert np.allclose(posts[1], env.station[1, :])                    # episode 1 asked for its line ...
    assert env.restations and np.allclose(env.restations[0]["station"][1][1], limit)   # ... and the replay shows it
    for _ in range(8):
        env.step()
    flown = env.dp[:, :, 1] - env.station[:, :, 1]
    assert (flown[0, 1:] > 150).all() and abs(flown[0, 0]) < 30 and (np.abs(flown[1]) < 30).all()


def test_pickets_spot_a_terrain_hugging_attack_sooner():
    o = ClashEnv(ClashConfig(), 1, seed=0).defender_obs()
    post = load_brain("pickets", "defender").act(o)["station"]
    forward = post[0, :, 1] > o.station[0, :, 1] + 1
    assert forward.sum() == ClashConfig().defenders // 3 and forward[1::3].all()
    cfg = ClashConfig(duration=100.0)
    _, line = clash("auction", "sneak", cfg, batch=6)
    _, pickets = clash("pickets", "sneak", cfg, batch=6)
    assert pickets["intercepted"].mean() > line["intercepted"].mean() + 0.05


def test_climbing_is_slower_than_flying_level():
    env = ClashEnv(SMALL, 1, seed=0)
    z0, y0 = env.ap[0, :, 2].copy(), env.ap[0, :, 1].copy()
    env.apply_attackers(env.attacker_obs(), np.zeros((1, SMALL.attackers)), np.ones((1, SMALL.attackers)),
                        altitude=np.full((1, SMALL.attackers), SMALL.ceiling))
    for _ in range(5):
        env.step()
    climbed = env.ap[0, :, 2] - z0
    assert (climbed > 0).all() and (climbed <= SMALL.attacker_climb * 5 + 1e-6).all()
    assert ((y0 - env.ap[0, :, 1]) > 2 * climbed).all()                   # flew much further than it climbed


def test_observations_are_finite_and_masked():
    env = ClashEnv(SMALL, 3, seed=1)
    for _ in range(25):
        env.step()
    do, ao = env.defender_obs(), env.attacker_obs()
    f, g = do.features(), ao.features()
    assert f.shape == (3, SMALL.defenders, KD + 1, DefenderObs.FEATURES) and np.isfinite(f).all()
    assert g.shape == (3, SMALL.attackers, AttackerObs.FEATURES) and np.isfinite(g).all()
    assert ao.slot_mask.shape[-1] == K and do.slot_mask.shape[-1] == KD
    assert (f[..., :KD, :DefenderObs.SLOT_FEATURES][~do.slot_mask] == 0).all()   # empty slots carry zeros
    assert (f[..., KD, DefenderObs.SLOT_FEATURES] == 1).all()                    # the last row is "hold"
    for role, cls in (("defender", DefenderObs), ("attacker", AttackerObs), ("recon", ReconObs)):
        assert len(FEATURE_NAMES[role]) == cls.FEATURES and len(OUTPUT_NAMES[role]) == OUTPUTS[role]


def test_replay_records_the_first_episode():
    env, _ = clash("auction", "decoy", batch=2, record=True)
    r = env.replay()
    assert r["kind"] == "clash" and len(r["frames"]) == env.steps + 1
    assert all(len(f["d"]) == 3 * SMALL.defenders and len(f["a"]) == 3 * SMALL.attackers for f in r["frames"])
    assert r["ceiling"] == SMALL.ceiling and len(r["station"][0]) == 3
    assert sum(e["type"] == "kill" for e in r["events"]) >= env.a_killed[0].sum()
    assert sum(e["type"] == "breach" for e in r["events"]) == env.a_breached[0].sum()
    assert all(0 <= e["z"] <= SMALL.ceiling for e in r["events"] if e["type"] in ("kill", "breach"))
    json.dumps(r, allow_nan=False)


def test_a_brain_too_slow_for_its_board_decides_less_often():
    class Heavy:
        role, macs = "attacker", 10**9

        def act(self, obs):
            return {"heading": 0.0, "speed": 1.0}

    heavy = PythonBrain(Heavy(), "attacker", "heavy")
    assert decide_every(heavy.macs, SMALL.attacker_device, SMALL.dt) > 1
    env, s = clash("greedy", heavy, batch=2)
    light_env, _ = clash("greedy", "rush", batch=2)
    assert env.stat["dec_a"].sum() < light_env.stat["dec_a"].sum() / 5
    assert (s["cpu_att"] <= 1.0 + 1e-9).all()


# ----------------------------------------------------------------------------- recon
def test_recon_bookkeeping_and_replay():
    env = ReconEnv(SMALL_RECON, 4, seed=2, record=True)
    s = play_recon(env, load_brain("lawnmower", "recon"))
    for key in ("coverage", "seen", "connected"):
        assert ((0 <= s[key]) & (s[key] <= 1)).all(), key
    assert (s["coverage"] <= s["seen"] + 1e-9).all()      # detail counts, not just having flown over
    assert not (env.reported & ~env.found).any()
    r = env.replay()
    level = np.zeros(SMALL_RECON.grid ** 2, int)          # the replay's coverage map, replayed
    for f in r["frames"]:
        assert len(f["p"]) == 3 * SMALL_RECON.drones
        level[f["nc"]] = f["nq"]
    assert (level == np.floor(env.quality[0] / QUALITY_STEP + 1e-9)).all()
    assert r["base"] == [SMALL_RECON.area / 2] * 2
    json.dumps(r, allow_nan=False)


def test_recon_baselines_show_the_tradeoffs():
    cfg = ReconConfig()
    s = {n: play_recon(ReconEnv(cfg, 4, seed=5), load_brain(n, "recon"))
         for n in ("lawnmower", "flock", "frontier", "relay_sweep")}
    assert s["lawnmower"]["coverage"].mean() > s["flock"]["coverage"].mean() + 0.2      # stays together, sees little
    assert s["relay_sweep"]["camera_duty"].mean() < 0.5 < s["lawnmower"]["camera_duty"].mean()  # load-aware
    assert s["relay_sweep"]["connected"].mean() > s["frontier"]["connected"].mean()     # keeps the radio chain
    assert s["relay_sweep"]["cpu"].mean() < s["lawnmower"]["cpu"].mean() / 2
    # height: flying high sees ground faster but in less detail
    assert s["frontier"]["altitude"].mean() > s["lawnmower"]["altitude"].mean() + 30
    assert s["frontier"]["seen"].mean() > s["frontier"]["coverage"].mean() + 0.1


# ----------------------------------------------------------------------------- brains
def test_untrained_networks_start_from_the_neutral_action():
    env = ClashEnv(SMALL, 2, seed=4)
    for _ in range(5):
        env.step()
    o = env.defender_obs()
    target = NeuralBrain("defender").act(o)["target"]
    assert (target == np.where(o.slot_mask[..., 0], 0, -1)).all()          # nearest first, like "greedy"
    ao = env.attacker_obs()
    a = NeuralBrain("attacker").act(ao)
    assert np.allclose(a["heading"], 0.0) and (a["speed"] > 0.85).all()     # straight for the line ...
    assert np.allclose(a["altitude"], ao.pos[..., 2])                       # ... at the height it flies
    ro = ReconEnv(SMALL_RECON, 2, seed=1).observe()
    r = NeuralBrain("recon").act(ro)
    d = r["direction"]
    cos = (d * ro.gap_rel).sum(-1) / (np.linalg.norm(d, axis=-1) * np.linalg.norm(ro.gap_rel, axis=-1))
    assert np.allclose(cos, 1.0)                                            # to the nearest gap ("frontier") ...
    assert np.allclose(r["altitude"], SMALL_RECON.detail_altitude)          # ... at full-detail height
    assert np.allclose(ro.footprint, ro.pos[..., 2] * np.tan(np.radians(SMALL_RECON.camera_fov / 2)))


def test_mesh_brain_batches_saves_and_compiles(tmp_path):
    rng = np.random.default_rng(0)
    env = ClashEnv(SMALL, 3, seed=1)
    for _ in range(20):
        env.step()
    x = env.defender_obs().features()
    b = NeuralBrain("defender", hidden=8, seed=1)
    b.theta = b.theta + rng.normal(0, 0.2, b.size)
    single = b.forward(x)
    np.testing.assert_allclose(b.forward(x, np.repeat(b.theta[None], 3, 0)), single, atol=1e-12)
    loaded = NeuralBrain.load(b.save(tmp_path / "d.npz", generations=3))
    assert loaded.role == "defender" and loaded.meta["generations"] == 3
    np.testing.assert_allclose(loaded.forward(x), single, atol=1e-4)
    from swarm.brain import native
    from swarm.brain.model import Sequential

    path = b.to_swm(tmp_path / "d.swm")
    rows = x.reshape(-1, x.shape[-1]).astype(np.float32)
    expect = single.reshape(len(rows), -1)
    model = Sequential.load(path)
    np.testing.assert_allclose(model.forward(rows.reshape(len(rows), -1, 1, 1)), expect, atol=1e-4)
    if native.available():
        with native.NativeModel(path) as m:
            np.testing.assert_allclose(m.run_batch(rows.reshape(len(rows), -1, 1, 1)), expect, atol=1e-4)
    meta = json.loads(str(np.load(tmp_path / "d.npz")["meta"]))
    bad = tmp_path / "bad.npz"
    np.savez(bad, theta=b.theta, meta=np.array(json.dumps({**meta, "n_in": 3})))
    with pytest.raises(BrainError, match="observation layout"):
        NeuralBrain.load(bad)


def test_mixed_brain_matches_separate_runs():
    env = ClashEnv(SMALL, 4, seed=7)
    for _ in range(12):
        env.step()
    o = env.defender_obs()
    greedy, auction = load_brain("greedy", "defender"), load_brain("auction", "defender")
    t = MixedBrain("defender", [(greedy, np.array([0, 2]), None), (auction, np.array([1, 3]), None)]).act(o)["target"]
    assert (t[[0, 2]] == greedy.act(o.take([0, 2]))["target"]).all()
    assert (t[[1, 3]] == auction.act(o.take([1, 3]))["target"]).all()
    # an optional action only one of the brains returns: the other's drones keep their last command
    ao = env.attacker_obs()
    rush, sneak = load_brain("rush", "attacker"), load_brain("sneak", "attacker")
    a = MixedBrain("attacker", [(rush, np.array([0, 2]), None), (sneak, np.array([1, 3]), None)]).act(ao)
    before = env.alt_cmd.copy()
    env.apply_attackers(ao, a["heading"], a["speed"], altitude=a["altitude"])
    assert np.array_equal(env.alt_cmd[[0, 2]], before[[0, 2]]) and np.allclose(env.alt_cmd[[1, 3]], 0.6 * SMALL.low_altitude)
    d = MixedBrain("defender", [(auction, np.array([0, 2]), None),
                                (load_brain("pickets", "defender"), np.array([1, 3]), None)]).act(o)
    assert np.isnan(d["station"][[0, 2]]).all() and np.isfinite(d["station"][[1, 3]]).all()


def test_mesh_defenders_saved_before_guard_posts_still_load(tmp_path):
    from swarm.arena.brains import NPZ_FORMAT, OLD_DEFENDER

    (i0, o0), h = OLD_DEFENDER, 8
    old = np.random.default_rng(3).normal(0, 0.3, i0 * h + h + h * o0 + o0).astype(np.float32)
    meta = {"format": NPZ_FORMAT, "role": "defender", "hidden": h, "n_in": i0, "n_out": o0, "name": "old"}
    np.savez(tmp_path / "old.npz", theta=old, meta=np.array(json.dumps(meta)))
    b = NeuralBrain.load(tmp_path / "old.npz")
    assert (b.n_in, b.n_out) == (DefenderObs.FEATURES, OUTPUTS["defender"])
    env = ClashEnv(SMALL, 2, seed=1)
    for _ in range(10):
        env.step()
    o = env.defender_obs()
    x, w = o.features(), old.astype(float)
    w1, b1 = w[:i0 * h].reshape(i0, h), w[i0 * h:i0 * h + h]
    w2, b2 = w[i0 * h + h:-o0].reshape(h, o0), w[-o0:]
    np.testing.assert_allclose(b.forward(x)[..., :o0], np.maximum(x[..., :i0] @ w1 + b1, 0) @ w2 + b2, atol=1e-9)
    assert "station" not in b.act(o)                       # zero weights there: it keeps the line, as it did


def test_actions_from_user_code_are_checked():
    o = ClashEnv(SMALL, 2, seed=0).defender_obs()
    assert check_actions("defender", {"target": 0}, o)["target"].shape == (2, SMALL.defenders)
    for bad, needle in [([1, 2], "dict"), ({}, "missing"), ({"target": 9}, "-1"), ({"target": 0.5}, "whole"),
                        ({"target": 0, "fly": 1}, "unknown"), ({"target": np.zeros((5, 5))}, "shape")]:
        with pytest.raises(BrainError, match=needle):
            check_actions("defender", bad, o)
    with pytest.raises(BrainError, match="NaN"):
        check_actions("attacker", {"heading": np.nan, "speed": 1.0}, ClashEnv(SMALL, 2, 0).attacker_obs())
    with pytest.raises(BrainError, match="NaN"):
        check_actions("attacker", {"heading": 0.0, "speed": 1.0, "altitude": np.nan}, ClashEnv(SMALL, 2, 0).attacker_obs())
    ro = ReconEnv(SMALL_RECON, 2, seed=0).observe()
    up = check_actions("recon", {"direction": np.ones((2, 3, 3)), "speed": 1.0, "camera": True}, ro)
    assert up["direction"].shape == (2, 3, 2)                               # a 3-D direction steers horizontally
    with pytest.raises(BrainError, match="defender brain"):
        load_brain("rush", "defender")


# ----------------------------------------------------------------------------- training
def test_es_climbs_a_hill():
    target = np.array([1.0, -2.0, 0.5])
    es = ES(np.zeros(3), sigma=0.1, lr=0.1, pairs=8, seed=0, l2=0.0)
    for _ in range(150):
        pop = es.ask()
        es.tell(-((pop - target) ** 2).sum(1))
    assert np.abs(es.theta - target).max() < 0.2
    assert centred_ranks(np.array([3.0, 1.0, 2.0])).tolist() == [0.5, -0.5, 0.0]


def test_training_writes_brains_history_and_matches(tmp_path):
    from swarm.arena.train import train

    tc = TrainConfig(generations=2, population=4, episodes=1, record_every=1, hidden=4, workers=1)
    events = []
    res = train(tc, SMALL, out_dir=tmp_path / "both", report=events.append)
    assert res["generations"] == 2 and not res["stopped"] and res["workers"] == 1
    assert [e["type"] for e in events] == ["start", "generation", "generation"]
    assert {"intercepted", "leaked", "advance"} <= set(events[-1]["h2h"]) and events[-1]["match"] == "gen_0002.json"
    for role in ("defender", "attacker"):
        assert NeuralBrain.load(tmp_path / "both" / f"{role}.npz").meta["generations"] == 2
        best = res["best"][role]
        assert best["gen"] in (1, 2) and NeuralBrain.load(best["path"]).meta["skill"] == best["skill"]
    assert events[-1]["best"]["defender"]["gen"] == res["best"]["defender"]["gen"]
    match = json.loads((tmp_path / "both" / "matches" / "gen_0001.json").read_text(encoding="utf-8"))
    assert match["replay"]["kind"] == "clash"
    assert train(tc, SMALL, out_dir=tmp_path / "stop", should_stop=lambda: True)["generations"] == 0
    one = TrainConfig(generations=1, population=4, episodes=1, hidden=4, learn_attackers=False, start_attacker="rush",
                      workers=1)
    solo = train(one, SMALL, out_dir=tmp_path / "def")
    assert list(solo["brains"]) == ["defender"] and solo["history"][0]["attacker_skill"] is None  # no skill test
    rec = train(TrainConfig(mode="recon", generations=1, population=4, episodes=1, hidden=4, workers=1),
                recon=SMALL_RECON, out_dir=tmp_path / "recon")
    assert NeuralBrain.load(rec["brains"]["recon"]).role == "recon" and set(rec["references"]) == {
        n for n, c in BUILTINS.items() if c.role == "recon"}
    assert rec["best"]["recon"]["gen"] == 1
    with pytest.raises(BrainError, match="cannot learn"):
        train(TrainConfig(generations=1, population=4, start_defender="auction"), SMALL, out_dir=tmp_path / "x")


def test_training_gives_the_same_result_on_any_number_of_cores(tmp_path):
    from swarm.arena.train import train

    def run(workers, mode="clash"):
        tc = TrainConfig(mode=mode, generations=2, population=4, episodes=1, hidden=4, record_every=1, workers=workers)
        kw = {"recon": SMALL_RECON} if mode == "recon" else {"clash": SMALL}
        res = train(tc, out_dir=tmp_path / f"{mode}{workers}", **kw)
        rows = [{k: v for k, v in r.items() if k != "seconds"} for r in res["history"]]
        return rows, {role: np.load(p)["theta"] for role, p in res["brains"].items()}

    for mode in ("clash", "recon"):
        one, many = run(1, mode), run(2, mode)
        assert one[0] == many[0], mode
        assert all(np.array_equal(one[1][r], many[1][r]) for r in one[1]), mode


# ----------------------------------------------------------------------------- keyed episodes and parallel play
def test_keyed_episodes_do_not_depend_on_their_batch():
    from swarm.arena.core import derive
    from swarm.arena.parallel import split

    keys = np.array([derive(9, i % 3) for i in range(9)], dtype=np.uint64)  # 3 situations, 3 episodes each
    rng = np.random.default_rng(0)
    mesh = NeuralBrain("defender", hidden=8, seed=2)
    rows = mesh.theta + 0.3 * rng.standard_normal((9, mesh.size))

    def clash_run(idx):
        defender = MixedBrain("defender", [(mesh, np.arange(len(idx)), rows[idx])])
        return play_clash(ClashEnv(SMALL, len(idx), seed=int(idx[0]), keys=keys[idx]), defender, load_brain("evade", "attacker"))

    whole = clash_run(np.arange(9))
    parts = [clash_run(np.arange(9)[s]) for s in split(9, 3)]
    for k, v in whole.items():
        np.testing.assert_array_equal(np.nan_to_num(v), np.nan_to_num(np.concatenate([p[k] for p in parts])), err_msg=k)
    # relayed tracks and guard posts too
    from dataclasses import replace

    relayed = replace(SMALL, relay=3)
    pickets, sneak = load_brain("pickets", "defender"), load_brain("sneak", "attacker")
    whole = play_clash(ClashEnv(relayed, 6, keys=keys[:6]), pickets, sneak)
    parts = [play_clash(ClashEnv(relayed, 3, seed=5, keys=keys[s]), pickets, sneak) for s in (slice(0, 3), slice(3, 6))]
    for k, v in whole.items():
        np.testing.assert_array_equal(np.nan_to_num(v), np.nan_to_num(np.concatenate([p[k] for p in parts])), err_msg=k)
    # the same key and the same brains: the same episode (common random numbers)
    twin = play_clash(ClashEnv(SMALL, 2, keys=keys[[0, 0]]), load_brain("auction", "defender"), load_brain("decoy", "attacker"))
    assert all(np.array_equal(np.nan_to_num(v[0]), np.nan_to_num(v[1])) for v in twin.values())
    recon = load_brain("relay_sweep", "recon")
    whole = play_recon(ReconEnv(SMALL_RECON, 6, keys=keys[:6]), recon)
    parts = [play_recon(ReconEnv(SMALL_RECON, 3, seed=7, keys=keys[s]), recon) for s in (slice(0, 3), slice(3, 6))]
    for k, v in whole.items():
        np.testing.assert_array_equal(np.nan_to_num(v), np.nan_to_num(np.concatenate([p[k] for p in parts])), err_msg=k)
    assert [s.stop - s.start for s in split(40, 8)] == [7, 6, 7, 7, 6, 7] and len(split(5, 8)) == 1


@pytest.mark.parametrize("relay", [1, 3])
def test_brains_in_one_batch_keep_their_own_pace(relay):
    class Heavy:
        role, macs = "attacker", 10**9   # too slow for the attackers' board: decides every few steps
        calls = 0

        def act(self, obs):
            Heavy.calls += 1
            return {"heading": 0.0, "speed": 1.0}

    from dataclasses import replace

    cfg = replace(SMALL, relay=relay)     # relayed tracks age only in the episodes that decide
    heavy, rush = PythonBrain(Heavy(), "attacker", "heavy"), load_brain("rush", "attacker")
    keys = np.arange(1, 5, dtype=np.uint64)
    env = ClashEnv(cfg, 4, keys=keys)
    s = play_clash(env, load_brain("greedy", "defender"),
                   MixedBrain("attacker", [(heavy, np.array([0, 1]), None), (rush, np.array([2, 3]), None)]))
    every = decide_every(heavy.macs, cfg.attacker_device, cfg.dt)
    assert Heavy.calls <= env.steps // every + 1           # asked only on its own turns
    alone = play_clash(ClashEnv(cfg, 2, keys=keys[2:]), load_brain("greedy", "defender"), rush)
    for k, v in alone.items():                            # the fast brain's episodes are unaffected by its neighbour
        np.testing.assert_array_equal(np.nan_to_num(v), np.nan_to_num(s[k][2:]), err_msg=k)


def test_library_policies_reach_worker_processes_by_id():
    import pickle

    pincer = load_brain("lib:pincer_attack", "attacker")
    clone = pickle.loads(pickle.dumps(pincer))
    assert clone.id == "lib:pincer_attack" and type(clone.impl).__name__ == type(pincer.impl).__name__
    env = ClashEnv(SMALL, 2, keys=np.array([3, 4], dtype=np.uint64))
    assert play_clash(env, load_brain("auction", "defender"), clone)["leaked"].shape == (2,)


# ----------------------------------------------------------------------------- the test runner
def test_run_test_report():
    doc = run_test("defence", SMALL_SETTINGS, {"defender": "auction", "attacker": "rush"}, episodes=3, seed=1)
    json.dumps(doc, allow_nan=False)
    assert doc["verdict"]["label"] and {"intercepted", "leaked", "overkill"} <= {c["key"] for c in doc["cards"]}
    assert sorted(r["brain"]["id"] for r in doc["compare"]) == sorted(n for n, c in BUILTINS.items() if c.role == "defender")
    assert sum(r["selected"] for r in doc["compare"]) == 1 and doc["replay"]["frames"]
    off = run_test("offence", SMALL_SETTINGS, {"attacker": "decoy"}, episodes=2, compare=False)
    assert off["subject"] == "attacker" and len(off["compare"]) == 1
    with pytest.raises(ValueError, match="attacker brain"):
        run_test("offence", None, {"attacker": "auction"})


# ----------------------------------------------------------------------------- policy packages
def test_policy_packages_templates_uploads_and_weights(tmp_path):
    from swarm.sdk import load_package, validate
    from swarm.sdk.install import install_npz, install_source
    from swarm.sdk.templates import new_package

    for role in ("defender", "attacker", "recon"):
        root = new_package(tmp_path, f"t_{role}", f"policy-{role}")
        rep = validate(root)
        assert rep.ok, rep.render_text()
        brain = load_package(root).make_policy()
        assert brain.role == role and brain.id == f"lib:t_{role}" and brain.kind == "python"
    src = b'class Hold:\n    role = "defender"\n\n    def act(self, obs):\n        return {"target": -1}\n'
    pkg = install_source(src, "hold.py", tmp_path / "lib")
    assert (pkg.type, pkg.role, pkg.format) == ("policy", "defender", "python")
    weights = NeuralBrain("attacker", hidden=4).save(tmp_path / "a.npz", generations=5)
    pkg = install_npz(weights.read_bytes(), "my wings.npz", tmp_path / "lib")
    assert pkg.name == "my_wings" and pkg.role == "attacker" and pkg.policy_info()["trained"]["generations"] == 5
    with pytest.raises(PackageError):
        install_npz(b"PK not an archive", "x.npz", tmp_path / "lib")


def test_shipped_library_brains():
    ids = {b["id"]: b for b in list_brains()}
    for name, role in (("pincer_attack", "attacker"), ("mesh_defender", "defender"), ("mesh_attacker", "attacker"),
                       ("mesh_recon", "recon")):
        assert ids[f"lib:{name}"]["role"] == role, name
    with pytest.raises(BrainError, match="no brain package"):
        load_brain("lib:nope", "attacker")
    # the trained wall learned to share out targets: it beats the uncoordinated greedy one on the same situations
    cfg = ClashConfig(duration=100.0)
    _, mesh = clash("lib:mesh_defender", "rush", cfg, batch=12)
    _, greedy = clash("greedy", "rush", cfg, batch=12)
    assert mesh["intercepted"].mean() > greedy["intercepted"].mean() and mesh["overkill"].mean() < greedy["overkill"].mean()


def test_policy_export(tmp_path):
    from swarm.sdk.export import export_for_device

    res = export_for_device("examples/brains/mesh_defender", tmp_path / "out")
    assert (tmp_path / "out" / "mesh_defender.swm").is_file() and "option row" in res["instructions"]
    with pytest.raises(PackageError, match="Python arena policy"):
        export_for_device("examples/brains/pincer_attack", tmp_path / "py")

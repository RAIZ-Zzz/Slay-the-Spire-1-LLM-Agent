"""Collect SFT data: MCTS plays, every decision becomes one record.

Settings come from config.json ("collect", plus "fight" and "mcts"). Games are
split by seed, never by step - steps of one game are near copies of each other,
so splitting a game across sets would leak answers into validation and test.

collect.mode "fight": one fight of fight.encounter per seed (seeds seed_start, ...).
collect.mode "act1": act 1 of the run collect.run_seed, once per rule seed (seed_start, ...):
MCTS plays every fight, sts_lightspeed's rules (tools.rule_action) everything else, until
the act 1 boss is beaten or the run dies. Only fight decisions become records.
run_seed null: game s is run s with rule seed s, so every game has its own map (act1_<games>runs).

    python sts1/collect.py

Writes <out_dir>/<encounter or act1_seed<run_seed>>/{train,val,test}.jsonl and meta.json. One record:
    seed, step       which game and which decision in it
    prompt           exactly what llm.py sends the model (build_prompt)
    actions          the legal actions, in order; the answer is a position here
    best             the action MCTS played (most visits)
    visits, values   per action: MCTS visit count and mean value
    game_outcome, game_hp   how that game ended (act1: ACT1_CLEARED or PLAYER_LOSS, and floor)
    forced           only one legal action: nothing to learn (llm.py does not ask the model)
    in_train         val/test only: the same prompt also occurs in train. Jaw Worm with the
                     starter deck has few distinct states (10 cards -> 1260 shuffles, 5 hp
                     values), so most val/test prompts repeat; score unseen ones separately.

Note: about two thirds of answers are position 0, because the simulator lists
actions in its own card-priority order. "Always answer 0" is the baseline to beat.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import random
import time

import llm
import tools
from tools import CONFIG, sts


def decide(game: tools.Game, last: str, seed: int, step: int) -> tuple[dict, tools.Action]:
    """Ask MCTS about the current fight decision; returns its record and the action to play."""
    prompt = llm.build_prompt(last, game)
    actions = tools.legal_actions(game)
    advice = tools.mcts_advice(game, CONFIG["mcts"]["simulations"])
    best = advice[0][0]
    visits, values = [0] * len(actions), [0.0] * len(actions)
    if isinstance(best, int):   # multi-card screens return [] and have no per-action scores
        for a, v, m in advice:
            visits[a], values[a] = v, round(m, 2)
    return {"seed": seed, "step": step, "prompt": prompt, "actions": actions, "best": best,
            "visits": visits, "values": values, "forced": len(actions) == 1}, best


def play_one(seed: int) -> list[dict]:
    """One fight with MCTS choosing every action; returns its records."""
    fight = CONFIG["fight"]
    game = tools.new_battle(fight["encounter"], seed, fight["ascension"])
    records, last = [], "nothing yet"
    while game.outcome == "UNDECIDED":
        record, best = decide(game, last, seed, len(records))
        records.append(record)
        last = tools.act(game, best)
    hp = game.bc.state()["player"]["hp"]
    for r in records:
        r["game_outcome"], r["game_hp"] = game.outcome, hp
    return records


def play_act1(rule_seed: int) -> list[dict]:
    """Act 1 of run collect.run_seed (null: of run rule_seed, a new map every game): MCTS in
    fights, rules (seeded by rule_seed) outside them."""
    rng = random.Random(rule_seed)
    run_seed = CONFIG["collect"]["run_seed"]
    game = tools.new_game(rule_seed if run_seed is None else run_seed, CONFIG["fight"]["ascension"])
    records, last = [], "nothing yet"
    while game.outcome == "UNDECIDED" and sts.game_state(game.gc)["act"] == 1:
        if game.bc is None:
            last = tools.act(game, tools.rule_action(game, rng))
            continue
        record, best = decide(game, last, rule_seed, len(records))
        records.append(record)
        last = tools.act(game, best)
    s = sts.game_state(game.gc)
    for r in records:
        r["game_outcome"] = "ACT1_CLEARED" if s["act"] > 1 else game.outcome
        r["game_hp"], r["game_floor"] = s["hp"], s["floor"]
    return records


def main():
    cfg, fight = CONFIG["collect"], CONFIG["fight"]
    act1 = cfg.get("mode", "fight") == "act1"
    play, name = ((play_act1, f"act1_seed{cfg['run_seed']}" if cfg["run_seed"] is not None else f"act1_{cfg['games']}runs")
                  if act1 else (play_one, fight["encounter"].lower()))
    out = tools.ROOT / cfg["out_dir"] / name
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(cfg["seed_start"], cfg["seed_start"] + cfg["games"]))
    total = sum(cfg["split"])
    n_train = len(seeds) * cfg["split"][0] // total
    n_val = len(seeds) * cfg["split"][1] // total
    splits = {"train": seeds[:n_train], "val": seeds[n_train:n_train + n_val],
              "test": seeds[n_train + n_val:]}

    started = time.time()
    meta = {"encounter": f"act 1 of {name}, rules outside fights" if act1 else fight["encounter"],
            "ascension": fight["ascension"],
            "mcts_simulations": CONFIG["mcts"]["simulations"], "system_prompt": llm.SYSTEM_PROMPT, "splits": {}}
    train_prompts: set[str] = set()
    with multiprocessing.Pool(os.cpu_count()) as pool:
        for name, split_seeds in splits.items():   # train first, so val/test can be checked against it
            games = pool.map(play, split_seeds, chunksize=1 if act1 else 20)
            records = [r for g in games for r in g]
            if name == "train":
                train_prompts = {r["prompt"] for r in records}
            else:
                for r in records:
                    r["in_train"] = r["prompt"] in train_prompts
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            wins = sum(g[0]["game_outcome"] in ("PLAYER_VICTORY", "ACT1_CLEARED") for g in games)
            stats = meta["splits"][name] = {
                "seeds": [split_seeds[0], split_seeds[-1]], "games": len(games), "records": len(records),
                "forced": sum(r["forced"] for r in records),
                "distinct_prompts": len({r["prompt"] for r in records}),
                "in_train": sum(r.get("in_train", False) for r in records),
                "answer_is_0": sum(r["best"] == 0 for r in records if not r["forced"]),
                "win_rate": wins / len(games),
                "mean_hp_left": sum(g[0]["game_hp"] for g in games) / len(games)}
            choices = len(records) - stats["forced"]
            print(f"{name:5s} seeds {split_seeds[0]}-{split_seeds[-1]}: {len(records)} records "
                  f"({stats['forced']} forced), {stats['distinct_prompts']} distinct, "
                  f"{stats['in_train']} also in train | answer 0 on {stats['answer_is_0'] / choices:.0%} of real "
                  f"choices | win {wins / len(games):.0%}, hp left {stats['mean_hp_left']:.1f} "
                  f"({time.time() - started:.0f}s)")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

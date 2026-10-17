"""Collect SFT data: MCTS plays, every decision becomes one record.

Settings come from config.json ("collect", plus "fight" and "mcts"). Games are
split by seed, never by step - steps of one game are near copies of each other,
so splitting a game across sets would leak answers into validation and test.

    python sts1/collect.py

Writes <out_dir>/<encounter>/{train,val,test}.jsonl and meta.json. One record:
    seed, step       which game and which decision in it
    prompt           exactly what llm.py sends the model (build_prompt)
    actions          the legal actions, in order; the answer is a position here
    best             the action MCTS played (most visits)
    visits, values   per action: MCTS visit count and mean value
    game_outcome, game_hp   how that game ended
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
import time

import llm
import tools
from tools import CONFIG


def play_one(seed: int) -> list[dict]:
    """One game with MCTS choosing every action; returns its records."""
    fight = CONFIG["fight"]
    sims = CONFIG["mcts"]["simulations"]
    game = tools.new_battle(fight["encounter"], seed, fight["ascension"])
    records, last = [], "nothing yet"
    while game.outcome == "UNDECIDED":
        prompt = llm.build_prompt(last, game)
        actions = tools.legal_actions(game)
        advice = tools.mcts_advice(game, sims)
        best = advice[0][0]
        visits, values = [0] * len(actions), [0.0] * len(actions)
        if isinstance(best, int):   # multi-card screens return [] and have no per-action scores
            for a, v, m in advice:
                visits[a], values[a] = v, round(m, 2)
        records.append({"seed": seed, "step": len(records), "prompt": prompt, "actions": actions,
                        "best": best, "visits": visits, "values": values, "forced": len(actions) == 1})
        last = tools.act(game, best)
    hp = game.bc.state()["player"]["hp"]
    for r in records:
        r["game_outcome"], r["game_hp"] = game.outcome, hp
    return records


def main():
    cfg, fight = CONFIG["collect"], CONFIG["fight"]
    out = tools.ROOT / cfg["out_dir"] / fight["encounter"].lower()
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(cfg["seed_start"], cfg["seed_start"] + cfg["games"]))
    total = sum(cfg["split"])
    n_train = len(seeds) * cfg["split"][0] // total
    n_val = len(seeds) * cfg["split"][1] // total
    splits = {"train": seeds[:n_train], "val": seeds[n_train:n_train + n_val],
              "test": seeds[n_train + n_val:]}

    started = time.time()
    meta = {"encounter": fight["encounter"], "ascension": fight["ascension"],
            "mcts_simulations": CONFIG["mcts"]["simulations"], "system_prompt": llm.SYSTEM_PROMPT, "splits": {}}
    train_prompts: set[str] = set()
    with multiprocessing.Pool(os.cpu_count()) as pool:
        for name, split_seeds in splits.items():   # train first, so val/test can be checked against it
            games = pool.map(play_one, split_seeds, chunksize=20)
            records = [r for g in games for r in g]
            if name == "train":
                train_prompts = {r["prompt"] for r in records}
            else:
                for r in records:
                    r["in_train"] = r["prompt"] in train_prompts
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            wins = sum(g[0]["game_outcome"] == "PLAYER_VICTORY" for g in games)
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

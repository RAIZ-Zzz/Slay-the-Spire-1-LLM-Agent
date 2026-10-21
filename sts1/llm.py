"""Let a model play: show it render_state, read back an action, act, repeat.

Settings come from config.json ("llm" and "play"). With model "random" it plays
random legal actions and calls nothing, which checks the loop end to end. With an
empty base_url the model runs locally through transformers (a GPU machine such as
TC2), optionally with the LoRA adapter in llm.adapter, e.g. "runs/sft/final".

    python sts1/llm.py

Prints one line per game and writes each game's full transcript to paths.log_dir.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request

# torch has to be imported before the simulator module: on TC2, importing torch after
# slaythespire segfaults inside torch's import, the other order works (tested 10-19).
# ponytail: import-order workaround; the real fix is in how the .so links libstdc++.
# Only when run as a script: collect.py imports this module and then forks workers,
# and on TC2 the forked workers hang (0% CPU) if torch is already loaded (10-20).
if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:   # API or random mode on a machine without torch
        pass

import tools
from tools import CONFIG

LLM, PLAY = CONFIG["llm"], CONFIG["play"]

SYSTEM_PROMPT = (
    "You are playing Slay the Spire as the Ironclad. Each message shows the current "
    "situation and ends with the legal actions. End your reply with one line "
    "'ANSWER: <number>' giving the number of the action you choose. On screens that "
    "ask for a list of hand numbers, end with 'ANSWER: [0, 2]', or 'ANSWER: []' for none."
)


def build_prompt(last: str, game: tools.Game) -> str:
    """The user message for one decision. collect.py uses it too, so training
    prompts match what the model sees when it plays."""
    return f"Last action: {last}\n\n{tools.render_state(game)}"


_local = None   # (tokenizer, model), loaded on the first local call


def ask_local(prompt: str) -> str:
    """One reply from the model in llm.model run with transformers, plus llm.adapter if set.

    Sources: load + apply_chat_template(add_generation_prompt=True) + decode only the new tokens -
    Qwen3-4B-Instruct-2507 model card Quickstart; base model + adapter - TRL PEFT Integration
    "Loading"; greedy when temperature is 0 - HF "Text generation" guide, Pitfalls. Same steps as
    sft.evaluate, so play and the accuracy evaluation see the same prompt.
    """
    global _local
    import torch
    if _local is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLM["model"])
        model = AutoModelForCausalLM.from_pretrained(LLM["model"], dtype=torch.bfloat16, device_map="auto")
        if LLM.get("adapter"):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(tools.ROOT / LLM["adapter"]))
        _local = tokenizer, model.eval()
    tokenizer, model = _local
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)
    sampling = {"do_sample": True, "temperature": LLM["temperature"]} if LLM["temperature"] > 0 else {"do_sample": False}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=LLM["max_tokens"], **sampling)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def ask_model(prompt: str) -> str:
    """One chat completion from the OpenAI-compatible endpoint in config.json."""
    body = json.dumps({
        "model": LLM["model"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "temperature": LLM["temperature"],
        "max_tokens": LLM["max_tokens"],
        **LLM.get("extra_body", {}),   # provider-specific fields, e.g. turning thinking off
    }).encode()
    # Cloudflare in front of some relays rejects Python's default User-Agent (error 1010)
    headers = {"Content-Type": "application/json", "User-Agent": "sts1-llm/1.0"}
    key = LLM.get("api_key") or os.environ.get(LLM.get("api_key_env", ""))
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(LLM["base_url"].rstrip("/") + "/chat/completions", body, headers)
    for wait in (2, 8, 30, None):   # rate limits and server errors are often transient
        try:
            with urllib.request.urlopen(req, timeout=LLM["timeout"]) as resp:
                choice = json.load(resp)["choices"][0]
            content = choice["message"].get("content") or ""
            if not content.strip():   # e.g. a thinking model spent max_tokens before answering
                return f"(empty reply, finish_reason={choice.get('finish_reason')})"
            return content
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace").strip()
            if wait is None or not (e.code == 429 or e.code >= 500):
                raise SystemExit(f"model endpoint answered HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            if wait is None:
                raise SystemExit(f"cannot reach {LLM['base_url']}: {e.reason}") from None
        time.sleep(wait)


_VALUE = r"(\[[\d,\s]*\]|-?\d+)"


def parse_action(reply: str) -> tools.Action | None:
    """The action in a reply: the last 'ANSWER: ...' line, or a reply that is only a
    number or a list. Numbers elsewhere in the reply are ignored - analysis text is
    full of them ("11 damage"), and action labels contain "[0]" as a target."""
    answers = re.findall(r"ANSWER:\s*" + _VALUE, reply, re.I)
    if answers:
        value = answers[-1]
    elif re.fullmatch(r"\s*" + _VALUE + r"\s*", reply):
        value = reply.strip()
    else:
        return None
    return json.loads(value) if value.startswith("[") else int(value)


def choose(game: tools.Game, prompt: str, rng: random.Random) -> tuple[tools.Action | None, str]:
    """(action, raw reply) from the configured model."""
    if LLM["model"] == "random":
        state = game.bc.state() if game.bc is not None else None
        if state is not None and tools._is_multi(state):
            action = rng.sample(range(len(state["hand"])), rng.randint(0, tools._multi_limit(state)))
        else:
            action = rng.randrange(len(tools.legal_actions(game)))
        return action, json.dumps(action)
    reply = ask_model(prompt) if LLM["base_url"] else ask_local(prompt)
    return parse_action(reply), reply


def play(game: tools.Game, log, rng: random.Random, act1: bool = False) -> dict:
    """Play one game to the end. Returns counts for the summary line.

    act1: the model plays only the fights; sts_lightspeed's rules (tools.rule_action, drawing
    from rng) take every step outside them, forced ones included, exactly as collect.py does,
    so the same rule seed gives the same run while the model plays like MCTS. Stops when act 1
    is over. A game ends early as STUCK when the model gives no legal action after all
    retries, or as TOO_LONG at play.max_steps; the next game still runs.
    """
    steps = invalid = 0
    last = "nothing yet"
    while game.outcome == "UNDECIDED" and not (act1 and tools.sts.game_state(game.gc)["act"] > 1):
        if steps >= PLAY["max_steps"]:
            return {"steps": steps, "invalid": invalid, "stopped": "TOO_LONG"}
        if act1 and game.bc is None:
            last = tools.act(game, tools.rule_action(game, rng))
            log.write(json.dumps({"step": steps, "rules": True, "result": last}) + "\n")
            steps += 1
            continue
        if game.bc is None or not tools._is_multi(game.bc.state()):
            if len(tools.legal_actions(game)) == 1:   # forced move: nothing to ask
                last = tools.act(game, 0)
                log.write(json.dumps({"step": steps, "forced": True, "result": last}) + "\n")
                steps += 1
                continue
        prompt = build_prompt(last, game)
        for attempt in range(1 + LLM["retries"]):
            action, reply = choose(game, prompt, rng)
            try:
                if action is None:
                    raise ValueError("no action number found in the reply")
                last = tools.act(game, action)
                break
            except ValueError as e:
                invalid += 1
                log.write(json.dumps({"step": steps, "reply": reply, "error": str(e)}) + "\n")
                prompt = f"Your answer {reply!r} was not accepted: {e}\n\n{tools.render_state(game)}"
        else:
            return {"steps": steps, "invalid": invalid, "stopped": "STUCK"}
        log.write(json.dumps({"step": steps, "reply": reply, "result": last}) + "\n")
        steps += 1
    return {"steps": steps, "invalid": invalid, "stopped": None}


def main():
    fight = CONFIG["fight"]
    log_dir = tools.ROOT / CONFIG["paths"]["log_dir"]
    log_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    act1 = PLAY["mode"] == "act1"   # seeds below are then rule seeds of run play.run_seed
    print(f"model {LLM['model']} | {PLAY['mode']} | {PLAY['games']} games from seed {fight['seed']}"
          + ((f" (rule seeds of run {PLAY['run_seed']})" if PLAY.get("run_seed") is not None
              else " (run s with rule seed s)") if act1 else ""))
    hp_left, wins, invalid = [], 0, 0   # single fights: HP left, 0 unless won (lost, stuck, too long)
    floors, cleared = [], 0             # act1: floor reached, act 1 boss beaten
    for i in range(PLAY["games"]):
        seed = fight["seed"] + i
        if act1:   # run_seed null: run `seed` with rule seed `seed`, as collect.py does
            game = tools.new_game(seed if PLAY.get("run_seed") is None else PLAY["run_seed"], fight["ascension"])
        elif PLAY["mode"] == "game":
            game = tools.new_game(seed, fight["ascension"])
        else:
            game = tools.new_battle(fight["encounter"], seed, fight["ascension"])
        path = log_dir / f"{stamp}-{PLAY['mode']}-seed{seed}.jsonl"
        started = time.time()
        with open(path, "w", encoding="utf-8") as log:
            counts = play(game, log, random.Random(seed), act1)
        outcome = counts["stopped"] or game.outcome
        if act1:
            s = tools.sts.game_state(game.gc)
            floors.append(s["floor"])
            cleared += s["act"] > 1
            outcome = "ACT1_CLEARED" if s["act"] > 1 else outcome
        if game.bc is not None and game.single_fight:   # a single fight: report the fight
            where = f"hp {game.bc.state()['player']['hp']}/{game.bc.state()['player']['max_hp']}"
        else:
            s = tools.sts.game_state(game.gc)
            where = f"act {s['act']} floor {s['floor']} hp {s['hp']}/{s['max_hp']}"
        print(f"seed {seed}: {outcome:15s} {where} | {counts['steps']} actions, "
              f"{counts['invalid']} invalid answers | {time.time() - started:.1f}s | {path.name}", flush=True)
        won = game.outcome == "PLAYER_VICTORY"
        wins += won
        invalid += counts["invalid"]
        if game.bc is not None and game.single_fight:
            hp_left.append(game.bc.state()["player"]["hp"] if won else 0)
    if floors:
        print(f"summary: {len(floors)} act-1 runs, act 1 boss beaten {cleared}/{len(floors)}, "
              f"mean floor reached {sum(floors) / len(floors):.1f}, {invalid} invalid answers")
    if hp_left:
        print(f"summary: {len(hp_left)} fights, win rate {wins / len(hp_left):.1%}, "
              f"mean HP left {sum(hp_left) / len(hp_left):.2f} (0 unless won), {invalid} invalid answers")


if __name__ == "__main__":
    main()

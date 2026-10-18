# STS1: an LLM agent that plays Slay the Spire

Goal: train a small local LLM to play Slay the Spire (Ironclad), using a fast
simulator as the training ground. Learning project.

Earlier work on Slay the Spire 2 (LLM through a game mod, then a DQN):
[RAIZ-Zzz/Slay-the-Spire-RL-model](https://github.com/RAIZ-Zzz/Slay-the-Spire-RL-model)

## How it started

1. **Simulator.** Writing a simulator was not the point, so we use
   [sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) (C++, RNG-accurate,
   Ironclad only, ~100k games in 1.4 s on 20 threads here). Its Python bindings only
   ran whole games as a black box, so we added step-by-step bindings for fights
   (`bindings/battle.cpp`) and for every out-of-combat screen (`bindings/game.cpp`).
2. **Two Tools.** Every decision in the game - play a card, take a reward, pick a
   path, an event option, buy, rest - is "pick #i from a numbered list"; three
   potion/relic screens pick several hand cards instead. The real game's
   CommunicationMod works the same way, so the interface carries over.
3. **No card text.** The model reads card names and numbers only, and learns what
   cards do by playing.
4. **Tools first.** `tools.py` is the hands and eyes; the LLM side is written separately.

## Status

- Tools cover a whole run; tests cover every potion, relic, event and act, plus
  thousands of random fights and runs (`simulate` must predict `act` exactly).
- SFT data: 10,000 Jaw Worm fights played by MCTS -> 85k labelled decisions, split by seed.
- Next: SFT a small open model on that data, then RL with the simulator's reward.

## Layout

```
build.ps1            build the simulator + Python module (Windows)
build.sh             the same on Linux (e.g. a GPU cluster)
sts_lightspeed/      the simulator (vendored, see below)
sts1/config.json     all settings (not in git; copy config.example.json to start)
sts1/tools.py        render_state / legal_actions / act / simulate / mcts_advice
sts1/test_tools.py   tests: every potion, relic, event, act, plus random fuzzing
sts1/collect.py      MCTS plays, every decision becomes an SFT record (data/)
sts1/llm.py          a model plays through the tools; logs in runs/
```

## Use (Windows)

Needs Python 3.14 (the compiled module only loads in the Python it was built with)
and a GCC: unzip a [WinLibs](https://winlibs.com/) x86_64 UCRT build into `tools/mingw64`.

```
powershell ./build.ps1          # once, or after changing C++
python sts1/tools.py            # print the opening of a run
python sts1/test_tools.py quick # tests (drop "quick" for the full ~13 min run)
python sts1/collect.py          # generate SFT data
python sts1/llm.py              # let the model in config.json play; "random" = dry run
```

## Use (Linux)

Needs cmake >= 3.19 and g++ with C++17. Same commands after the build:

```
PY=$(which python) ./build.sh   # or "./build.sh debug" for a UBSan build in build-debug/
cp sts1/config.example.json sts1/config.json
python sts1/test_tools.py quick
```

## The vendored simulator

`sts_lightspeed/` is [gamerpuppy/sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed)
at commit `7476a81` (MIT, see its LICENSE), with `json` (nlohmann/json) and `pybind11`
(upgraded to v3.1.0, the original v2.7.1 cannot build for Python >= 3.11) included.
Our changes:

- `bindings/battle.cpp`, `bindings/game.cpp`: new step-by-step bindings
- `src/game/GameContext.cpp`: five lambdas captured `this`, so a copied run wrote into the original
- `src/combat/BattleContext.cpp`: `player.cc` (character class) was never set, so in-combat
  random cards/potions indexed tables with garbage; harmless by luck on Windows, hangs on Linux
- `CMakeLists.txt`, `bindings/slaythespire.cpp`: register the new bindings

Simulator issues worked around in the bindings rather than patched: potion actions
outside fights also ran the screen action, Singing Bowl was never offered, Smoke Bomb
does nothing, the card colour table is out of step with the card ids, relic ids >= 128
alias in fights. Details are in comments next to each workaround.

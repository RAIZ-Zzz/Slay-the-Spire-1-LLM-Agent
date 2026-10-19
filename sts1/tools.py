"""Tool functions: everything a player (or an LLM) can see and do in Slay the Spire.

Plain Python over the sts_lightspeed simulator. No prompt, model or training code
lives here - that is the LLM side, written separately.

    game = new_game()              # a whole run, from Neow
    game = new_battle("JAW_WORM")  # or a single fight
    print(render_state(game))      # what the model reads, ending with numbered actions
    print(act(game, 1))            # do action #1 for real
    print(simulate(game, 1))       # try it on a copy instead; game is unchanged

There are two hands, and every screen of the game uses one of them:
  * act(game, i)       - pick action #i from legal_actions(game). Playing a card,
                         ending the turn, taking a reward, choosing a path, an
                         event option, buying, resting, removing a card: all this.
  * act(game, [i, j])  - pick several hand cards, only on the three multi-card
                         screens (Elixir, Gambler's Brew, Gambling Chip); [] = none.

The same two hands fit the real game: CommunicationMod offers a numbered choice
list on every screen too. Only the Game class below talks to the simulator, so a
real-game backend is another class with the same methods.

God mode: fights show the draw pile in true draw order. Cards appear by name
only - no rules text - so the model learns what each card does by playing.

    python sts1/test_tools.py      # tests
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # the folder that holds sts1/
_CONFIG_FILE = Path(__file__).with_name("config.json")  # not in git; config.example.json is the template
if not _CONFIG_FILE.exists():
    raise SystemExit(f"{_CONFIG_FILE} is missing: copy config.example.json to config.json and fill it in")
CONFIG = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))

sys.path.insert(0, str(ROOT / CONFIG["paths"]["sim_build_dir"]))
import slaythespire as sts  # noqa: E402

Action = int | list[int]


# --- the game: the only part that knows the simulator --------------------------

class Game:
    """One run, or one fight, in the simulator.

    gc is the run; bc is the current fight (None between fights). A single-fight
    game keeps bc after the fight ends, so its outcome is the fight's outcome.
    """

    def __init__(self, gc, bc=None, single_fight: bool = False):
        self.gc, self.bc, self.single_fight = gc, bc, single_fight

    @property
    def outcome(self) -> str:
        if self.bc is not None:
            return self.bc.outcome
        return self.gc.outcome.name

    def copy(self) -> Game:
        return Game(sts.copy_game(self.gc), self.bc.copy() if self.bc else None, self.single_fight)


def new_game(seed: int = CONFIG["fight"]["seed"], ascension: int = CONFIG["fight"]["ascension"]) -> Game:
    """A whole Ironclad run, starting at Neow."""
    return Game(sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension))


def new_battle(encounter: str = CONFIG["fight"]["encounter"], seed: int = CONFIG["fight"]["seed"],
               ascension: int = CONFIG["fight"]["ascension"], potions: tuple[str, ...] = (),
               relics: tuple[str, ...] = (), extra_cards: tuple[str, ...] = ()) -> Game:
    """A single fight: Ironclad starter deck and relic, plus anything listed, against `encounter`.

    Names are the simulator's enum names: potions "FIRE_POTION", relics "VAJRA",
    cards "INFLAME", encounters "GREMLIN_NOB". Only content an Ironclad can get
    is accepted: the simulator implements no other character, and unimplemented
    content aborts the whole process.
    """
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
    for name in extra_cards:
        card = _enum(sts.CardId, name)
        if name not in _IRONCLAD_CARDS:   # also keeps out status cards: Deck.cpp:178 asserts on them
            raise ValueError(f"{name} is not a card an Ironclad deck can hold")
        gc.obtain_card(sts.Card(card))
    for name in relics:  # before potions: Potion Belt adds slots, Sozu blocks them
        if name not in _IRONCLAD_RELICS:
            raise ValueError(f"{name} is not a relic an Ironclad can get")
        sts.obtain_relic(gc, _enum(sts.RelicId, name))
    for name in potions:
        if name not in sts.ironclad_potions():
            raise ValueError(f"{name} is not a potion an Ironclad can get")
        if not sts.obtain_potion(gc, name):
            raise ValueError(f"no free potion slot for {name}")
    return Game(gc, sts.BattleContext(gc, _enum(sts.MonsterEncounter, encounter)), single_fight=True)


# --- tools: observation --------------------------------------------------------

def legal_actions(game: Game) -> list[str]:
    """What can be done right now, as text.

    Usually the list position is the action number. On multi-card screens the
    list is your hand, and you answer with a list of positions - with an empty
    hand that list is empty and [] is the only answer. Use game.outcome, not an
    empty list, to tell whether the game is over.
    """
    if game.outcome != "UNDECIDED":
        return []
    if game.bc is not None:
        state = game.bc.state()
        if _is_multi(state):
            return [_card(c) for c in state["hand"]]
        return [label for _, label in _combat_options(game.bc, state)]
    return [label for _, label in _run_options(game, sts.game_state(game.gc))]


def render_state(game: Game) -> str:
    """The whole situation as text, ending with what can be done now."""
    if game.bc is not None:
        return _render_combat(game)
    return _render_run(game)


# --- tools: action -------------------------------------------------------------

def act(game: Game, action: Action) -> str:
    """Do `action` for real and report what changed. Raises ValueError if it is not legal."""
    if game.outcome != "UNDECIDED":
        raise ValueError(f"the game is over: {game.outcome}")

    if game.bc is not None:
        before = game.bc.state()
        label = _combat_take(game.bc, action, before)
        line = f"{label}: {_combat_changes(before, game.bc.state())}"
        if game.bc.outcome != "UNDECIDED" and not game.single_fight:
            game.bc.exit_battle(game.gc)
            game.bc = None
            line += f" | fight over, {_screen_title(sts.game_state(game.gc))}"
        return line

    before = sts.game_state(game.gc)
    label = _run_take(game, action, before)
    after = sts.game_state(game.gc)
    line = f"{label}: {_run_changes(before, after)}"
    if after["screen"] == "BATTLE":
        game.bc = sts.BattleContext(game.gc)
        line += f" | fight starts: {game.bc.encounter}"
    return line


def simulate(game: Game, action: Action) -> str:
    """Try `action` on a copy and show the result; game is untouched.

    Returns what changed plus the full situation after that one step, so the
    model sees exactly what act(game, action) would lead to.
    """
    sim = game.copy()
    return f"If you {act(sim, action)}\n\n{render_state(sim)}"


def mcts_advice(game: Game, simulations: int = CONFIG["mcts"]["simulations"]) -> list[tuple[Action, int, float]]:
    """Ask the built-in MCTS about the current fight. Returns (action, visits, mean value), most visited first.

    Teacher signal for imitation and a baseline to compare against; not meant to
    be offered to the model as a tool. Fights only. On multi-card screens MCTS
    only ever considers picking nothing (BattleScumSearcher2.cpp:377), so it returns [].
    """
    if game.bc is None or game.outcome != "UNDECIDED":
        raise ValueError("MCTS only advises inside an ongoing fight")
    state = game.bc.state()
    if _is_multi(state):
        return [([], 1, 0.0)]
    position = {bits: i for i, (bits, _) in enumerate(_combat_options(game.bc, state))}
    rows = [(position[r["action"]], r["visits"], r["mean_value"])
            for r in game.bc.mcts(simulations) if r["action"] in position]
    return sorted(rows, key=lambda r: -r[1])


def rule_action(game: Game, rng) -> int:
    """The action number sts_lightspeed's own out-of-combat rules pick here
    (ScumSearchAgent2::stepOutOfCombatPolicy: card rewards by ExpertKnowledge weights,
    boss relics by a fixed ordering, rest site by HP, random map path). Outside fights only.
    Like mcts_advice, a stand-in for decisions the model does not make yet, not a model tool.
    """
    if game.bc is not None or game.outcome != "UNDECIDED":
        raise ValueError("the rules only decide outside fights in an ongoing run")
    bits = sts.rule_action(game.gc, rng.getrandbits(32))
    for i, (b, _) in enumerate(_run_options(game, sts.game_state(game.gc))):
        if b == bits:
            return i
    raise RuntimeError(f"rule action {bits:#x} is not among the legal actions")


# --- helpers: shared -----------------------------------------------------------

_IRONCLAD_CARDS = set(sts.ironclad_cards())
_IRONCLAD_RELICS = (set(sts.class_relics(sts.CharacterClass.IRONCLAD))
                    | (set(sts.RelicId.__members__) - {"INVALID"}
                       - set().union(*(sts.class_relics(c) for c in (sts.CharacterClass.SILENT,
                                                                       sts.CharacterClass.DEFECT,
                                                                       sts.CharacterClass.WATCHER)))))


def _enum(enum, name: str):
    try:
        return getattr(enum, name)
    except AttributeError:
        raise ValueError(f"unknown {enum.__name__} name: {name}") from None


def _card(c: dict) -> str:
    name = c["name"] + "+" * min(c["upgrades"], 1) + (str(c["upgrades"]) if c["upgrades"] > 1 else "")
    return f"{name} (cost {c['cost']}, {c['type']})"


def _names(cards: list[dict]) -> str:
    return ", ".join(c["name"] + ("+" if c["upgrades"] else "") for c in cards) or "empty"


def _counted(cards: list[dict]) -> str:
    counts = collections.Counter(c["name"] + ("+" if c["upgrades"] else "") for c in cards)
    return ", ".join(f"{n} x{k}" if k > 1 else n for n, k in counts.items()) or "empty"


def _potion(name: str) -> str:
    return "empty" if name == "EMPTY_POTION_SLOT" else name


def _diff_list(label: str, before: list[str], after: list[str]) -> list[str]:
    b, a = collections.Counter(before), collections.Counter(after)
    out = [f"{label} +{n}" for n in (a - b).elements()]
    return out + [f"{label} -{n}" for n in (b - a).elements()]


# --- helpers: combat -----------------------------------------------------------

# Which pile a single card select picks from (CardSelectInfo.h tasks).
_SELECT_PILE = {
    **dict.fromkeys(("ARMAMENTS", "DUAL_WIELD", "EXHAUST_ONE", "FORETHOUGHT", "RECYCLE",
                     "SETUP", "WARCRY", "NIGHTMARE"), "hand"),
    **dict.fromkeys(("HEADBUTT", "HOLOGRAM", "LIQUID_MEMORIES_POTION", "MEDITATE"), "discard_pile"),
    "EXHUME": "exhaust_pile",
    **dict.fromkeys(("SECRET_TECHNIQUE", "SECRET_WEAPON", "SEEK"), "draw_pile"),
}
_MULTI_TASKS = ("EXHAUST_MANY", "GAMBLE")


def _is_multi(state: dict) -> bool:
    return state.get("select", {}).get("task") in _MULTI_TASKS


def _multi_limit(state: dict) -> int:
    hand = len(state["hand"])
    if state["select"]["task"] == "EXHAUST_MANY":
        return min(hand, state["select"]["pick_count"])
    return hand


def _render_combat(game: Game) -> str:
    bc = game.bc
    s = bc.state()
    p = s["player"]
    run = sts.game_state(game.gc)
    lines = [f"Act {run['act']}, floor {run['floor']} | fight: {bc.encounter} | turn {s['turn'] + 1} "
             f"| {s['outcome']}",   # the engine counts turns from 0
             f"You: HP {p['hp']}/{p['max_hp']}, Block {p['block']}, Energy {p['energy']}, "
             f"Statuses: {_statuses(p['statuses'])}",
             f"Relics: {', '.join(p['relics']) or 'none'}",
             f"Potions: {', '.join(_potion(x) for x in s['potions'])}",
             "Enemies:"]
    for m in s["monsters"]:
        if m["alive"]:
            lines.append(f"  [{m['idx']}] {m['name']}: HP {m['hp']}/{m['max_hp']}, Block {m['block']}, "
                         f"Intent: {_intent(m)}, Statuses: {_statuses(m['statuses'])}")
    lines.append("Hand:")
    lines += [f"  {i}. {_card(c)}" for i, c in enumerate(s["hand"])]
    lines.append(f"Draw pile (next draw first): {_names(s['draw_pile'])}")
    lines.append(f"Discard pile: {_names(s['discard_pile'])}")
    lines.append(f"Exhaust pile: {_names(s['exhaust_pile'])}")

    if s["outcome"] != "UNDECIDED":
        return "\n".join(lines)
    if _is_multi(s):
        lines.append(f"{s['select']['task']}: pick up to {_multi_limit(s)} hand cards - answer with a "
                     f"list of hand numbers, [] for none")
        return "\n".join(lines)
    if "select" in s:
        lines.append(f"Choosing a card for: {s['select']['task']}")
    lines.append("Legal actions:")
    lines += [f"  {i}. {d}" for i, d in enumerate(legal_actions(game))]
    return "\n".join(lines)


def _combat_options(bc, state: dict) -> list[tuple[int, str]]:
    """(action int, label) for every legal action on a normal or single-select fight screen."""
    out = []
    for bits in bc.legal_actions():
        a = sts.decode_action(bits)
        # Smoke Bomb is still a todo in the simulator (BattleContext.cpp:2398):
        # drinking it does nothing, so discarding it is offered instead (the
        # engine itself only lists discards for potions that cannot be drunk).
        if a["type"] == "POTION" and a["target"] <= 5 and state["potions"][a["source"]] == "Smoke Bomb":
            bits = sts.discard_potion_action(a["source"])
            a = sts.decode_action(bits)
            if any(b == bits for b, _ in out):
                continue
        out.append((bits, _combat_label(bc, state, a)))
    return out


def _combat_take(bc, action: Action, state: dict) -> str:
    """Execute `action` in the fight and return its label. Raises ValueError if it is not legal."""
    if _is_multi(state):
        if not isinstance(action, list):
            raise ValueError("this screen picks several cards: answer with a list of hand numbers, "
                             "e.g. [0, 2], or [] for none")
        hand, limit = state["hand"], _multi_limit(state)
        if any(type(i) is not int for i in action):
            raise ValueError(f"hand numbers must be integers, got {action}")
        if len(set(action)) != len(action):
            raise ValueError(f"a card can be picked only once, got {action}")
        if any(not 0 <= i < len(hand) for i in action):
            raise ValueError(f"hand numbers are 0..{len(hand) - 1}, got {action}")
        if len(action) > limit:
            raise ValueError(f"pick at most {limit} cards, got {len(action)}")
        picked = ", ".join(_card(hand[i]) for i in sorted(action)) or "nothing"
        bc.step(sts.multi_select_action(sorted(action)))
        return f"{state['select']['task']}: pick {picked}"

    bits, label = _pick(_combat_options(bc, state), action)
    bc.step(bits)
    return label


def _pick(options: list[tuple[int, str]], action: Action) -> tuple[int, str]:
    """The option chosen by a single action number. Raises ValueError otherwise."""
    if type(action) is not int:
        raise ValueError(f"answer with one action number, got {action!r}")
    if not 0 <= action < len(options):
        raise ValueError(f"action {action} does not exist; legal actions are 0..{len(options) - 1}")
    return options[action]


def _combat_label(bc, state: dict, a: dict) -> str:
    if a["type"] == "END_TURN":
        return "end turn"
    if a["type"] == "CARD":
        card = state["hand"][a["source"]]
        target = f" -> {_monster(state, a['target'])}" if card["needs_target"] else ""
        return f"play {_card(card)}{target}"
    if a["type"] == "POTION":
        potion = _potion(state["potions"][a["source"]])
        if a["target"] > 5:
            return f"discard {potion}"
        targets = state["potions"][a["source"]] in ("Fear Potion", "Fire Potion", "Poison Potion",
                                                    "Weak Potion")  # Potions.h:263
        return f"drink {potion}" + (f" -> {_monster(state, a['target'])}" if targets else "")

    # single card select
    select = state["select"]
    task, idx = select["task"], a["source"]
    if task in ("DISCOVERY", "CODEX"):
        if task == "CODEX" and idx == 3:
            return "CODEX: skip"
        return f"{task}: take {select['generated'][idx]}"
    pile = _SELECT_PILE.get(task)
    if pile is None:  # not reachable for Ironclad cards and potions
        return f"{task}: choose option {idx}"
    cards = state[pile]
    if pile == "draw_pile":  # the engine indexes the draw pile bottom-first; state() is top-first
        idx = len(cards) - 1 - idx
    return f"{task}: choose {_card(cards[idx])}"


def _combat_changes(before: dict, after: dict) -> str:
    """One line of what differs between two fight states."""
    out = []
    pb, pa = before["player"], after["player"]
    for key, label in (("hp", "your HP"), ("max_hp", "your max HP"), ("block", "your Block"),
                       ("energy", "Energy")):
        if pb[key] != pa[key]:
            out.append(f"{label} {pb[key]}->{pa[key]}")
    out += _status_changes("you", pb["statuses"], pa["statuses"])
    for mb, ma in zip(before["monsters"], after["monsters"]):
        if mb["alive"] and not ma["alive"]:
            out.append(f"{mb['name']} died")
            continue
        out += [f"{mb['name']} {label} {mb[key]}->{ma[key]}"
                for key, label in (("hp", "HP"), ("block", "Block")) if mb[key] != ma[key]]
        out += _status_changes(mb["name"], mb["statuses"], ma["statuses"])
    for ma in after["monsters"][len(before["monsters"]):]:
        out.append(f"{ma['name']} appeared")
    if len(before["hand"]) != len(after["hand"]):
        out.append(f"hand {len(before['hand'])}->{len(after['hand'])} cards")
    if before["potions"] != after["potions"]:
        out.append(f"potions now {', '.join(_potion(x) for x in after['potions'])}")
    if before["turn"] != after["turn"]:
        out.append(f"now turn {after['turn'] + 1}")
    if after["outcome"] != "UNDECIDED":
        out.append(after["outcome"])
    return ", ".join(out) or "no visible change"


def _status_changes(who: str, before: dict, after: dict) -> list[str]:
    return [f"{who} {k} {before.get(k, 0)}->{after.get(k, 0)}"
            for k in sorted(set(before) | set(after)) if before.get(k, 0) != after.get(k, 0)]


def _monster(state: dict, idx: int) -> str:
    return f"[{idx}] {state['monsters'][idx]['name']}"


def _intent(m: dict) -> str:
    if m["attacking"]:
        hits = f" x{m['hits']}" if m["hits"] > 1 else ""
        return f"attack {m['damage_per_hit']}{hits} ({m['move']})"
    return f"no attack ({m['move']})"


def _statuses(statuses: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in statuses.items()) or "none"


# --- helpers: run (everything outside fights) -----------------------------------

_REST = {0: "rest (heal 30% of max HP)", 1: "smith (upgrade a card)", 2: "recall (take the ruby key)",
         3: "lift (Girya: +1 Strength)", 4: "toke (Peace Pipe: remove a card)",
         5: "dig (Shovel: get a relic)", 6: "skip (nothing to do)"}
_SELECT_VERB = {"TRANSFORM": "transform", "TRANSFORM_UPGRADE": "transform and upgrade", "UPGRADE": "upgrade",
                "REMOVE": "remove", "DUPLICATE": "duplicate", "OBTAIN": "obtain", "BOTTLE": "bottle",
                "BONFIRE_SPIRITS": "offer"}


def _screen_title(s: dict) -> str:
    info = s["screen_info"]
    if s["outcome"] != "UNDECIDED":
        return s["outcome"]
    if s["screen"] == "EVENT":
        return f"event: {info['event']}"
    if s["screen"] == "CARD_SELECT":
        return f"choose a card to {_SELECT_VERB[info['type']]} ({info['picked']}/{info['pick_count']})"
    return s["screen"].lower().replace("_", " ") + " screen"


def _render_run(game: Game) -> str:
    s = sts.game_state(game.gc)
    keys = [k for k, have in zip(("ruby", "emerald", "sapphire"), s["keys"]) if have]
    lines = [f"Act {s['act']}, floor {s['floor']} | {_screen_title(s)}",
             f"You: HP {s['hp']}/{s['max_hp']}, Gold {s['gold']}, Keys: {', '.join(keys) or 'none'}",
             f"Deck ({len(s['deck'])}): {_counted(s['deck'])}",
             f"Relics: {', '.join(name + (f' ({n})' if n else '') for name, n in s['relics'])}",
             f"Potions: {', '.join(_potion(x) for x in s['potions'])}",
             f"Act boss: {s['boss']}"]
    if s["outcome"] != "UNDECIDED":
        return "\n".join(lines)
    if s["screen"] == "MAP":
        lines += _render_map(s)
    if s["screen"] == "SHOP":
        lines += _render_shop(s["screen_info"])
    lines.append("Legal actions:")
    lines += [f"  {i}. {d}" for i, d in enumerate(legal_actions(game))]
    return "\n".join(lines)


def _render_map(s: dict) -> list[str]:
    """The rows ahead, keeping only rooms still reachable from where you stand."""
    info = s["screen_info"]
    rows = info["rows"]
    if info["y"] == 14:
        return ["Map: next is the boss"]
    reachable = ({n["x"] for n in rows[0]} if info["y"] == -1
                 else set(next(n["next_x"] for n in rows[info["y"]] if n["x"] == info["x"])))
    lines = ["Map ahead (row: column ROOM -> next columns):"]
    for y in range(info["y"] + 1, 15):
        nodes = [n for n in rows[y] if n["x"] in reachable]
        lines.append(f"  row {y + 1}: " + " | ".join(
            f"{n['x']} {n['room']}{' (burning elite)' if n['burning_elite'] else ''}"
            + (f" -> {','.join(map(str, n['next_x']))}" if y < 14 else "") for n in nodes))
        reachable = {x for n in nodes for x in n["next_x"]}
    lines.append("  then: the boss")
    return lines


def _render_shop(info: dict) -> list[str]:
    items = [f"{_card(c)} {c['price']}g" for c in info["cards"] if c["price"] != -1]
    items += [f"relic {name} {price}g" for name, price in info["relics"] if price != -1]
    items += [f"potion {name} {price}g" for name, price in info["potions"] if price != -1]
    if info["remove_cost"] != -1:
        items.append(f"card removal {info['remove_cost']}g")
    return ["For sale: " + "; ".join(items)]


def _run_options(game: Game, s: dict) -> list[tuple[int, str]]:
    """(action int, label) for every legal action outside fights."""
    return [(bits, _run_label(s, sts.decode_game_action(bits))) for bits in sts.game_actions(game.gc)]


def _run_take(game: Game, action: Action, s: dict) -> str:
    bits, label = _pick(_run_options(game, s), action)
    sts.game_step(game.gc, bits)
    return label


def _run_label(s: dict, a: dict) -> str:
    info, screen, i = s["screen_info"], s["screen"], a["idx1"]
    if a["potion"]:
        return f"{'discard' if a['discard'] else 'drink'} {_potion(s['potions'][i])}"
    if screen == "EVENT":
        return _event_option(info["text"], i)
    if screen == "REWARDS":
        return _reward_label(s, a)
    if screen == "BOSS_RELIC":
        return "skip the boss relic" if i == 3 else f"take boss relic {info['relics'][i]}"
    if screen == "CARD_SELECT":
        return f"{_SELECT_VERB[info['type']]} {_card(info['cards'][i])}"
    if screen == "MAP":
        if info["y"] == 14:
            return f"go to the boss: {s['boss']}"
        row = info["y"] + 1
        node = next(n for n in info["rows"][row] if n["x"] == i)
        burning = " (burning elite: holds the emerald key)" if node["burning_elite"] else ""
        return f"go to {node['room']} (row {row + 1}, column {i}){burning}"
    if screen == "TREASURE":
        return f"open the {info['chest'].lower()} chest" if i == 0 else "leave the chest"
    if screen == "REST":
        return _REST[i]
    if screen == "SHOP":
        return _shop_label(s, a)
    raise AssertionError(f"no label for screen {screen}")


def _event_option(text: str, idx: int) -> str:
    """The console simulator's line for option idx, e.g. "[Sacrifice] Gain 5 Max HP. Lose 7 HP."."""
    for m in re.finditer(r"^(\d+):\s*(.*)$", text, re.M):
        if int(m.group(1)) == idx:
            return re.sub(r"\s+", " ", m.group(2)).strip()
    return f"event option {idx}"


def _no_free_slot(s: dict) -> bool:
    return "EMPTY_POTION_SLOT" not in s["potions"] or any(name == "Sozu" for name, _ in s["relics"])


def _reward_label(s: dict, a: dict) -> str:
    info, i = s["screen_info"], a["idx1"]
    if a["type"] == "GOLD":
        return f"take {info['gold'][i]} gold"
    if a["type"] == "CARD":
        which = f" (card reward {i + 1})" if len(info["cards"]) > 1 else ""
        if a["idx2"] == 5:
            if any(name == "Singing Bowl" for name, _ in s["relics"]):
                return f"skip the card{which}: +2 max HP (Singing Bowl)"
            return f"skip the card{which}"
        return f"take card {_card(info['cards'][i][a['idx2']])}{which}"
    if a["type"] == "RELIC":
        lose_key = info["sapphire_key"] and i == len(info["relics"]) - 1
        return f"take relic {info['relics'][i]}" + (" (gives up the sapphire key)" if lose_key else "")
    if a["type"] == "KEY":
        if info["sapphire_key"]:
            lose = f" (gives up relic {info['relics'][-1]})" if info["relics"] else ""
            return f"take the sapphire key{lose}"
        return "take the emerald key"
    if a["type"] == "POTION":
        lost = " (no free slot: it would be lost)" if _no_free_slot(s) else ""
        return f"take potion {info['potions'][i]}{lost}"
    left = info["gold"] or info["cards"] or info["relics"] or info["potions"] \
        or info["emerald_key"] or info["sapphire_key"]
    return "leave the rewards" + (" (skips what is left)" if left else "")


def _shop_label(s: dict, a: dict) -> str:
    info, i = s["screen_info"], a["idx1"]
    if a["type"] == "CARD":
        card = info["cards"][i]
        return f"buy {_card(card)} for {card['price']} gold"
    if a["type"] == "RELIC":
        name, price = info["relics"][i]
        return f"buy relic {name} for {price} gold"
    if a["type"] == "POTION":
        name, price = info["potions"][i]
        lost = " (no free slot: it would be lost)" if _no_free_slot(s) else ""
        return f"buy potion {name} for {price} gold{lost}"
    if a["type"] == "CARD_REMOVE":
        return f"pay {info['remove_cost']} gold to remove a card"
    return "leave the shop"


def _run_changes(before: dict, after: dict) -> str:
    """One line of what differs between two run states."""
    out = []
    for key, label in (("hp", "HP"), ("max_hp", "max HP"), ("gold", "gold"), ("act", "act"),
                       ("floor", "floor")):
        if before[key] != after[key]:
            out.append(f"{label} {before[key]}->{after[key]}")
    out += _diff_list("deck", [_names([c]) for c in before["deck"]], [_names([c]) for c in after["deck"]])
    out += _diff_list("relic", [n for n, _ in before["relics"]], [n for n, _ in after["relics"]])
    out += _diff_list("potion", [p for p in before["potions"] if p != "EMPTY_POTION_SLOT"],
                      [p for p in after["potions"] if p != "EMPTY_POTION_SLOT"])
    for name, b, a in zip(("ruby key", "emerald key", "sapphire key"), before["keys"], after["keys"]):
        if a and not b:
            out.append(f"got the {name}")
    if (before["screen"], before["screen_info"].get("event")) != (after["screen"], after["screen_info"].get("event")):
        out.append(f"now {_screen_title(after)}")
    elif after["outcome"] != "UNDECIDED":
        out.append(after["outcome"])
    return ", ".join(out) or "no visible change"


if __name__ == "__main__":
    print(render_state(new_game()))

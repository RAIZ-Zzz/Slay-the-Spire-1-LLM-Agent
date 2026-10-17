"""Tests for tools.py. Plain asserts, no framework.

    python sts1/test_tools.py           # everything (about 10 minutes)
    python sts1/test_tools.py quick     # skip the slow ones (seconds)
"""
import random
import sys
import time

import tools
from tools import sts

STEP_CAP = 20000  # a game that runs longer than this is treated as a hang


# --- helpers -------------------------------------------------------------------

def random_action(game, rng):
    """A uniformly random legal action, in the form tools expects on this screen."""
    if game.bc is not None and tools._is_multi(state := game.bc.state()):
        return rng.sample(range(len(state["hand"])), rng.randint(0, tools._multi_limit(state)))
    return rng.randrange(len(tools.legal_actions(game)))


def finish(game, rng, check=False):
    """Random legal actions until the game ends. With check=True, verify invariants every step."""
    for _ in range(STEP_CAP):
        if game.outcome != "UNDECIDED":
            return
        labels = tools.legal_actions(game)
        assert labels or _multi_screen(game), f"no legal action while undecided:\n{tools.render_state(game)}"
        action = random_action(game, rng)
        if check:
            assert not any("option" in x for x in labels), labels
            text = tools.render_state(game)
            assert "INVALID" not in text and "EMPTY_POTION_SLOT" not in text, text
            snapshot = _snapshot(game)
            predicted = tools.simulate(game, action).split("\n")[0]
            assert _snapshot(game) == snapshot, "simulate changed the real game"
            result = tools.act(game, action)
            assert predicted == "If you " + result, (predicted, result)
        else:
            tools.act(game, action)
    raise AssertionError(f"game did not end within {STEP_CAP} actions:\n{tools.render_state(game)}")


def _multi_screen(game):
    return game.bc is not None and tools._is_multi(game.bc.state())


def _snapshot(game):
    return (sts.game_state(game.gc), game.bc.state() if game.bc is not None else None)


def index_of(game, prefix):
    for i, label in enumerate(tools.legal_actions(game)):
        if label.startswith(prefix):
            return i
    raise AssertionError(f"no action starting with {prefix!r} in {tools.legal_actions(game)}")


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError(f"{fn.__name__}{args} should have raised ValueError")


def sensible_action(game, rng, sims=300):
    """MCTS in fights; outside them take rewards, rest when low, smith, else random. Gets runs deep."""
    if game.bc is not None:
        return tools.mcts_advice(game, sims)[0][0]
    labels = tools.legal_actions(game)
    s = sts.game_state(game.gc)
    takes = [i for i, x in enumerate(labels) if x.startswith("take") and "lost" not in x]
    if takes:
        return rng.choice(takes)
    if s["hp"] < 0.6 * s["max_hp"] and any(x.startswith("rest") for x in labels):
        return index_of(game, "rest")
    if any(x.startswith("smith") for x in labels):
        return index_of(game, "smith")
    return random_action(game, rng)


def play_checked(game, choose, rng, steps=STEP_CAP):
    """Play with `choose`, checking simulate == act and the label invariants every step."""
    for _ in range(steps):
        if game.outcome != "UNDECIDED":
            return
        labels = tools.legal_actions(game)
        assert labels or _multi_screen(game), tools.render_state(game)
        assert not any("option" in x for x in labels), labels
        action = choose(game, rng)
        snapshot = _snapshot(game)
        predicted = tools.simulate(game, action).split("\n")[0]
        assert _snapshot(game) == snapshot, "simulate changed the real game"
        assert predicted == "If you " + tools.act(game, action)


def run_until(game, screen, rng, limit=3000):
    """Random play until the run reaches `screen` (outside fights). Returns False if the run ended first."""
    for _ in range(limit):
        if game.outcome != "UNDECIDED":
            return False
        if game.bc is None and sts.game_state(game.gc)["screen"] == screen:
            return True
        if game.bc is not None:   # fights: let MCTS play so the run survives long enough
            tools.act(game, tools.mcts_advice(game, 30)[0][0])
        else:
            tools.act(game, random_action(game, rng))
    return False


# --- fights --------------------------------------------------------------------

def test_basics():
    game = tools.new_battle()
    assert tools.legal_actions(game)[-1] == "end turn"
    before = _snapshot(game)
    tools.simulate(game, 1)
    assert _snapshot(game) == before

    assert "does not exist" in raises(tools.act, game, 99)
    assert "does not exist" in raises(tools.act, game, -1)
    assert "one action number" in raises(tools.act, game, [0])
    assert "one action number" in raises(tools.act, game, True)
    assert "one action number" in raises(tools.act, game, "1")
    assert "unknown" in raises(tools.new_battle, "NOT_A_FIGHT")
    assert "unknown" in raises(tools.new_battle, extra_cards=("NOT_A_CARD",))
    assert "not a potion an Ironclad" in raises(tools.new_battle, potions=("POISON_POTION",))
    assert "not a relic an Ironclad" in raises(tools.new_battle, relics=("HOLY_WATER",))
    for card in ("NEUTRALIZE", "WOUND", "SMITE", "BULLET_TIME"):   # other class / status / generated-only
        assert "not a card an Ironclad deck" in raises(tools.new_battle, extra_cards=(card,))
    assert "no free potion slot" in raises(tools.new_battle, potions=("FIRE_POTION",) * 4)

    # Bash: 8 damage, 2 Vulnerable - the status change must be reported
    game = tools.new_battle(extra_cards=("BASH",) * 5)
    out = tools.act(game, index_of(game, "play Bash"))
    assert "JAW_WORM HP" in out and "Vulnerable 0->2" in out, out

    # a finished single fight refuses further actions
    while game.outcome == "UNDECIDED":
        tools.act(game, tools.mcts_advice(game, 100)[0][0])
    assert tools.legal_actions(game) == [] and "game is over" in raises(tools.act, game, 0)


def test_potions():
    pool = sts.ironclad_potions()
    assert len(pool) == 33 and "STANCE_POTION" not in pool
    for name in pool:
        rng = random.Random(name)
        game = tools.new_battle("JAW_WORM", potions=(name,))
        labels = tools.legal_actions(game)
        drinks = [x for x in labels if x.startswith("drink")]
        if name in ("SMOKE_BOMB", "FAIRY_POTION"):
            assert not drinks, (name, labels)        # Smoke Bomb is a no-op; Fairy triggers on death
            assert any(x.startswith("discard") for x in labels), (name, labels)
        else:
            assert drinks, (name, labels)
            tools.act(game, index_of(game, "drink"))
        finish(game, rng, check=True)

    # numbers from drinkPotion (BattleContext.cpp:2239)
    game = tools.new_battle(potions=("FIRE_POTION",))
    tools.act(game, index_of(game, "drink Fire Potion"))
    m = game.bc.state()["monsters"][0]
    assert m["hp"] == m["max_hp"] - 20
    game = tools.new_battle(potions=("BLOCK_POTION",))
    assert "your Block 0->12" in tools.act(game, index_of(game, "drink"))
    game = tools.new_battle(potions=("ENERGY_POTION",))
    assert "Energy 3->5" in tools.act(game, index_of(game, "drink"))
    game = tools.new_battle(potions=("WEAK_POTION",))
    assert "JAW_WORM Weak 0->3" in tools.act(game, index_of(game, "drink"))

    # discovery potions show the three generated cards by name
    for name in ("ATTACK_POTION", "SKILL_POTION", "POWER_POTION", "COLORLESS_POTION"):
        game = tools.new_battle(potions=(name,))
        tools.act(game, index_of(game, "drink"))
        labels = tools.legal_actions(game)
        assert len(labels) == 3 and all(x.startswith("DISCOVERY: take ") for x in labels), labels
        hand = len(game.bc.state()["hand"])
        tools.act(game, 0)
        assert len(game.bc.state()["hand"]) == hand + 1


def test_multi_select():
    # Elixir: exhaust any number of hand cards
    game = tools.new_battle(potions=("ELIXIR_POTION",))
    tools.act(game, index_of(game, "drink"))
    state = game.bc.state()
    assert tools._is_multi(state) and len(tools.legal_actions(game)) == len(state["hand"])
    assert "list of hand numbers" in tools.render_state(game)
    assert "list of hand numbers" in raises(tools.act, game, 0)
    assert "only once" in raises(tools.act, game, [1, 1])
    assert "hand numbers are" in raises(tools.act, game, [0, 9])
    assert "integers" in raises(tools.act, game, [0, "1"])
    picked = sorted([state["hand"][0]["name"], state["hand"][2]["name"]])
    tools.act(game, [2, 0])
    after = game.bc.state()
    assert len(after["hand"]) == len(state["hand"]) - 2
    assert sorted(c["name"] for c in after["exhaust_pile"]) == picked

    # Gambler's Brew: discard any number, draw that many
    game = tools.new_battle(potions=("GAMBLERS_BREW",))
    tools.act(game, index_of(game, "drink"))
    state = game.bc.state()
    tools.act(game, [0, 1])
    after = game.bc.state()
    assert len(after["hand"]) == len(state["hand"])
    assert len(after["discard_pile"]) == len(state["discard_pile"]) + 2

    # picking nothing is legal, and MCTS can only suggest that
    game = tools.new_battle(potions=("GAMBLERS_BREW",))
    tools.act(game, index_of(game, "drink"))
    assert tools.mcts_advice(game, 50)[0][0] == []
    tools.act(game, [])
    assert not tools._is_multi(game.bc.state())


def test_relics():
    rng = random.Random(7)
    for name in sorted(tools._IRONCLAD_RELICS):
        game = tools.new_battle("JAW_WORM", relics=(name,))
        finish(game, rng, check=True)

    # relics that open a choice
    game = tools.new_battle(relics=("TOOLBOX",))
    labels = tools.legal_actions(game)
    assert len(labels) == 3 and all(x.startswith("DISCOVERY: take ") for x in labels), labels
    game = tools.new_battle(relics=("GAMBLING_CHIP",))
    assert tools._is_multi(game.bc.state()), tools.render_state(game)
    game = tools.new_battle(relics=("NILRYS_CODEX",))
    tools.act(game, index_of(game, "end turn"))
    labels = tools.legal_actions(game)
    assert labels[-1] == "CODEX: skip" and len(labels) == 4, labels
    # start-of-battle relics show up as statuses / block
    assert tools.new_battle(relics=("VAJRA",)).bc.state()["player"]["statuses"].get("Strength") == 1
    assert tools.new_battle(relics=("ANCHOR",)).bc.state()["player"]["block"] == 10


def test_fight_fuzz(fights=300):
    """Random fights with random decks, relics and potions; every invariant checked every step."""
    rng = random.Random(2026)
    for _ in range(fights):
        finish(random_battle(rng), rng, check=True)


def random_battle(rng):
    """A fight with a random encounter, deck, relics and potions (all Ironclad-reachable)."""
    while True:
        try:
            return tools.new_battle(rng.choice([n for n in sts.MonsterEncounter.__members__ if n != "INVALID"]),
                                    seed=rng.randrange(1, 10**9), ascension=rng.choice((0, 20)),
                                    potions=tuple(rng.sample(sts.ironclad_potions(), rng.randint(0, 3))),
                                    relics=tuple(rng.sample(sorted(tools._IRONCLAD_RELICS), rng.randint(0, 6))),
                                    extra_cards=tuple(rng.choices(sorted(tools._IRONCLAD_CARDS),
                                                                  k=rng.randint(0, 15))))
        except ValueError as e:   # A20 has 2 potion slots and Sozu none: draw again
            assert "no free potion slot" in str(e), e


def test_mcts():
    game = tools.new_battle()
    for action, visits, _ in tools.mcts_advice(game, 200):
        assert 0 <= action < len(tools.legal_actions(game)) and visits > 0
    while game.outcome == "UNDECIDED":
        tools.act(game, tools.mcts_advice(game, 200)[0][0])
    assert game.outcome == "PLAYER_VICTORY"
    assert "only advises inside" in raises(tools.mcts_advice, tools.new_game())


# --- whole runs ----------------------------------------------------------------

def test_run_screens():
    rng = random.Random(11)

    # Neow: four options with their text
    game = tools.new_game()
    labels = tools.legal_actions(game)
    assert len(labels) == 4 and not any("option" in x for x in labels), labels

    # map: choosing a room moves you one floor up
    game = tools.new_game(seed=5)
    assert run_until(game, "MAP", rng)
    floor = sts.game_state(game.gc)["floor"]
    assert "Map ahead" in tools.render_state(game)
    out = tools.act(game, index_of(game, "go to"))
    assert f"floor {floor}->{floor + 1}" in out, out

    # shop, rest site and card removal, found by random play over a few seeds
    seen = set()
    for seed in range(1, 40):
        game = tools.new_game(seed=seed)
        while run_until(game, "SHOP", rng, limit=400) and "shop" not in seen:
            s = sts.game_state(game.gc)
            assert "For sale" in tools.render_state(game)
            buys = [i for i, x in enumerate(tools.legal_actions(game)) if x.startswith("buy")]
            if buys:
                out = tools.act(game, buys[0])
                assert f"gold {s['gold']}->" in out, out
                seen.add("shop")
            break
        game = tools.new_game(seed=seed)
        if run_until(game, "REST", rng, limit=600) and "rest" not in seen:
            labels = tools.legal_actions(game)
            if any(x.startswith("rest") for x in labels):
                hp = sts.game_state(game.gc)
                out = tools.act(game, index_of(game, "rest"))
                assert hp["hp"] == hp["max_hp"] or "HP" in out, out
                seen.add("rest")
        game = tools.new_game(seed=seed)
        if run_until(game, "CARD_SELECT", rng, limit=600) and "select" not in seen:
            deck = len(sts.game_state(game.gc)["deck"])
            tools.act(game, 0)
            seen.add("select")
            assert len(sts.game_state(game.gc)["deck"]) in (deck - 1, deck, deck + 1)
    assert seen == {"shop", "rest", "select"}, seen


def test_run_fuzz(runs=60):
    """Whole runs with random choices everywhere; every invariant checked every step."""
    rng = random.Random(99)
    floors = []
    for _ in range(runs):
        game = tools.new_game(seed=rng.randrange(1, 10**9), ascension=rng.choice((0, 20)))
        finish(game, rng, check=True)
        floors.append(sts.game_state(game.gc)["floor"])
    print(f"      random runs reached floor {min(floors)}..{max(floors)}, mean {sum(floors) / len(floors):.1f}")


def test_run_fuzz_deep(runs=100):
    """Runs played sensibly (MCTS fights, rewards taken), so they reach acts 2-3 and more events."""
    rng = random.Random(5)
    floors = []
    for _ in range(runs):
        game = tools.new_game(seed=rng.randrange(1, 10**9), ascension=rng.choice((0, 20)))
        play_checked(game, sensible_action, rng)
        floors.append(sts.game_state(game.gc)["floor"])
    print(f"      floors reached: max {max(floors)}, past act 1: {sum(f > 17 for f in floors)}/{runs}")


# every event, opened directly; colosseum and match-and-keep are disabled in the simulator
# (GameContext.h:152-153), the rest are rooms or Neow, covered elsewhere
_NOT_EVENTS = {"INVALID", "MONSTER", "REST", "SHOP", "TREASURE", "NEOW", "The Colosseum", "Match and Keep"}


def test_every_event():
    rng = random.Random(3)
    events = [e for e in sts.event_names() if e not in _NOT_EVENTS]
    assert len(events) == len(sts.event_names()) - len(_NOT_EVENTS), "a name in _NOT_EVENTS is misspelled"
    for event in events:
        opened = 0
        for seed in range(1, 3):
            for act_no, hp in ((1, 200), (2, 200), (3, 200), (2, 80)):
                game = tools.new_game(seed=seed)
                gc = game.gc
                sts.obtain_relic(gc, sts.RelicId.ANCHOR)              # N'loth needs two relics
                gc.obtain_card(sts.Card(sts.CardId.REGRET))           # Divine Fountain needs a curse
                if act_no > 1:
                    sts.debug_transition_to_act(gc, act_no)          # heals, so set HP after
                gc.gold, gc.max_hp, gc.cur_hp = 999, 200, hp        # every option affordable
                gc.floor_num = 10   # Dead Adventurer and Mushrooms need floor > 6, Moai Head half HP
                try:
                    sts.debug_start_event(gc, event)
                except ValueError as e:
                    assert "cannot appear here" in str(e), e
                    continue
                opened += 1
                if game.bc is None and sts.game_state(game.gc)["screen"] == "BATTLE":
                    game.bc = sts.BattleContext(game.gc)   # an event can open straight into a fight
                for i in range(len(tools.legal_actions(game))):
                    branch = game.copy()
                    tools.act(branch, i)
                    play_checked(branch, sensible_action, rng, steps=60)
        assert opened, f"{event} never opened in acts 1-3"


def late_act_game(seed, act_no, keys):
    game = tools.new_game(seed=seed)
    # 30000 HP so a starter deck grinds through; even so it cannot get past
    # Shield and Spear when dropped straight into act 4 (the shield blocks forever)
    game.gc.max_hp = game.gc.cur_hp = 30000
    if keys:
        sts.debug_set_keys(game.gc)
    sts.debug_transition_to_act(game.gc, act_no)
    return game


def test_late_acts():
    rng = random.Random(4)
    for act_no in (2, 4):   # the flow runs cleanly; winning is not expected
        play_checked(late_act_game(1, act_no, keys=act_no == 4), sensible_action, rng)

    # act 3: the boss ends the run without keys, and leads on to the Heart with them
    for keys in (False, True):
        for seed in range(1, 6):
            game = late_act_game(seed, 3, keys)
            play_checked(game, sensible_action, rng)
            s = sts.game_state(game.gc)
            if s["outcome"] == "PLAYER_VICTORY":
                assert s["act"] == (4 if keys else 3), (keys, s["act"])
                break
        else:
            raise AssertionError(f"never won from act 3 (keys={keys})")


SLOW = ("test_fight_fuzz", "test_run_fuzz", "test_run_fuzz_deep", "test_every_event", "test_late_acts")

if __name__ == "__main__":
    quick = "quick" in sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith("test_") and not (quick and name in SLOW):
            t = time.time()
            fn()
            print(f"ok  {name}  ({time.time() - t:.1f}s)")
    print("all passed")

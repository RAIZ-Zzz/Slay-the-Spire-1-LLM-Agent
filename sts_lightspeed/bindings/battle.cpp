// Combat-level Python bindings: step through one fight action by action.
//
// Exposes BattleContext so Python can read the full state (god mode: the draw pile
// comes back in true draw order), list legal actions, execute one, copy the battle
// for look-ahead, and ask the built-in MCTS for per-action visit counts.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <stdexcept>

#include "combat/BattleContext.h"
#include "game/GameContext.h"
#include "sim/search/Action.h"
#include "sim/search/BattleScumSearcher2.h"

namespace py = pybind11;
using namespace sts;
using search::BattleScumSearcher2;

namespace {

py::dict cardDict(const BattleContext &bc, const CardInstance &c, bool checkPlayable) {
    py::dict d;
    d["id"] = static_cast<int>(c.getId());
    d["name"] = c.getName();
    d["upgrades"] = c.getUpgradeCount();
    d["cost"] = static_cast<int>(c.costForTurn);
    d["base_cost"] = static_cast<int>(c.cost);
    d["type"] = cardTypeStrings[static_cast<int>(c.getType())];
    d["needs_target"] = c.requiresTarget();
    d["exhausts"] = c.doesExhaust();
    d["ethereal"] = c.isEthereal();
    if (checkPlayable) {
        d["playable"] = c.canUseOnAnyTarget(bc);
    }
    return d;
}

template <typename It>
py::list cardList(const BattleContext &bc, It begin, It end, bool checkPlayable = false) {
    py::list out;
    for (auto it = begin; it != end; ++it) {
        out.append(cardDict(bc, *it, checkPlayable));
    }
    return out;
}

py::dict monsterDict(const BattleContext &bc, const Monster &m) {
    py::dict d;
    d["idx"] = m.idx;
    d["id"] = static_cast<int>(m.id);
    d["name"] = m.getName();
    d["hp"] = m.curHp;
    d["max_hp"] = m.maxHp;
    d["block"] = m.block;
    d["alive"] = m.isAlive();
    d["targetable"] = m.isTargetable();
    d["half_dead"] = m.halfDead;

    // The move the monster will make this enemy turn - what the intent icon shows.
    d["move"] = monsterMoveStrings[static_cast<int>(m.moveHistory[0])];
    d["last_move"] = monsterMoveStrings[static_cast<int>(m.moveHistory[1])];
    d["attacking"] = m.isAttacking();
    const auto dmg = m.getMoveBaseDamage(bc);
    // damage_per_hit already includes the monster's strength/weak and the
    // player's vulnerable, i.e. the number the game draws on the intent.
    d["damage_per_hit"] = m.isAttacking() ? m.calculateDamageToPlayer(bc, dmg.damage) : 0;
    d["hits"] = m.isAttacking() ? dmg.attackCount : 0;

    py::dict statuses;
    for (int i = static_cast<int>(MS::ARTIFACT); i <= static_cast<int>(MS::STASIS); ++i) {
        const auto s = static_cast<MS>(i);
        if (m.hasStatusInternal(s)) {
            statuses[enemyStatusStrings[i]] = m.getStatusInternal(s);
        }
    }
    d["statuses"] = statuses;
    return d;
}

py::dict playerDict(const BattleContext &bc) {
    const auto &p = bc.player;
    py::dict d;
    d["hp"] = p.curHp;
    d["max_hp"] = p.maxHp;
    d["block"] = p.block;
    d["energy"] = p.energy;

    py::dict statuses;
    for (int i = 1; i < static_cast<int>(std::size(playerStatusStrings)); ++i) {
        const auto s = static_cast<PlayerStatus>(i);
        if (!p.hasStatusRuntime(s)) {
            continue;
        }
        // getStatusRuntime does statusMap.at(s), which throws for flag-only statuses
        // (a bit is set but no amount is stored); report those as 1.
        const bool inMap = p.statusMap.count(s) > 0;
        const bool inField = s == PS::ARTIFACT || s == PS::DEXTERITY || s == PS::FOCUS || s == PS::STRENGTH;
        statuses[playerStatusStrings[i]] = (inMap || inField) ? p.getStatusRuntime(s) : 1;
    }
    d["statuses"] = statuses;

    // Player keeps relics in two 64-bit masks (Player.cpp:15), so only ids < 128 are
    // representable in combat; higher ids alias other bits. The full relic list
    // lives in GameContext.relics.
    py::list relics;
    for (int i = 0; i < 128; ++i) {
        const auto r = static_cast<RelicId>(i);
        if (r != RelicId::INVALID && p.hasRelicRuntime(r)) {
            relics.append(relicNames[i]);
        }
    }
    d["relics"] = relics;
    return d;
}

std::vector<std::uint32_t> legalActions(const BattleContext &bc) {
    std::vector<std::uint32_t> out;
    if (bc.outcome != Outcome::UNDECIDED) {
        return out;
    }
    // Reuse the searcher's enumeration so "legal" means exactly what MCTS considers.
    // It lists one action per group of identical hand cards, which is equivalent play.
    BattleScumSearcher2 searcher(bc);
    BattleScumSearcher2::Node node;
    searcher.enumerateActionsForNode(node, bc);
    out.reserve(node.edges.size());
    for (const auto &e : node.edges) {
        out.push_back(e.action.bits);
    }
    return out;
}

const char *inputStateName(InputState s) {
    switch (s) {
        case InputState::PLAYER_NORMAL: return "PLAYER_NORMAL";
        case InputState::CARD_SELECT: return "CARD_SELECT";
        case InputState::EXECUTING_ACTIONS: return "EXECUTING_ACTIONS";
        default: return "OTHER";
    }
}

}  // namespace

void def_battle(py::module &m) {
    m.def("decode_action", [](std::uint32_t bits) {
        const search::Action a(bits);
        static constexpr const char *types[] = {
            "CARD", "POTION", "SINGLE_CARD_SELECT", "MULTI_CARD_SELECT", "END_TURN"};
        py::dict d;
        d["type"] = types[static_cast<int>(a.getActionType())];
        d["source"] = a.getSourceIdx();   // hand idx / potion slot / select idx
        d["target"] = a.getTargetIdx();   // monster idx; only meaningful if the card/potion targets
        const auto sel = a.getSelectedIdxs();  // multi card select only
        d["selected"] = std::vector<int>(sel.begin(), sel.end());
        return d;
    }, py::arg("action"), "split an action int into type / source / target / selected");

    m.def("multi_select_action", [](const std::vector<int> &handIdxs) {
        std::uint32_t mask = 0;
        for (int i : handIdxs) {
            if (i < 0 || i >= 10) {
                throw std::invalid_argument("hand index out of range: " + std::to_string(i));
            }
            mask |= 1u << i;
        }
        return search::Action(search::ActionType::MULTI_CARD_SELECT, static_cast<int>(mask)).bits;
    }, py::arg("hand_indices"),
       "action int for a multi card select screen (Elixir, Gambler's Brew, Gambling Chip)");

    m.def("discard_potion_action", [](int slot) {
        // a potion target above 5 means discard (Action.cpp:439)
        return search::Action(search::ActionType::POTION, slot, 0x1FFF).bits;
    }, py::arg("slot"));

    // --- setting up a fight: the existing bindings can read potions/relics but not add them
    m.def("obtain_potion", [](GameContext &gc, const std::string &name) {
        for (int i = 2; i < static_cast<int>(std::size(potionEnumNames)); ++i) {  // skip INVALID / EMPTY
            if (name == potionEnumNames[i]) {
                const int before = gc.potionCount;
                gc.obtainPotion(static_cast<Potion>(i));
                return gc.potionCount > before;  // false: slots full or Sozu
            }
        }
        throw std::invalid_argument("unknown potion: " + name);
    }, py::arg("gc"), py::arg("name"), "add a potion by enum name, e.g. FIRE_POTION");
    m.def("obtain_relic", [](GameContext &gc, RelicId r) { return gc.obtainRelic(r); },
          py::arg("gc"), py::arg("relic"), "add a relic (applies its pickup effect); false if already owned");
    m.def("card_color", [](CardId id) { return cardColorStrings[static_cast<int>(getCardColor(id))]; },
          py::arg("card"), "RED / GREEN / BLUE / PURPLE / COLORLESS / CURSE");
    m.def("potion_names", []() {
        return std::vector<std::string>(std::begin(potionEnumNames) + 2, std::end(potionEnumNames));
    });
    m.def("ironclad_potions", []() {
        std::vector<std::string> out;
        for (int i = 0; i < PotionPool::poolSize; ++i) {
            out.emplace_back(potionEnumNames[static_cast<int>(
                PotionPool::getPotionForClass(CharacterClass::IRONCLAD, i))]);
        }
        return out;
    }, "the 33 potions an Ironclad can find (Potions.h potionPool)");

    py::class_<BattleContext> bc(m, "BattleContext");

    bc.def(py::init([](const GameContext &gc, MonsterEncounter encounter) {
              BattleContext b;
              b.init(gc, encounter);
              return b;
          }),
          py::arg("gc"), py::arg("encounter"),
          "start a fight against `encounter` with gc's deck, relics, hp and potions")
        .def(py::init([](const GameContext &gc) {
              if (gc.screenState != ScreenState::BATTLE) {
                  throw std::invalid_argument("GameContext is not on a battle screen");
              }
              BattleContext b;
              b.init(gc);
              return b;
          }),
          py::arg("gc"), "start the fight the GameContext is currently in")

        .def("copy", [](const BattleContext &b) { return BattleContext(b); },
             "independent copy; stepping it does not touch the original")

        .def_property_readonly("outcome", [](const BattleContext &b) {
            return battleOutcomeStrings[static_cast<int>(b.outcome)];
        })
        .def_property_readonly("input_state", [](const BattleContext &b) {
            return inputStateName(b.inputState);
        })
        .def_property_readonly("card_select_task", [](const BattleContext &b) {
            return cardSelectTaskStrings[static_cast<int>(b.cardSelectInfo.cardSelectTask)];
        })
        .def_readonly("turn", &BattleContext::turn)
        .def_property_readonly("encounter", [](const BattleContext &b) {
            return monsterEncounterStrings[static_cast<int>(b.encounter)];
        })

        .def("state", [](const BattleContext &b) {
            py::dict d;
            d["turn"] = b.turn;
            d["outcome"] = battleOutcomeStrings[static_cast<int>(b.outcome)];
            d["input_state"] = inputStateName(b.inputState);
            d["player"] = playerDict(b);

            const auto &cards = b.cards;
            d["hand"] = cardList(b, cards.hand.begin(), cards.hand.begin() + cards.cardsInHand, true);
            // drawPile.back() is the next card drawn (CardManager.cpp:141), so reverse
            // to make draw_pile[0] the next draw. This is the god-mode field.
            d["draw_pile"] = cardList(b, cards.drawPile.rbegin(), cards.drawPile.rend());
            d["discard_pile"] = cardList(b, cards.discardPile.begin(), cards.discardPile.end());
            d["exhaust_pile"] = cardList(b, cards.exhaustPile.begin(), cards.exhaustPile.end());

            if (b.inputState == InputState::CARD_SELECT) {
                const auto &info = b.cardSelectInfo;
                py::dict sel;
                sel["task"] = cardSelectTaskStrings[static_cast<int>(info.cardSelectTask)];
                sel["pick_count"] = info.pickCount;   // EXHAUST_MANY limit
                // Discovery-style screens (potions, Toolbox, Nilry's Codex) pick from
                // three generated cards that sit in no pile.
                py::list generated;
                if (info.cardSelectTask == CardSelectTask::DISCOVERY ||
                    info.cardSelectTask == CardSelectTask::CODEX) {
                    for (const auto id : info.cards) {
                        generated.append(getCardName(id));
                    }
                }
                sel["generated"] = generated;
                d["select"] = sel;
            }

            py::list monsters;
            for (int i = 0; i < b.monsters.monsterCount; ++i) {
                monsters.append(monsterDict(b, b.monsters.arr[i]));
            }
            d["monsters"] = monsters;

            py::list potions;
            for (int i = 0; i < b.potionCapacity; ++i) {
                potions.append(potionNames[static_cast<int>(b.potions[i])]);
            }
            d["potions"] = potions;
            return d;
        }, "full state as plain dicts/lists (draw_pile[0] is the next card drawn)")

        .def("legal_actions", &legalActions,
             "legal actions as opaque ints; pass one back to step()")
        .def("describe", [](const BattleContext &b, std::uint32_t bits) {
            std::ostringstream os;
            search::Action(bits).printDesc(os, b);
            return os.str();
        }, py::arg("action"))
        .def("step", [](BattleContext &b, std::uint32_t bits) {
            const search::Action a(bits);
            if (!a.isValidAction(b)) {
                throw std::invalid_argument("illegal action " + std::to_string(bits));
            }
            a.execute(b);
        }, py::arg("action"), "execute one action and run the game until the next decision")

        .def("mcts", [](const BattleContext &b, std::int64_t simulations) {
            BattleScumSearcher2 searcher(b);
            searcher.search(simulations);
            py::list out;
            for (const auto &e : searcher.root.edges) {
                py::dict d;
                d["action"] = e.action.bits;
                d["visits"] = e.node.simulationCount;
                d["mean_value"] = e.node.simulationCount
                    ? e.node.evaluationSum / static_cast<double>(e.node.simulationCount) : 0.0;
                out.append(d);
            }
            return out;
        }, py::arg("simulations"),
           "run the built-in MCTS from here; per root action: visits and mean value")

        .def("exit_battle", [](const BattleContext &b, GameContext &gc) { b.exitBattle(gc); },
             py::arg("gc"), "write hp, deck changes, potions, relic counters back to gc")

        .def("__repr__", [](const BattleContext &b) {
            std::ostringstream os;
            os << b;
            return os.str();
        });
}

// Run-level Python bindings: every out-of-combat screen of a whole run.
//
// Mirrors battle.cpp: read the state as plain dicts, list legal actions as ints,
// execute one, copy the run for look-ahead. Fights themselves go through
// BattleContext (battle.cpp) once the run reaches a BATTLE screen.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>

#include "combat/CardInstance.h"
#include "constants/RelicPools.h"
#include "game/GameContext.h"
#include "sim/ConsoleSimulator.h"
#include "sim/search/GameAction.h"

namespace py = pybind11;
using namespace sts;
using search::GameAction;

namespace {

// GameAction bit layout (GameAction.cpp:15-31): bit 31 = potion action, bit 30 = discard.
constexpr std::uint32_t POTION_BIT = 0x80000000U;
constexpr std::uint32_t DISCARD_BIT = 0x40000000U;

const char *screenName(ScreenState s) {
    static constexpr const char *names[] = {"INVALID", "EVENT", "REWARDS", "BOSS_RELIC", "CARD_SELECT",
                                            "MAP", "TREASURE", "REST", "SHOP", "BATTLE"};
    return names[static_cast<int>(s)];
}

const char *selectTypeName(CardSelectScreenType t) {
    static constexpr const char *names[] = {"INVALID", "TRANSFORM", "TRANSFORM_UPGRADE", "UPGRADE", "REMOVE",
                                            "DUPLICATE", "OBTAIN", "BOTTLE", "BONFIRE_SPIRITS"};
    return names[static_cast<int>(t)];
}

py::dict cardDict(const Card &c) {
    py::dict d;
    d["id"] = static_cast<int>(c.getId());
    d["name"] = c.getName();
    d["upgrades"] = c.getUpgraded();
    d["cost"] = static_cast<int>(CardInstance(c).cost);
    d["type"] = cardTypeStrings[static_cast<int>(c.getType())];
    return d;
}

std::string eventText(const GameContext &gc) {
    // The console simulator already writes every event option with its real
    // numbers ("1: [Sacrifice] Gain 5 Max HP. Lose 7 HP."); reuse it for labels.
    ConsoleSimulator cs;
    cs.gc = const_cast<GameContext *>(&gc);  // printEventActions only reads
    std::ostringstream os;
    cs.printEventActions(os);
    return os.str();
}

py::list mapRows(const Map &map) {
    py::list rows;
    for (int y = 0; y < 15; ++y) {
        py::list row;
        for (int x = 0; x < 7; ++x) {
            const auto &n = map.getNode(x, y);
            if (n.edgeCount == 0 && n.parentCount == 0) {
                continue;  // unused grid cell
            }
            py::dict node;
            node["x"] = x;
            node["room"] = roomStrings[static_cast<int>(n.room)];
            node["next_x"] = std::vector<int>(n.edges.begin(), n.edges.begin() + n.edgeCount);
            node["burning_elite"] = map.burningEliteX == x && map.burningEliteY == y;
            row.append(node);
        }
        rows.append(row);
    }
    return rows;
}

py::dict screenDict(const GameContext &gc) {
    py::dict d;
    const auto &info = gc.info;
    switch (gc.screenState) {
        case ScreenState::EVENT_SCREEN:
            d["event"] = eventGameNames[static_cast<int>(gc.curEvent)];
            d["text"] = eventText(gc);
            break;

        case ScreenState::REWARDS: {
            const auto &r = info.rewardsContainer;
            d["gold"] = std::vector<int>(r.gold.begin(), r.gold.begin() + r.goldRewardCount);
            py::list cards;
            for (int i = 0; i < r.cardRewardCount; ++i) {
                py::list choice;
                for (const auto &c : r.cardRewards[i]) {
                    choice.append(cardDict(c));
                }
                cards.append(choice);
            }
            d["cards"] = cards;
            py::list relics, potions;
            for (int i = 0; i < r.relicCount; ++i) relics.append(getRelicName(r.relics[i]));
            for (int i = 0; i < r.potionCount; ++i) potions.append(getPotionName(r.potions[i]));
            d["relics"] = relics;
            d["potions"] = potions;
            d["emerald_key"] = r.emeraldKey;
            d["sapphire_key"] = r.sapphireKey;   // taking it gives up the last relic
            d["singing_bowl"] = gc.hasRelic(RelicId::SINGING_BOWL);
            break;
        }

        case ScreenState::BOSS_RELIC_REWARDS: {
            py::list relics;
            for (const auto r : info.bossRelics) relics.append(getRelicName(r));
            d["relics"] = relics;
            break;
        }

        case ScreenState::CARD_SELECT: {
            d["type"] = selectTypeName(info.selectScreenType);
            d["pick_count"] = info.toSelectCount;
            d["picked"] = static_cast<int>(info.haveSelectedCards.size());
            py::list cards;
            for (const auto &s : info.toSelectCards) cards.append(cardDict(s.card));
            d["cards"] = cards;
            break;
        }

        case ScreenState::MAP_SCREEN:
            d["x"] = gc.curMapNodeX;
            d["y"] = gc.curMapNodeY;   // -1 before the first floor of an act, 14 = last row before the boss
            d["rows"] = mapRows(*gc.map);
            break;

        case ScreenState::TREASURE_ROOM:
            d["chest"] = chestSizeNames[static_cast<int>(info.chestSize)];
            break;

        case ScreenState::SHOP_ROOM: {
            const auto &s = info.shop;
            py::list cards, relics, potions;
            for (int i = 0; i < 7; ++i) {
                py::dict c = cardDict(s.cards[i]);
                c["price"] = s.cardPrice(i);   // -1 = sold
                cards.append(c);
            }
            for (int i = 0; i < 3; ++i) {
                relics.append(py::make_tuple(getRelicName(s.relics[i]), s.relicPrice(i)));
                potions.append(py::make_tuple(getPotionName(s.potions[i]), s.potionPrice(i)));
            }
            d["cards"] = cards;
            d["relics"] = relics;
            d["potions"] = potions;
            d["remove_cost"] = s.removeCost;  // -1 = already used
            break;
        }

        default:
            break;
    }
    return d;
}

}  // namespace

template <typename... Pools>
std::vector<std::string> relicEnumNamesOf(RelicId starter, const Pools &...pools) {
    std::vector<std::string> out{relicEnumNames[static_cast<int>(starter)]};
    ([&] { for (auto r : pools) out.emplace_back(relicEnumNames[static_cast<int>(r)]); }(), ...);
    return out;
}

#define CLASS_POOLS(ns) ns::commonRelicPool, ns::uncommonRelicPool, ns::rareRelicPool, \
                        ns::bossRelicPool, ns::shopRelicPool

void def_game(py::module &m) {
    m.def("class_relics", [](CharacterClass cc) {
        // RelicPools.h: starter relic + the five pools each character draws from
        switch (cc) {
            case CharacterClass::IRONCLAD: return relicEnumNamesOf(RelicId::BURNING_BLOOD, CLASS_POOLS(Ironclad));
            case CharacterClass::SILENT: return relicEnumNamesOf(RelicId::RING_OF_THE_SNAKE, CLASS_POOLS(Silent));
            case CharacterClass::DEFECT: return relicEnumNamesOf(RelicId::CRACKED_CORE, CLASS_POOLS(Defect));
            default: return relicEnumNamesOf(RelicId::PURE_WATER, CLASS_POOLS(Watcher));
        }
    }, py::arg("character"), "enum names of the relics a character can find");

    m.def("ironclad_cards", []() {
        // The cardColors table in Cards.h is out of step with the CardId enum
        // (Bullet Time and Concentrate read as RED), so build the set from the
        // pools the run actually draws from instead (CardPools.h).
        std::vector<std::string> out;
        const auto add = [&](CardId id) { out.emplace_back(cardEnumStrings[static_cast<int>(id)]); };
        const int red = RarityCardPool::groupOffset[1][0];   // the Ironclad group is cardBlob[0, 72)
        for (int i = 0; i < red; ++i) add(RarityCardPool::cardBlob[i]);
        for (auto id : {CardId::STRIKE_RED, CardId::DEFEND_RED, CardId::BASH}) add(id);
        for (auto id : baseColorlessPool) add(id);
        for (auto id : curseCardPool) add(id);
        // given by events / Neow / ascension: J.A.X. (Augmenter), Bite (Vampires),
        // Apparition (Council of Ghosts), Ritual Dagger (The Nest), and their curses
        for (auto id : {CardId::JAX, CardId::BITE, CardId::APPARITION, CardId::RITUAL_DAGGER,
                        CardId::ASCENDERS_BANE, CardId::NECRONOMICURSE, CardId::CURSE_OF_THE_BELL}) add(id);
        return out;
    }, "enum names of every card an Ironclad run can put in the deck");

    m.def("copy_game", [](const GameContext &gc) {
        GameContext c(gc);
        // map is a shared_ptr that transitionToAct overwrites in place
        // (GameContext.cpp:709); give the copy its own.
        c.map = std::make_shared<Map>(*gc.map);
        return c;
    }, py::arg("gc"), "independent copy of a run");

    m.def("game_state", [](const GameContext &gc) {
        py::dict d;
        d["screen"] = screenName(gc.screenState);
        d["outcome"] = gc.outcome == GameOutcome::UNDECIDED ? "UNDECIDED"
                     : gc.outcome == GameOutcome::PLAYER_VICTORY ? "PLAYER_VICTORY" : "PLAYER_LOSS";
        d["act"] = gc.act;
        d["floor"] = gc.floorNum;
        d["ascension"] = gc.ascension;
        d["hp"] = gc.curHp;
        d["max_hp"] = gc.maxHp;
        d["gold"] = gc.gold;
        d["keys"] = py::make_tuple(gc.redKey, gc.greenKey, gc.blueKey);   // ruby, emerald, sapphire
        d["boss"] = monsterEncounterStrings[static_cast<int>(gc.boss)];

        py::list deck, relics, potions;
        for (const auto &c : gc.deck.cards) deck.append(cardDict(c));
        for (const auto &r : gc.relics.relics) relics.append(py::make_tuple(getRelicName(r.id), r.data));
        for (int i = 0; i < gc.potionCapacity; ++i) potions.append(getPotionName(gc.potions[i]));
        d["deck"] = deck;
        d["relics"] = relics;     // (name, counter)
        d["potions"] = potions;
        d["screen_info"] = screenDict(gc);
        return d;
    }, py::arg("gc"), "the run as plain dicts; screen_info depends on the screen");

    m.def("game_actions", [](const GameContext &gc) {
        std::vector<std::uint32_t> out;
        if (gc.outcome != GameOutcome::UNDECIDED || gc.screenState == ScreenState::BATTLE) {
            return out;
        }
        for (const auto &a : GameAction::getAllActionsInState(gc)) {
            out.push_back(a.bits);
        }
        // Singing Bowl: +2 max HP instead of a card. Valid (GameAction.cpp:190) but not listed.
        if (gc.screenState == ScreenState::REWARDS && gc.hasRelic(RelicId::SINGING_BOWL)) {
            for (int i = 0; i < gc.info.rewardsContainer.cardRewardCount; ++i) {
                out.push_back(GameAction(GameAction::RewardsActionType::CARD, i, 5).bits);
            }
        }
        // getAllActionsInState lists no potion actions; add the legal ones.
        for (int i = 0; i < gc.potionCapacity; ++i) {
            for (const std::uint32_t bits : {POTION_BIT | i, POTION_BIT | DISCARD_BIT | i}) {
                if (GameAction(bits).isValidAction(gc)) {
                    out.push_back(bits);
                }
            }
        }
        return out;
    }, py::arg("gc"), "legal out-of-combat actions as ints");

    m.def("decode_game_action", [](std::uint32_t bits) {
        const GameAction a(bits);
        static constexpr const char *types[] = {"CARD", "GOLD", "KEY", "POTION", "RELIC", "CARD_REMOVE", "SKIP", "?"};
        py::dict d;
        d["potion"] = a.isPotionAction();
        d["discard"] = a.isPotionDiscard();
        d["type"] = types[static_cast<int>(a.getRewardsActionType())];  // rewards / shop screens only
        d["idx1"] = a.getIdx1();
        d["idx2"] = a.getIdx2();
        return d;
    }, py::arg("action"));

    m.def("event_names", []() {
        return std::vector<std::string>(std::begin(eventGameNames), std::end(eventGameNames));
    });

    // --- test hooks: reach rare content without playing a whole run to it ------
    m.def("debug_start_event", [](GameContext &gc, const std::string &name) {
        for (int i = 0; i < static_cast<int>(std::size(eventGameNames)); ++i) {
            if (name == eventGameNames[i]) {
                const auto e = static_cast<Event>(i);
                // the game's own spawn rules; skipping them breaks events, e.g. N'loth
                // with one relic reads an uninitialised relic index (GameContext.cpp:387, 1005)
                if (!gc.canAddEvent(e) || !gc.canAddOneTimeEvent(e)) {
                    throw std::invalid_argument("event cannot appear here: " + name);
                }
                // what entering an event room does first: leaving the event returns
                // to the map (returnToMapAction, GameContext.cpp:20)
                gc.regainControlAction = [](GameContext &g) {
                    g.screenState = ScreenState::MAP_SCREEN;
                    g.regainControlAction = nullptr;
                };
                gc.curEvent = static_cast<Event>(i);
                gc.setupEvent();
                return;
            }
        }
        throw std::invalid_argument("unknown event: " + name);
    }, py::arg("gc"), py::arg("event"), "test hook: open an event by its game name");
    m.def("debug_transition_to_act", [](GameContext &gc, int act) {
        gc.transitionToAct(act);
        gc.screenState = ScreenState::MAP_SCREEN;
    }, py::arg("gc"), py::arg("act"), "test hook: jump to the start of an act (4 = the Heart)");
    m.def("debug_set_keys", [](GameContext &gc) {
        gc.redKey = gc.greenKey = gc.blueKey = true;
    }, py::arg("gc"), "test hook: hold all three keys, so act 3 leads to act 4");

    m.def("game_step", [](GameContext &gc, std::uint32_t bits) {
        const GameAction a(bits);
        if (!a.isValidAction(gc)) {
            throw std::invalid_argument("illegal game action " + std::to_string(bits));
        }
        if (a.isPotionAction()) {
            // GameAction::execute handles the potion and then falls through into the
            // screen switch with the slot as the option (GameAction.cpp:425-433),
            // e.g. discarding slot 0 on the map would also walk to column 0.
            if (a.isPotionDiscard()) {
                gc.discardPotionAtIdx(a.getIdx1());
            } else {
                gc.drinkPotionAtIdx(a.getIdx1());
            }
            return;
        }
        a.execute(gc);
    }, py::arg("gc"), py::arg("action"), "execute one out-of-combat action");
}

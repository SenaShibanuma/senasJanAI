# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import gzip
import numpy as np
import urllib.parse
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.meld import Meld
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.hand import HandCalculator
import os

class TransformerParser:
    def __init__(self):
        self.shanten_calculator = Shanten()
        self.hand_calculator = HandCalculator()
        self.tiles_converter = TilesConverter()
        self.reset_game_state()

    def reset_game_state(self):
        self.game_state = {'rules': {}, 'players': [''] * 4, 'config': HandConfig()}
        self.reset_round_state()

    def reset_round_state(self):
        self.round_state = {
            'events': [], 'round': 0, 'honba': 0, 'riichi_sticks': 0, 'dora_indicators': [],
            'scores': [25000] * 4, 'hands_136': [[] for _ in range(4)], 'melds': [[] for _ in range(4)],
            'rivers': [[] for _ in range(4)], 'is_riichi': [False] * 4, 'riichi_turn': [-1] * 4,
            'oya_player_id': 0, 'remaining_tiles': 70, 'turn_num': 0, 'last_drawn_tile': [None] * 4,
        }
        self.training_data = []

    def parse_log_file(self, filepath):
        self.reset_game_state()
        try:
            open_func = gzip.open if filepath.endswith('.gz') else open
            with open_func(filepath, 'rb') as f:
                xml_content = f.read()
            log_text = xml_content.decode('utf-8').strip().replace("</mjloggm>", "")
            root = ET.fromstring(f"<root>{log_text}</root>")
            all_tags = list(root)
            for i, element in enumerate(all_tags):
                self.process_tag(element, all_tags, i)
        except Exception:
            return []
        return self.training_data

    def process_tag(self, element, all_tags, index):
        tag, attrib = element.tag, element.attrib
        if tag == 'GO': self._process_go(attrib)
        elif tag == 'UN': self._process_un(attrib)
        elif tag == 'INIT': self._process_init(attrib)
        elif tag[0] in "TUVW" and tag[1:].isdigit(): self._process_draw(tag, all_tags, index)
        elif tag[0] in "DEFG" and tag[1:].isdigit(): self._process_discard(tag, all_tags, index)
        elif tag == 'N': self._process_meld(attrib)
        elif tag == 'REACH': self._process_riichi(attrib)
        elif tag == 'DORA': self._process_new_dora(attrib)
        elif tag == 'AGARI': self._add_event({'event_id': 'AGARI', 'who': int(attrib.get('who')), 'from': int(attrib.get('fromWho'))})
        elif tag == 'RYUUKYOKU': self._add_event({'event_id': 'RYUUKYOKU', 'type': attrib.get('type')})

    def _add_event(self, event_dict):
        self.round_state['events'].append(event_dict)

    def _parse_rules(self, type_attribute):
        type_val = int(type_attribute)
        has_kuitan = bool(type_val & 0x0010)
        has_aka_dora = bool(type_val & 0x0040)
        self.game_state['config'] = HandConfig(options=OptionalRules(has_open_tanyao=has_kuitan, has_aka_dora=has_aka_dora))
        return {'has_kuitan': has_kuitan, 'has_aka_dora': has_aka_dora}

    def _process_go(self, attrib):
        type_attr = attrib.get('type')
        if type_attr: self.game_state['rules'] = self._parse_rules(type_attr)
        self._add_event({'event_id': 'GAME_START', 'rules': self.game_state['rules']})

    def _process_un(self, attrib):
        self.game_state['players'] = [urllib.parse.unquote(attrib.get(f'n{i}', f'P{i}')) for i in range(4)]

    def _process_init(self, attrib):
        self.reset_round_state()
        seed = [int(s) for s in attrib.get('seed').split(',')]
        self.round_state.update({
            'round': seed[0], 'honba': seed[1], 'riichi_sticks': seed[2], 'dora_indicators': [seed[5]],
            'scores': [int(s) for s in attrib.get('ten').split(',')], 'oya_player_id': int(attrib.get('oya'))
        })
        for i in range(4):
            self.round_state['hands_136'][i] = sorted([int(p) for p in attrib.get(f'hai{i}', '').split(',') if p])
        self._add_event({'event_id': 'INIT', 'round': self.round_state['round'], 'honba': self.round_state['honba'], 'dora_indicator': seed[5]})

    def _process_draw(self, tag, all_tags, index):
        player = "TUVW".find(tag[0])
        tile = int(tag[1:])
        self.round_state['remaining_tiles'] -= 1
        self.round_state['turn_num'] += 1
        self.round_state['hands_136'][player].append(tile)
        self.round_state['hands_136'][player].sort()
        self.round_state['last_drawn_tile'][player] = tile
        self._add_event({'event_id': 'DRAW', 'player': player, 'tile': tile})
        self._generate_my_turn_training_data(player, all_tags, index)

    def _process_discard(self, tag, all_tags, index):
        player = "DEFG".find(tag[0])
        tile = int(tag[1:])
        is_tedashi = tile != self.round_state['last_drawn_tile'][player]
        if tile in self.round_state['hands_136'][player]:
            self.round_state['hands_136'][player].remove(tile)
        self.round_state['rivers'][player].append(tile)
        self._add_event({'event_id': 'DISCARD', 'player': player, 'tile': tile, 'is_tedashi': is_tedashi})
        self._generate_opponent_turn_training_data(player, tile, all_tags, index)

    def _process_meld(self, attrib):
        player = int(attrib.get('who'))
        m = int(attrib.get('m'))
        meld = Meld.decode_m(m)
        self.round_state['melds'][player].append(meld)
        tiles_to_remove = [t for t in meld.tiles if t != meld.called_tile]
        for tile in tiles_to_remove:
            if tile in self.round_state['hands_136'][player]:
                 self.round_state['hands_136'][player].remove(tile)
        self._add_event({'event_id': 'MELD', 'player': player, 'type': meld.type, 'tiles': meld.tiles})

    def _process_riichi(self, attrib):
        player = int(attrib.get('who'))
        step = int(attrib.get('step'))
        if step == 1:
            self.round_state['is_riichi'][player] = True
            self.round_state['riichi_turn'][player] = self.round_state['turn_num']
            self._add_event({'event_id': 'RIICHI_DECLARED', 'player': player})
        elif step == 2:
            self.round_state['scores'][player] -= 1000
            self.round_state['riichi_sticks'] += 1
            self._add_event({'event_id': 'RIICHI_ACCEPTED', 'player': player})

    def _process_new_dora(self, attrib):
        dora_indicator = int(attrib.get('hai'))
        self.round_state['dora_indicators'].append(dora_indicator)
        self._add_event({'event_id': 'NEW_DORA', 'dora_indicator': dora_indicator})

    def _create_and_add_training_point(self, player_pov, choices, label):
        if not choices or label not in choices: return
        context = self.round_state['events'].copy()
        self.training_data.append({"context": context, "choices": choices, "label": label, "player_pov": player_pov})

    def _generate_my_turn_training_data(self, player, all_tags, current_tag_index):
        choices = self._get_my_turn_actions(player, self.round_state['last_drawn_tile'][player])
        if len(choices) <= 1: return
        
        label = "ACTION_PASS"
        next_tag_index = current_tag_index + 1
        if next_tag_index < len(all_tags):
            next_element = all_tags[next_tag_index]
            tag, attrib = next_element.tag, next_element.attrib
            if tag[0] in "DEFG": label = f"DISCARD_{int(tag[1:])}"
            elif tag == "AGARI" and int(attrib.get('who')) == player: label = "ACTION_TSUMO_AGARI"
            elif tag == "REACH" and int(attrib.get('step')) == 1 and int(attrib.get('who')) == player:
                # Find the subsequent discard tag for the riichi action
                if next_tag_index + 1 < len(all_tags):
                    riichi_discard_tag = all_tags[next_tag_index + 1]
                    if riichi_discard_tag.tag[0] in "DEFG":
                        label = f"ACTION_RIICHI_{int(riichi_discard_tag.tag[1:])}"
        
        self._create_and_add_training_point(player, choices, label)

    def _generate_opponent_turn_training_data(self, discarder, discarded_tile, all_tags, current_tag_index):
        next_tag_index = current_tag_index + 1
        actual_action_map = {}
        if next_tag_index < len(all_tags):
            next_element = all_tags[next_tag_index]
            tag, attrib = next_element.tag, next_element.attrib
            who = int(attrib.get('who', -1))
            if tag == 'N' and who != -1:
                actual_action_map[who] = self._meld_obj_to_action_label(Meld.decode_m(int(attrib.get('m'))))
            elif tag == 'AGARI' and who != -1:
                actual_action_map[who] = "ACTION_RON_AGARI"

        for player in range(4):
            if player == discarder: continue
            choices = self._get_opponent_turn_actions(player, discarder, discarded_tile)
            if len(choices) <= 1: continue
            label = actual_action_map.get(player, "ACTION_PASS")
            self._create_and_add_training_point(player, choices, label)

    def _get_my_turn_actions(self, player, win_tile):
        actions = []
        hand_136 = self.round_state['hands_136'][player]
        unique_tiles_136 = sorted(list(set(hand_136)))
        for t in unique_tiles_136: actions.append(f"DISCARD_{t}")
        if self._can_agari(hand_136, win_tile, is_tsumo=True, player_id=player): actions.append("ACTION_TSUMO_AGARI")
        
        shanten, _ = self._calculate_shanten_and_ukeire(hand_136)
        if shanten == 0 and not self.round_state['is_riichi'][player]:
            for t in unique_tiles_136:
                 temp_hand = hand_136.copy(); temp_hand.remove(t)
                 if self._calculate_shanten_and_ukeire(temp_hand)[0] == 0:
                     actions.append(f"ACTION_RIICHI_{t}")
        return list(dict.fromkeys(actions))

    def _get_opponent_turn_actions(self, player, discarder, discarded_tile):
        actions = ["ACTION_PASS"]
        hand_136 = self.round_state['hands_136'][player]
        if self._can_agari(hand_136 + [discarded_tile], discarded_tile, is_tsumo=False, player_id=player):
            actions.append("ACTION_RON_AGARI")
        
        hand_34_counts = TilesConverter.to_34_array(hand_136)
        tile_34 = discarded_tile // 4
        if hand_34_counts[tile_34] >= 2: actions.append(f"ACTION_PUNG")
        if (discarder + 1) % 4 == player and tile_34 < 27:
            for p in [[-2, -1], [-1, 1], [1, 2]]:
                t1, t2 = tile_34 + p[0], tile_34 + p[1]
                if t1 >= 0 and t2 < 27 and (t1 // 9 == t2 // 9 == tile_34 // 9):
                    if hand_34_counts[t1] > 0 and hand_34_counts[t2] > 0:
                        actions.append(f"ACTION_CHII_{t1}_{t2}")
        if hand_34_counts[tile_34] >= 3: actions.append("ACTION_DAIMINKAN")
        return list(dict.fromkeys(actions))

    def _can_agari(self, hand_136, win_tile, is_tsumo, player_id):
        result = self.hand_calculator.estimate_hand_value(
            tiles=hand_136, win_tile=win_tile, melds=self.round_state['melds'][player_id],
            dora_indicators=self.round_state['dora_indicators'], config=self._get_current_hand_config(is_tsumo, player_id)
        )
        return result.error is None

    def _calculate_shanten_and_ukeire(self, hand_136):
        try: return self.shanten_calculator.calculate_shanten(TilesConverter.to_34_array(hand_136)), 0
        except: return 9, 0

    def _get_current_hand_config(self, is_tsumo, player_id):
        return HandConfig(
            is_tsumo=is_tsumo, is_riichi=self.round_state['is_riichi'][player_id],
            player_wind=Meld.EAST + ((player_id - self.round_state['oya_player_id'] + 4) % 4),
            round_wind=Meld.EAST + self.round_state['round'] // 4, options=self.game_state['config'].options
        )

    def _meld_obj_to_action_label(self, meld_obj):
        if meld_obj.type == Meld.CHI:
            consumed = sorted([t // 4 for t in meld_obj.tiles if t != meld_obj.called_tile])
            return f"ACTION_CHII_{consumed[0]}_{consumed[1]}"
        elif meld_obj.type == Meld.PON: return "ACTION_PUNG"
        elif meld_obj.type == Meld.KAN: return "ACTION_DAIMINKAN"
        return "ACTION_UNKNOWN"

    def run(self, filepaths):
        all_data = []
        for filepath in filepaths: all_data.extend(self.parse_log_file(filepath))
        return all_data
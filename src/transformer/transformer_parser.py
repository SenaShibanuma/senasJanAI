# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import gzip
import urllib.parse
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.meld import Meld
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.hand import HandCalculator
import os

class TransformerParser:
    """
    【XML解析・最終修正版】
    正常なXMLファイルを破壊していたロジックを修正。
    まずXMLをそのまま解析し、失敗した場合にのみ不完全なログの修復を試みる。
    """
    def __init__(self):
        self.shanten_calculator = Shanten()
        self.hand_calculator = HandCalculator()
        self.reset_game_state()

    def reset_game_state(self):
        self.game_state = {'rules': {}, 'players': [''] * 4, 'config': HandConfig()}
        self.pending_my_turn_data = None
        self.pending_opponent_turn_data = None
        self.reset_round_state()

    def reset_round_state(self):
        self.round_state = {
            'events': [], 'round': 0, 'honba': 0, 'riichi_sticks': 0, 'dora_indicators': [],
            'scores': [25000] * 4, 'hands_136': [[] for _ in range(4)], 'melds': [[] for _ in range(4)],
            'is_riichi': [False] * 4, 'oya_player_id': 0, 'last_drawn_tile': [None] * 4,
        }
        self.training_data = []
        self.pending_my_turn_data = None
        self.pending_opponent_turn_data = None

    def parse_log_file(self, filepath):
        self.reset_game_state()
        try:
            # --- ここからが修正箇所 ---
            # ファイル読み込み (gzip/通常ファイル両対応)
            try:
                with gzip.open(filepath, 'rb') as f: xml_content = f.read()
            except (gzip.BadGzipFile, OSError):
                with open(filepath, 'rb') as f: xml_content = f.read()

            log_text = xml_content.decode('utf-8').strip()

            # XML解析ロジックの修正
            root_elements = []
            try:
                # 1. まず、そのままXMLとして解析を試みる (正常なファイル向け)
                root = ET.fromstring(log_text)
                root_elements = [root] 
            except ET.ParseError:
                # 2. 失敗した場合、不完全なログと仮定して<root>で囲んで修復を試みる
                try:
                    repaired_text = f"<root>{log_text}</root>"
                    root = ET.fromstring(repaired_text)
                    root_elements = list(root)
                except ET.ParseError:
                    return []
            # --- ここまでが修正箇所 ---
            
            final_elements = []
            for element in root_elements:
                if element.tag == 'mjloggm':
                    final_elements.extend(list(element))
                else:
                    final_elements.append(element)

            for element in final_elements:
                self.process_tag(element)

        except Exception:
            return []
        return self.training_data

    def process_tag(self, element):
        tag, attrib = element.tag, element.attrib
        self._resolve_pending_actions(element)

        if tag == 'GO': self._process_go(attrib)
        elif tag == 'UN': self._process_un(attrib)
        elif tag == 'INIT': self._process_init(attrib)
        elif tag[0] in "TUVW" and tag[1:].isdigit(): self._process_draw(tag)
        elif tag[0] in "DEFG" and tag[1:].isdigit(): self._process_discard(tag)
        elif tag == 'N': self._process_meld(attrib)
        elif tag == 'REACH': self._process_riichi(attrib)
        elif tag == 'DORA': self._add_event({'event_id': 'NEW_DORA', 'dora_indicator': int(attrib.get('hai'))})
        elif tag in ['AGARI', 'RYUUKYOKU', 'TAIKYOKU']:
            self.pending_my_turn_data = None
            self.pending_opponent_turn_data = None
            if tag == 'AGARI': self._add_event({'event_id': 'AGARI', 'who': int(attrib.get('who')), 'from': int(attrib.get('fromWho'))})
            elif tag == 'RYUUKYOKU': self._add_event({'event_id': 'RYUUKYOKU', 'type': attrib.get('type')})

    def _resolve_pending_actions(self, current_element):
        if self.pending_opponent_turn_data:
            acting_player, label = -1, "ACTION_PASS"
            if current_element.tag in ['N', 'AGARI']:
                acting_player = int(current_element.attrib.get('who'))
                if current_element.tag == 'N': label = self._meld_obj_to_action_label(Meld.decode_m(int(current_element.attrib.get('m'))))
                else: label = "ACTION_RON_AGARI"
            for data in self.pending_opponent_turn_data:
                final_label = label if data["player_pov"] == acting_player else "ACTION_PASS"
                self._create_and_add_training_point(data["player_pov"], data["choices"], final_label)
            self.pending_opponent_turn_data = None

        if self.pending_my_turn_data:
            player, label = self.pending_my_turn_data["player_pov"], None
            if current_element.tag[0] in "DEFG" and "DEFG".find(current_element.tag[0]) == player:
                tile = int(current_element.tag[1:])
                label = f"ACTION_RIICHI_{tile}" if self.pending_my_turn_data.get('is_riichi_declared_this_turn') else f"DISCARD_{tile}"
            elif current_element.tag == 'AGARI' and int(current_element.attrib.get('who')) == player:
                label = "ACTION_TSUMO_AGARI"
            if label:
                self._create_and_add_training_point(player, self.pending_my_turn_data["choices"], label)
                self.pending_my_turn_data = None

    def _add_event(self, event_dict): self.round_state['events'].append(event_dict)
    def _parse_rules(self, type_attr):
        val = int(type_attr)
        self.game_state['config'] = HandConfig(options=OptionalRules(has_open_tanyao=bool(val & 0x10), has_aka_dora=bool(val & 0x40)))
        return {'has_kuitan': bool(val & 0x10), 'has_aka_dora': bool(val & 0x40)}

    def _process_go(self, attrib):
        self.game_state['rules'] = self._parse_rules(attrib.get('type'))
        self._add_event({'event_id': 'GAME_START', 'rules': self.game_state['rules']})

    def _process_un(self, attrib):
        self.game_state['players'] = [urllib.parse.unquote(attrib.get(f'n{i}', '')) for i in range(4)]

    def _process_init(self, attrib):
        self.reset_round_state()
        seed = [int(s) for s in attrib.get('seed').split(',')]
        self.round_state.update({'round': seed[0], 'honba': seed[1], 'riichi_sticks': seed[2],
                                 'dora_indicators': [seed[5]], 'oya_player_id': int(attrib.get('oya')),
                                 'scores': [int(s) for s in attrib.get('ten').split(',')]})
        for i in range(4): self.round_state['hands_136'][i] = sorted([int(p) for p in attrib.get(f'hai{i}').split(',') if p])
        self._add_event({'event_id': 'INIT', **self.round_state})

    def _process_draw(self, tag):
        player, tile = "TUVW".find(tag[0]), int(tag[1:])
        self.round_state['hands_136'][player].append(tile); self.round_state['hands_136'][player].sort()
        self.round_state['last_drawn_tile'][player] = tile
        self._add_event({'event_id': 'DRAW', 'player': player, 'tile': tile})
        self.pending_my_turn_data = {"player_pov": player, "choices": self._get_my_turn_actions(player, tile)}

    def _process_discard(self, tag):
        player, tile = "DEFG".find(tag[0]), int(tag[1:])
        if tile in self.round_state['hands_136'][player]: self.round_state['hands_136'][player].remove(tile)
        self._add_event({'event_id': 'DISCARD', 'player': player, 'tile': tile})
        self.pending_opponent_turn_data = []
        for p_idx in range(4):
            if p_idx != player:
                choices = self._get_opponent_turn_actions(p_idx, player, tile)
                if len(choices) > 1: self.pending_opponent_turn_data.append({"player_pov": p_idx, "choices": choices})

    def _process_meld(self, attrib):
        player, m = int(attrib.get('who')), int(attrib.get('m'))
        meld = Meld.decode_m(m)
        self.round_state['melds'][player].append(meld)
        tiles_to_remove = [t for t in meld.tiles if t != meld.called_tile]
        for tile in tiles_to_remove:
            if tile in self.round_state['hands_136'][player]: self.round_state['hands_136'][player].remove(tile)
        self._add_event({'event_id': 'MELD', 'player': player, 'meld': meld})

    def _process_riichi(self, attrib):
        player, step = int(attrib.get('who')), int(attrib.get('step'))
        if step == 1 and self.pending_my_turn_data and self.pending_my_turn_data["player_pov"] == player:
            self.pending_my_turn_data['is_riichi_declared_this_turn'] = True
        elif step == 2:
            self.round_state['is_riichi'][player] = True
            self.round_state['scores'][player] -= 1000
            self.round_state['riichi_sticks'] += 1

    def _create_and_add_training_point(self, pov, choices, label):
        if choices and label in choices:
            self.training_data.append({"context": self.round_state['events'].copy(), "choices": choices, "label": label, "player_pov": pov})

    def _get_my_turn_actions(self, p_idx, win_tile):
        actions, hand = [], self.round_state['hands_136'][p_idx]
        unique_tiles = sorted(list(set(hand)))
        for t in unique_tiles: actions.append(f"DISCARD_{t}")
        if self._can_agari(hand, win_tile, True, p_idx): actions.append("ACTION_TSUMO_AGARI")
        try: shanten = self.shanten_calculator.calculate_shanten(TilesConverter.to_34_array(hand))
        except: shanten = 9
        if shanten == 0 and not self.round_state['is_riichi'][p_idx]:
            for t in unique_tiles:
                temp_hand = hand.copy(); temp_hand.remove(t)
                try:
                    if self.shanten_calculator.calculate_shanten(TilesConverter.to_34_array(temp_hand)) == 0:
                        actions.append(f"ACTION_RIICHI_{t}")
                except: continue
        return list(dict.fromkeys(actions))

    def _get_opponent_turn_actions(self, p_idx, discarder, tile):
        actions, hand = ["ACTION_PASS"], self.round_state['hands_136'][p_idx]
        if self._can_agari(hand + [tile], tile, False, p_idx): actions.append("ACTION_RON_AGARI")
        counts, tile_34 = TilesConverter.to_34_array(hand), tile // 4
        if counts[tile_34] >= 2: actions.append("ACTION_PUNG")
        if (discarder + 1) % 4 == p_idx and tile_34 < 27:
            for p in [[-2,-1],[-1,1],[1,2]]:
                t1,t2 = tile_34+p[0], tile_34+p[1]
                if t1>=0 and t2<27 and t1//9==t2//9==tile_34//9 and counts[t1]>0 and counts[t2]>0:
                    actions.append(f"ACTION_CHII_{min(t1,t2)}_{max(t1,t2)}")
        if counts[tile_34] >= 3: actions.append("ACTION_DAIMINKAN")
        return list(dict.fromkeys(actions))
    
    def _get_config(self, is_tsumo, p_idx):
        return HandConfig(is_tsumo=is_tsumo, is_riichi=self.round_state['is_riichi'][p_idx],
                          player_wind=Meld.EAST + ((p_idx-self.round_state['oya_player_id']+4)%4),
                          round_wind=Meld.EAST + self.round_state['round']//4,
                          options=self.game_state['config'].options)

    def _can_agari(self, hand, win_tile, is_tsumo, p_idx):
        try:
            res = self.hand_calculator.estimate_hand_value(hand, win_tile, melds=self.round_state['melds'][p_idx],
                                                            dora_indicators=self.round_state['dora_indicators'],
                                                            config=self._get_config(is_tsumo, p_idx))
            return res.error is None
        except: return False

    def _meld_obj_to_action_label(self, meld):
        if meld.type == Meld.CHI:
            tiles = sorted([t//4 for t in meld.tiles if t != meld.called_tile])
            return f"ACTION_CHII_{tiles[0]}_{tiles[1]}"
        elif meld.type == Meld.PON: return "ACTION_PUNG"
        elif meld.type == Meld.KAN: return "ACTION_DAIMINKAN"
        return "UNKNOWN"

    def run(self, filepaths):
        data = []
        for f in filepaths: data.extend(self.parse_log_file(f))
        return data
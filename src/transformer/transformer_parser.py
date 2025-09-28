# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import gzip
import numpy as np
import urllib.parse
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.meld import Meld
import os

class TransformerParser:
    """
    天鳳の.mjlogファイルを解析し、Transformerモデル用の学習データを生成するクラス。
    mahjong==1.3.0の機能と、自前の判定ロジックで動作する。
    """
    def __init__(self):
        self.shanten_calculator = Shanten()
        self.reset_game_state()

    def reset_game_state(self):
        self.game_state = {'rules': {}, 'players': [''] * 4}
        self.reset_round_state()

    def reset_round_state(self):
        self.round_state = {
            'events': [], 'round': 0, 'honba': 0, 'riichi_sticks': 0,
            'dora_indicators': [], 'scores': [25000] * 4,
            'hands_136': [[] for _ in range(4)],
            'melds': [[] for _ in range(4)],
            'rivers': [[] for _ in range(4)],
            'is_riichi': [False] * 4, 'riichi_turn': [-1] * 4,
            'oya_player_id': 0, 'remaining_tiles': 70, 'turn_num': 0,
            'last_drawn_tile': [None] * 4,
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

            all_tags = list(root.iter())
            for i, element in enumerate(all_tags):
                if element.tag in ['root', 'mjloggm']: continue
                next_element = all_tags[i + 1] if i + 1 < len(all_tags) else None
                self.process_tag(element, next_element)
        except Exception:
            return []
        return self.training_data

    def process_tag(self, element, next_element):
        tag, attrib = element.tag, element.attrib
        if tag == 'GO': self._process_go(attrib)
        elif tag == 'UN': self._process_un(attrib)
        elif tag == 'INIT': self._process_init(attrib)
        elif tag[0] in "TUVW" and tag[1:].isdigit(): self._process_draw(tag, next_element)
        elif tag[0] in "DEFG" and tag[1:].isdigit(): self._process_discard(tag, next_element)
        elif tag == 'N': self._process_meld(attrib)
        elif tag == 'REACH': self._process_riichi(attrib)
        elif tag == 'DORA': self._process_new_dora(attrib)
        elif tag == 'AGARI': self._process_agari(attrib)
        elif tag == 'RYUUKYOKU': self._process_ryukyoku(attrib)

    def _add_event(self, event_dict):
        self.round_state['events'].append(event_dict)

    def _parse_rules(self, type_attribute):
        type_val = int(type_attribute)
        return {'has_kuitan': bool(type_val & 0x0010), 'has_aka_dora': bool(type_val & 0x0040)}

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
            'round': seed[0], 'honba': seed[1], 'riichi_sticks': seed[2],
            'dora_indicators': [seed[5]],
            'scores': [int(s) for s in attrib.get('ten').split(',')],
            'oya_player_id': int(attrib.get('oya'))
        })
        for i in range(4):
            self.round_state['hands_136'][i] = sorted([int(p) for p in attrib.get(f'hai{i}', '').split(',') if p])
        self._add_event({'event_id': 'INIT', 'round': self.round_state['round'], 'dora': seed[5] // 4})

    def _process_draw(self, tag, next_element):
        player = "TUVW".find(tag[0])
        tile = int(tag[1:])
        self.round_state['remaining_tiles'] -= 1
        self.round_state['hands_136'][player].append(tile)
        self.round_state['hands_136'][player].sort()
        self.round_state['last_drawn_tile'][player] = tile
        self._add_event({'event_id': 'DRAW', 'player': player, 'tile': tile // 4, 'is_red': tile in [16, 52, 88]})
        self._generate_my_turn_training_data(player, next_element, win_tile=tile)

    def _process_discard(self, tag, next_element):
        player = "DEFG".find(tag[0])
        tile = int(tag[1:])
        is_tedashi = tile != self.round_state['last_drawn_tile'][player]
        if tile in self.round_state['hands_136'][player]:
            self.round_state['hands_136'][player].remove(tile)
        self.round_state['rivers'][player].append(tile)
        self._add_event({'event_id': 'DISCARD', 'player': player, 'tile': tile // 4, 'is_tedashi': is_tedashi, 'is_red': tile in [16, 52, 88]})
        self._generate_opponent_turn_training_data(player, tile, next_element)
    
    def _process_meld(self, attrib):
        player = int(attrib.get('who'))
        m = int(attrib.get('m'))
        meld = Meld.decode_m(m)
        self.round_state['melds'][player].append(meld)
        tiles_to_remove = [t for t in meld.tiles if t != meld.called_tile]
        for tile in tiles_to_remove:
            if tile in self.round_state['hands_136'][player]:
                 self.round_state['hands_136'][player].remove(tile)
        self._add_event({'event_id': 'MELD', 'player': player, 'type': meld.type})
    
    def _process_riichi(self, attrib):
        player = int(attrib.get('who'))
        step = int(attrib.get('step'))
        if step == 1:
            self.round_state['is_riichi'][player] = True
            self.round_state['riichi_turn'][player] = len(self.round_state['rivers'][player])
            self._add_event({'event_id': 'RIICHI', 'player': player})
        elif step == 2:
            self.round_state['scores'][player] -= 1000
            self.round_state['riichi_sticks'] += 1

    def _process_new_dora(self, attrib):
        dora_indicator = int(attrib.get('hai'))
        self.round_state['dora_indicators'].append(dora_indicator)
        self._add_event({'event_id': 'NEW_DORA', 'dora': dora_indicator // 4})

    def _process_agari(self, attrib):
        self._add_event({'event_id': 'AGARI', 'who': int(attrib.get('who'))})

    def _process_ryukyoku(self, attrib):
        self._add_event({'event_id': 'RYUKYOKU', 'type': attrib.get('type')})

    def _create_and_add_training_point(self, choices, label):
        context = self.round_state['events'].copy()
        self.training_data.append({"context": context, "choices": choices, "label": label})

    def _generate_my_turn_training_data(self, player, next_element, win_tile):
        choices = self._get_my_turn_actions(player, win_tile)
        if len(choices) <= 1: return
        
        label = "ACTION_PASS"
        if next_element:
            tag, attrib = next_element.tag, next_element.attrib
            if tag[0] in "DEFG": label = f"DISCARD_{int(tag[1:])}"
            elif tag == "AGARI" and int(attrib.get('who')) == player: label = "ACTION_TSUMO_AGARI"
            elif tag == "REACH" and int(attrib.get('step')) == 1: label = "ACTION_RIICHI"
        
        # ラベルを、赤ドラ情報を含んだ選択肢の形式に合わせる
        if label.startswith("DISCARD_"):
            tile_136 = int(label.split('_')[1])
            is_red = tile_136 in [16, 52, 88]
            label = f"DISCARD_{tile_136 // 4}_{is_red}"
        
        # リーチは打牌を伴うため、選択肢の中から対応する打牌を探す必要がある（今回は簡略化）
        found_choice = any(c.startswith(label.split('_')[0]) for c in choices)
        if found_choice:
             self._create_and_add_training_point(choices, label)
    
    def _generate_opponent_turn_training_data(self, discarder, tile, next_element):
        actual_action_map = {}
        if next_element:
            tag, attrib = next_element.tag, next_element.attrib
            who = int(attrib.get('who', -1))
            if tag == 'N' and who != -1: actual_action_map[who] = self._meld_obj_to_action_label(Meld.decode_m(int(attrib.get('m'))))
            elif tag == 'AGARI' and who != -1: actual_action_map[who] = "ACTION_RON_AGARI"
        
        for player in range(4):
            if player == discarder: continue
            choices = self._get_opponent_turn_actions(player, discarder, tile)
            if len(choices) <= 1: continue
            label = actual_action_map.get(player, "ACTION_PASS")
            if label in choices:
                self._create_and_add_training_point(choices, label)

    def _get_my_turn_actions(self, player, win_tile):
        """【自前ロジック】自分の手番で可能な行動を洗い出す。"""
        actions = []
        hand_136 = self.round_state['hands_136'][player]
        
        # 1. 打牌
        for t in sorted(list(set(hand_136))):
            actions.append(f"DISCARD_{t // 4}_{t in [16, 52, 88]}")
        
        # 2. ツモアガリ
        if self._can_agari(hand_136):
            actions.append("ACTION_TSUMO_AGARI")
        
        # 3. リーチ
        if self.shanten_calculator.calculate_shanten(TilesConverter.to_34_array(hand_136)) == 0 and not self.round_state['is_riichi'][player]:
            # リーチ可能な打牌全てを候補として追加（今回は簡略化して1つだけ）
            actions.append("ACTION_RIICHI")

        # 4. カン
        hand_34 = TilesConverter.to_34_array(hand_136)
        if hand_34.count(4) > 0: # 暗槓
             ankan_tile = hand_34.index(4)
             actions.append(f"ACTION_CLOSED_KAN_{ankan_tile}")
        for meld in self.round_state['melds'][player]: # 加槓
            if meld.type == Meld.PON and hand_34[meld.tiles[0] // 4] == 1:
                actions.append(f"ACTION_PROMOTED_KAN_{meld.tiles[0] // 4}")

        return list(dict.fromkeys(actions))

    def _get_opponent_turn_actions(self, player, discarder, tile):
        """【自前ロジック】他家の捨て牌に対して可能な行動を洗い出す。"""
        actions = ["ACTION_PASS"]
        hand_136 = self.round_state['hands_136'][player]
        
        # 1. ロンアガリ
        temp_hand_136 = hand_136 + [tile]
        if self._can_agari(temp_hand_136):
            actions.append("ACTION_RON_AGARI")
            
        # 2. ポン
        hand_34 = TilesConverter.to_34_array(hand_136)
        if hand_34[tile // 4] >= 2:
            actions.append(f"ACTION_PUNG_{tile // 4}")

        # 3. チー
        if (discarder + 1) % 4 == player and tile // 4 < 27: # 上家からで、かつ数牌
            tile_type = tile // 4
            # 嵌張、辺張、両面のチーをチェック
            patterns = [[-2, -1], [-1, 1], [1, 2]]
            for p in patterns:
                t1 = tile_type + p[0]
                t2 = tile_type + p[1]
                if t1 >= 0 and t2 < 27 and (t1 // 9 == t2 // 9 == tile_type // 9):
                    if hand_34[t1] > 0 and hand_34[t2] > 0:
                        actions.append(f"ACTION_CHII_{t1}_{t2}")

        return actions
        
    def _can_agari(self, hand_136):
        """【自前ロジック】向聴数が-1になるかで、和了可能かを判定する。"""
        hand_34 = TilesConverter.to_34_array(hand_136)
        # 役の有無は問わず、純粋にアガリ形かを判定
        return self.shanten_calculator.calculate_shanten(hand_34, tiles_count_34=hand_34) == -1

    def _meld_obj_to_action_label(self, meld_obj):
        """Meldオブジェクトを学習データ用の文字列ラベルに変換する"""
        tile_type = meld_obj.called_tile // 4
        if meld_obj.type == Meld.CHI:
            tiles = sorted([t // 4 for t in meld_obj.tiles if t != meld_obj.called_tile])
            return f"ACTION_CHII_{tiles[0]}_{tiles[1]}"
        elif meld_obj.type == Meld.PON: return f"ACTION_PUNG_{tile_type}"
        elif meld_obj.type == Meld.KAN: return f"ACTION_DAIMINKAN_{tile_type}"
        return "ACTION_UNKNOWN"

    def run(self, filepaths):
        all_data = []
        for i, filepath in enumerate(filepaths):
            if (i + 1) % 100 == 0:
                print(f"Processing file {i+1}/{len(filepaths)}: {os.path.basename(filepath)}")
            all_data.extend(self.parse_log_file(filepath))
        return all_data

if __name__ == '__main__':
    parser = TransformerParser()
    print("TransformerParser v3.2 (Final/Self-Contained) loaded. The ultimate sensei is ready.")


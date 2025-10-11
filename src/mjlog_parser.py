# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import gzip
import numpy as np
import urllib.parse # URLデコード用のライブラリをインポート
from mahjong.shanten import Shanten

class MjlogParser:
    """
    天鳳の.mjlogファイルを解析し、AIの学習データを生成するクラス
    """
    def __init__(self):
        self.game_state = {}
        self.training_data = []
        self.shanten_calculator = Shanten()
        self.reset_game_state()

    def reset_game_state(self):
        """対局の状態を初期化する"""
        self.game_state = {
            'players': [],
            'round': 0,
            'honba': 0,
            'riichi_sticks': 0,
            'dora_indicators': [],
            'scores': [25000] * 4,
            'hands': [[] for _ in range(4)],
            'rivers': [[] for _ in range(4)],
            'is_riichi': [False] * 4,
        }
        self.training_data = []

    def parse_log_file(self, filepath):
        """単一のログファイルを解析する"""
        self.reset_game_state()
        try:
            with gzip.open(filepath, 'rb') as f:
                xml_content = f.read()
            
            log_text = xml_content.decode('utf-8').strip()
            root = ET.fromstring(f"<root>{log_text}</root>")

            for element in root.iter():
                if element.tag in ['root', 'mjloggm']:
                    continue
                self.process_tag(element)

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            return []

        return self.training_data

    def process_tag(self, element):
        """XMLタグに応じて、ゲーム状態を更新またはデータを生成する"""
        tag = element.tag
        
        if tag == 'UN':
            # URLエンコードされたプレイヤー名をデコードする
            self.game_state['players'] = [
                urllib.parse.unquote(element.attrib.get('n0', 'P0')),
                urllib.parse.unquote(element.attrib.get('n1', 'P1')),
                urllib.parse.unquote(element.attrib.get('n2', 'P2')),
                urllib.parse.unquote(element.attrib.get('n3', 'P3')),
            ]
            print(f"Players: {self.game_state['players']}")

        elif tag == 'INIT':
            self.initialize_round(element.attrib)

        elif tag[0] in "TUVW" and tag[1:].isdigit():
            self.process_draw(tag)

        elif tag[0] in "DEFG" and tag[1:].isdigit():
            self.process_discard(tag)
    
    def initialize_round(self, attrib):
        """INITタグから局情報を初期化する"""
        seed = [int(s) for s in attrib.get('seed', '0,0,0,0,0,0').split(',')]
        self.game_state['round'] = seed[0]
        self.game_state['honba'] = seed[1]
        self.game_state['riichi_sticks'] = seed[2]
        self.game_state['dora_indicators'] = [seed[5] // 4]

        self.game_state['scores'] = [int(s) for s in attrib.get('ten', '25000,25000,25000,25000').split(',')]
        
        for i in range(4):
            hand_str = attrib.get(f'hai{i}', '')
            self.game_state['hands'][i] = sorted([int(p) for p in hand_str.split(',')]) if hand_str else []
            self.game_state['rivers'][i] = []
            self.game_state['is_riichi'][i] = False
        
        round_map = ['E1','E2','E3','E4','S1','S2','S3','S4', 'W1', 'W2', 'W3', 'W4'] # 西入なども考慮
        if self.game_state['round'] < len(round_map):
            print(f"\n--- Round {round_map[self.game_state['round']]} - {self.game_state['honba']} Honba ---")

    def process_draw(self, tag):
        """ツモ処理。手牌に牌を追加する"""
        player = "TUVW".find(tag[0])
        tile = int(tag[1:])
        if player != -1:
            self.game_state['hands'][player].append(tile)
            self.game_state['hands'][player].sort()

    def process_discard(self, tag):
        """打牌処理。学習データを生成し、手牌と河を更新する"""
        player = "DEFG".find(tag[0])
        tile = int(tag[1:])
        
        if player == 0:
            feature_tensor = self.create_feature_tensor(player)
            action_label = tile // 4
            self.training_data.append((feature_tensor, action_label))

        if tile in self.game_state['hands'][player]:
            self.game_state['hands'][player].remove(tile)
        self.game_state['rivers'][player].append(tile)

    def create_feature_tensor(self, player_id):
        """現在のゲーム状態からCNN用の特徴量テンソルを生成する"""
        NUM_CHANNELS = 15
        NUM_TILE_TYPES = 34
        tensor = np.zeros((NUM_CHANNELS, NUM_TILE_TYPES), dtype=np.float32)

        # ch0: 自分の手牌
        for tile in self.game_state['hands'][player_id]:
            tensor[0, tile // 4] += 1
        
        for i in range(4):
            player_pos = (player_id + i) % 4
            if i == 0:
                for tile in self.game_state['rivers'][player_pos]:
                    tensor[1, tile // 4] = 1
                if self.game_state['is_riichi'][player_pos]:
                    tensor[3, :] = 1
            else:
                ch_offset = 4 + (i - 1) * 3
                for tile in self.game_state['rivers'][player_pos]:
                    tensor[ch_offset, tile // 4] = 1
                if self.game_state['is_riichi'][player_pos]:
                    tensor[ch_offset + 2, :] = 1

        # ch13: ドラ表示牌
        for dora_indicator in self.game_state['dora_indicators']:
             tensor[13, dora_indicator] = 1

        # ch14: シャンテン数
        hand_counts = [0] * 34
        for tile in self.game_state['hands'][player_id]:
            hand_counts[tile // 4] += 1
        
        num_tiles = sum(hand_counts)
        if num_tiles % 3 == 2:
            # 【バグ修正】chiitoi, kokushi引数を削除
            shanten_val = self.shanten_calculator.calculate_shanten(hand_counts)
            tensor[14, :] = min(shanten_val, 8) / 8.0
        
        return tensor
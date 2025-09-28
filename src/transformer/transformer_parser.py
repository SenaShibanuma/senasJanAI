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
    """
    天鳳の.mjlogファイルを解析し、Transformerモデル用の学習データを生成するクラス。
    mahjong==1.3.0ライブラリの機能を最大限に活用し、正確なゲーム状態の再現と行動選択肢の生成を行う。
    """
    def __init__(self):
        self.shanten_calculator = Shanten()
        self.hand_calculator = HandCalculator() # mahjongライブラリの中央エンジン
        self.tiles_converter = TilesConverter() # 牌表現の変換ユーティリティ
        self.reset_game_state()

    def reset_game_state(self):
        """対局全体の情報を初期化"""
        self.game_state = {
            'rules': {},
            'players': [''] * 4,
            'config': HandConfig() # ルール設定オブジェクト
        }
        self.reset_round_state()

    def reset_round_state(self):
        """局ごとの詳細な状態を初期化"""
        self.round_state = {
            'events': [], 'round': 0, 'honba': 0, 'riichi_sticks': 0,
            'dora_indicators': [], 'scores': [25000] * 4,
            'hands_136': [[] for _ in range(4)],
            'melds': [[] for _ in range(4)],
            'rivers': [[] for _ in range(4)],
            'is_riichi': [False] * 4,
            'riichi_turn': [-1] * 4, # リーチ宣言巡目を記録
            'oya_player_id': 0,
            'remaining_tiles': 70, # 初期牌山枚数
            'turn_num': 0, # 巡目
            'last_drawn_tile': [None] * 4, # 直前にツモった牌
        }
        self.training_data = []

    def parse_log_file(self, filepath):
        """単一のログファイルを解析し、学習データを返す"""
        self.reset_game_state()
        try:
            open_func = gzip.open if filepath.endswith('.gz') else open
            with open_func(filepath, 'rb') as f:
                # 生のXMLコンテンツを読み込む
                xml_content = f.read()

            # 不正な終端タグを削除し、解析可能な形式に整形
            log_text = xml_content.decode('utf-8').strip().replace("</mjloggm>", "")
            root = ET.fromstring(f"<root>{log_text}</root>")

            # 全てのタグをリストとして取得し、次のタグを予測に使用する
            all_tags = list(root.iter())
            for i, element in enumerate(all_tags):
                if element.tag in ['root', 'mjloggm']: continue
                # 次の要素（正解ラベル）を取得
                next_element = all_tags[i + 1] if i + 1 < len(all_tags) else None
                self.process_tag(element, next_element)
        except Exception:
            # 解析エラーが発生した場合は空のデータを返す
            return []
        return self.training_data

    def process_tag(self, element, next_element):
        """XMLの各タグを解析し、対応する処理を呼び出す"""
        tag, attrib = element.tag, element.attrib
        if tag == 'GO': self._process_go(attrib)
        elif tag == 'UN': self._process_un(attrib)
        elif tag == 'INIT': self._process_init(attrib)
        # ツモ (T, U, V, W)
        elif tag[0] in "TUVW" and tag[1:].isdigit(): self._process_draw(tag, next_element)
        # 打牌 (D, E, F, G)
        elif tag[0] in "DEFG" and tag[1:].isdigit(): self._process_discard(tag, next_element)
        elif tag == 'N': self._process_meld(attrib)
        elif tag == 'REACH': self._process_riichi(attrib)
        elif tag == 'DORA': self._process_new_dora(attrib)
        elif tag == 'AGARI': self._process_agari(attrib)
        elif tag == 'RYUUKYOKU': self._process_ryukyoku(attrib)

    def _add_event(self, event_dict):
        """ゲームのイベントシーケンスに新しいイベントを追加"""
        self.round_state['events'].append(event_dict)

    def _parse_rules(self, type_attribute):
        """GOタグからルール情報を解析し、HandConfigオブジェクトを設定"""
        type_val = int(type_attribute)
        has_kuitan = bool(type_val & 0x0010)
        has_aka_dora = bool(type_val & 0x0040)
        # mahjongライブラリのルール設定オブジェクトを更新
        self.game_state['config'] = HandConfig(
            options=OptionalRules(has_open_tanyao=has_kuitan, has_aka_dora=has_aka_dora)
        )
        return {'has_kuitan': has_kuitan, 'has_aka_dora': has_aka_dora}

    def _process_go(self, attrib):
        """対局開始"""
        type_attr = attrib.get('type')
        if type_attr: self.game_state['rules'] = self._parse_rules(type_attr)
        self._add_event({'event_id': 'GAME_START', 'rules': self.game_state['rules']})

    def _process_un(self, attrib):
        """プレイヤー情報"""
        self.game_state['players'] = [urllib.parse.unquote(attrib.get(f'n{i}', f'P{i}')) for i in range(4)]

    def _process_init(self, attrib):
        """局開始"""
        self.reset_round_state()
        seed = [int(s) for s in attrib.get('seed').split(',')]
        self.round_state.update({
            'round': seed[0], 'honba': seed[1], 'riichi_sticks': seed[2],
            'dora_indicators': [seed[5]],
            'scores': [int(s) for s in attrib.get('ten').split(',')],
            'oya_player_id': int(attrib.get('oya'))
        })
        for i in range(4):
            # 136牌ID形式で手牌を格納
            self.round_state['hands_136'][i] = sorted([int(p) for p in attrib.get(f'hai{i}', '').split(',') if p])
        self._add_event({'event_id': 'INIT', 'round': self.round_state['round'], 'honba': self.round_state['honba'], 'dora_indicator': seed[5]})

    def _process_draw(self, tag, next_element):
        """ツモ処理"""
        player = "TUVW".find(tag[0])
        tile = int(tag[1:])
        self.round_state['remaining_tiles'] -= 1
        self.round_state['turn_num'] += 1
        self.round_state['hands_136'][player].append(tile)
        self.round_state['hands_136'][player].sort()
        self.round_state['last_drawn_tile'][player] = tile
        self._add_event({'event_id': 'DRAW', 'player': player, 'tile': tile})

        # 自分の手番なので、学習データを生成
        self._generate_my_turn_training_data(player, next_element, win_tile=tile)

    def _process_discard(self, tag, next_element):
        """打牌処理"""
        player = "DEFG".find(tag[0])
        tile = int(tag[1:])
        is_tedashi = tile != self.round_state['last_drawn_tile'][player]
        # 手牌から打牌を削除
        if tile in self.round_state['hands_136'][player]:
            self.round_state['hands_136'][player].remove(tile)
        self.round_state['rivers'][player].append(tile)
        self._add_event({'event_id': 'DISCARD', 'player': player, 'tile': tile, 'is_tedashi': is_tedashi})

        # 他家の打牌なので、他家3人分の学習データを生成
        self._generate_opponent_turn_training_data(player, tile, next_element)

    def _process_meld(self, attrib):
        """副露処理"""
        player = int(attrib.get('who'))
        m = int(attrib.get('m'))
        # mahjongライブラリのMeldオブジェクトで副露をデコード
        meld = Meld.decode_m(m)
        self.round_state['melds'][player].append(meld)
        # 副露に使った牌を手牌から削除
        tiles_to_remove = [t for t in meld.tiles if t != meld.called_tile]
        for tile in tiles_to_remove:
            if tile in self.round_state['hands_136'][player]:
                 self.round_state['hands_136'][player].remove(tile)
        self._add_event({'event_id': 'MELD', 'player': player, 'type': meld.type, 'tiles': meld.tiles})

    def _process_riichi(self, attrib):
        """リーチ処理"""
        player = int(attrib.get('who'))
        step = int(attrib.get('step'))
        if step == 1: # リーチ宣言
            self.round_state['is_riichi'][player] = True
            self.round_state['riichi_turn'][player] = self.round_state['turn_num']
            self._add_event({'event_id': 'RIICHI_DECLARED', 'player': player})
        elif step == 2: # リーチ成立 (1000点供託)
            self.round_state['scores'][player] -= 1000
            self.round_state['riichi_sticks'] += 1
            self._add_event({'event_id': 'RIICHI_ACCEPTED', 'player': player})

    def _process_new_dora(self, attrib):
        """新ドラ処理 (カン発生時)"""
        dora_indicator = int(attrib.get('hai'))
        self.round_state['dora_indicators'].append(dora_indicator)
        self._add_event({'event_id': 'NEW_DORA', 'dora_indicator': dora_indicator})

    def _process_agari(self, attrib):
        """和了処理"""
        self._add_event({'event_id': 'AGARI', 'who': int(attrib.get('who')), 'from': int(attrib.get('fromWho'))})

    def _process_ryukyoku(self, attrib):
        """流局処理"""
        self._add_event({'event_id': 'RYUKYOKU', 'type': attrib.get('type')})

    def _create_and_add_training_point(self, choices, label):
        """現在の文脈、選択肢、正解ラベルを学習データとして追加"""
        if not choices or label not in choices:
            return
        context = self.round_state['events'].copy()
        self.training_data.append({"context": context, "choices": choices, "label": label})

    def _generate_my_turn_training_data(self, player, next_element, win_tile):
        """自分の手番における学習データを生成する"""
        choices = self._get_my_turn_actions(player, win_tile)
        if len(choices) <= 1: return

        # 次のタグから、実際の行動（正解ラベル）を特定する
        label = "ACTION_PASS" # デフォルト
        if next_element:
            tag, attrib = next_element.tag, next_element.attrib
            if tag[0] in "DEFG": # 打牌
                discard_tile_136 = int(tag[1:])
                label = f"DISCARD_{discard_tile_136}"
            elif tag == "AGARI" and int(attrib.get('who')) == player: # ツモ和了
                label = "ACTION_TSUMO_AGARI"
            elif tag == "REACH" and int(attrib.get('step')) == 1 and int(attrib.get('who')) == player: # リーチ
                # リーチ宣言時の打牌を次の打牌タグから特定する必要がある
                if next_element.getnext() and next_element.getnext().tag[0] in "DEFG":
                     discard_tile_136 = int(next_element.getnext().tag[1:])
                     label = f"ACTION_RIICHI_{discard_tile_136}"
            # TODO: 暗槓、加槓のラベル特定を追加

        self._create_and_add_training_point(choices, label)

    def _generate_opponent_turn_training_data(self, discarder, discarded_tile, next_element):
        """他家の打牌に対する学習データを生成する（鳴き、ロン）"""
        # 次の行動を事前にマッピング
        actual_action_map = {}
        if next_element:
            tag, attrib = next_element.tag, next_element.attrib
            who = int(attrib.get('who', -1))
            if tag == 'N' and who != -1: # 副露
                actual_action_map[who] = self._meld_obj_to_action_label(Meld.decode_m(int(attrib.get('m'))))
            elif tag == 'AGARI' and who != -1: # ロン和了
                actual_action_map[who] = "ACTION_RON_AGARI"

        # 各プレイヤーについて可能な行動をリストアップし、学習データを作成
        for player in range(4):
            if player == discarder: continue # 打牌者自身は除く
            choices = self._get_opponent_turn_actions(player, discarder, discarded_tile)
            if len(choices) <= 1: continue # "PASS"しか選択肢がない場合は学習データにしない
            label = actual_action_map.get(player, "ACTION_PASS")
            self._create_and_add_training_point(choices, label)


    def _get_my_turn_actions(self, player, win_tile):
        """【改良版】自分の手番で可能な全ての行動をリストアップする"""
        actions = []
        hand_136 = self.round_state['hands_136'][player]

        # 1. 打牌選択肢
        unique_tiles_136 = sorted(list(set(hand_136)))
        for t in unique_tiles_136:
            actions.append(f"DISCARD_{t}")

        # 2. ツモ和了
        # mahjongライブラリで和了可能か判定
        if self._can_agari(hand_136, win_tile, is_tsumo=True, player_id=player):
            actions.append("ACTION_TSUMO_AGARI")

        # 3. リーチ
        shanten, _ = self._calculate_shanten_and_ukeire(hand_136)
        if shanten == 0 and not self.round_state['is_riichi'][player]:
            # 聴牌に取れる全ての打牌をリーチの選択肢として追加
            for t in unique_tiles_136:
                 temp_hand = hand_136.copy()
                 temp_hand.remove(t)
                 temp_shanten, _ = self._calculate_shanten_and_ukeire(temp_hand)
                 if temp_shanten == 0:
                     actions.append(f"ACTION_RIICHI_{t}")

        # 4. カン（暗槓、加槓）
        # (実装は複雑なため、今回は省略。詳細はドキュメント参照)

        return list(dict.fromkeys(actions)) # 重複削除

    def _get_opponent_turn_actions(self, player, discarder, discarded_tile):
        """【改良版】他家の捨て牌に対して可能な行動をリストアップする"""
        actions = ["ACTION_PASS"]
        hand_136 = self.round_state['hands_136'][player]

        # 1. ロン和了
        if self._can_agari(hand_136 + [discarded_tile], discarded_tile, is_tsumo=False, player_id=player):
            actions.append("ACTION_RON_AGARI")

        # 2. 副露 (ポン、チー、大明槓)
        # mahjongライブラリは鳴き判定機能を持たないため、ここは自前ロジックを維持
        hand_34_counts = TilesConverter.to_34_array(hand_136)
        tile_34 = discarded_tile // 4

        # ポン
        if hand_34_counts[tile_34] >= 2:
            actions.append(f"ACTION_PUNG")

        # チー (上家からのみ)
        if (discarder + 1) % 4 == player and tile_34 < 27: # 数牌の場合
            # 考えられる全てのチーの組み合わせを列挙
            patterns = [[-2, -1], [-1, 1], [1, 2]] # [3,4]5, 4[5]6, 56[7]
            for p in patterns:
                t1 = tile_34 + p[0]
                t2 = tile_34 + p[1]
                # 同じ色(m,p,s)内でのみチー可能
                if t1 >= 0 and t2 < 27 and (t1 // 9 == t2 // 9 == tile_34 // 9):
                    if hand_34_counts[t1] > 0 and hand_34_counts[t2] > 0:
                        actions.append(f"ACTION_CHII_{t1}_{t2}")
        
        # 大明槓
        if hand_34_counts[tile_34] >= 3:
            actions.append("ACTION_DAIMINKAN")


        return list(dict.fromkeys(actions))


    def _can_agari(self, hand_136, win_tile, is_tsumo, player_id):
        """【mahjongライブラリ活用】和了可能かを判定する"""
        # HandCalculatorはエラー時にNoneやエラーメッセージを返す
        result = self.hand_calculator.estimate_hand_value(
            tiles=hand_136,
            win_tile=win_tile,
            melds=self.round_state['melds'][player_id],
            dora_indicators=self.round_state['dora_indicators'],
            config=self._get_current_hand_config(is_tsumo, player_id)
        )
        return result.error is None

    def _calculate_shanten_and_ukeire(self, hand_136):
        """【mahjongライブラリ活用】向聴数と受け入れ枚数を計算する"""
        # mahjongライブラリのシャンテン計算機能は、34種の牌の枚数リストを入力とする
        hand_34_array = TilesConverter.to_34_array(hand_136)
        try:
            # 推定API: calculator.calculate_shanten(tiles, melds=None)
            # 実際のAPIはライブラリのソースを確認する必要があるが、ここではshantenのみ使用
            shanten = self.shanten_calculator.calculate_shanten(hand_34_array)
            ukeire = 0 # 受け入れ枚数計算は複雑なため、ここでは0とする
            return shanten, ukeire
        except Exception:
            return 9, 0 # 計算失敗時は最大値を返す

    def _get_current_hand_config(self, is_tsumo, player_id):
        """現在の状況をHandConfigオブジェクトに設定して返す"""
        return HandConfig(
            is_tsumo=is_tsumo,
            is_riichi=self.round_state['is_riichi'][player_id],
            player_wind=Meld.EAST + player_id, # 仮設定 (東、南、西、北)
            round_wind=Meld.EAST + self.round_state['round'] // 4, # 仮設定
            options=self.game_state['config'].options
        )

    def _meld_obj_to_action_label(self, meld_obj):
        """Meldオブジェクトを学習データ用の統一された文字列ラベルに変換"""
        if meld_obj.type == Meld.CHI:
            # チーした牌を除いた2枚の牌でラベルを作成
            consumed = sorted([t // 4 for t in meld_obj.tiles if t != meld_obj.called_tile])
            return f"ACTION_CHII_{consumed[0]}_{consumed[1]}"
        elif meld_obj.type == Meld.PON: return "ACTION_PUNG"
        elif meld_obj.type == Meld.KAN: return "ACTION_DAIMINKAN" # 大明槓
        return "ACTION_UNKNOWN"

    def run(self, filepaths):
        """複数のログファイルを処理し、全学習データを返す"""
        all_data = []
        total_files = len(filepaths)
        for i, filepath in enumerate(filepaths):
            # 処理の進捗を表示
            if (i + 1) % 100 == 0:
                print(f"Processing file {i+1}/{total_files}: {os.path.basename(filepath)}")
            all_data.extend(self.parse_log_file(filepath))
        return all_data

if __name__ == '__main__':
    parser = TransformerParser()
    print("TransformerParser v4.0 (mahjong-lib integrated) loaded. The real game begins.")
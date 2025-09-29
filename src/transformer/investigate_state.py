# -*- coding: utf-8 -*-
import os
from pprint import pprint
from mahjong.tile import TilesConverter
from mahjong.meld import Meld

from src.transformer.transformer_parser import TransformerParser

class GameStateSimulator:
    """
    mjaiのイベントを元に、簡易的なゲーム状態をシミュレートするクラス（完成版）
    """
    def __init__(self):
        self.parser = TransformerParser()
        self.state = {}
        self.pai_map = self._create_pai_map()

    def _create_pai_map(self):
        """牌のIDと文字列表現の対応表を作成"""
        suits = ['m', 'p', 's']
        honors = ['E', 'S', 'W', 'N', 'H', 'G', 'C']
        mapping = {}
        for i in range(136):
            tile_34 = i // 4
            if tile_34 < 27:
                num = (tile_34 % 9) + 1
                suit = suits[tile_34 // 9]
                mapping[i] = f"{num}{suit}"
            else:
                mapping[i] = honors[tile_34 - 27]
        return mapping

    def _convert_hand_to_str(self, hand_136):
        """136形式の手牌をソートされた文字列に変換"""
        m, p, s, z = [], [], [], []
        jihai_order = {108: 'E', 112: 'S', 116: 'W', 120: 'N', 124: 'H', 128: 'G', 132: 'C'}
        for tile_id in sorted(hand_136):
            tile_34 = tile_id // 4
            if tile_34 < 9: m.append(str((tile_34 % 9) + 1))
            elif tile_34 < 18: p.append(str((tile_34 % 9) + 1))
            elif tile_34 < 27: s.append(str((tile_34 % 9) + 1))
            else:
                base_id = tile_34 * 4
                z.append((base_id, jihai_order.get(base_id, '')))
        z.sort()
        z_str = "".join([val for key, val in z])
        hand_parts = []
        if m: hand_parts.append("".join(m) + "m")
        if p: hand_parts.append("".join(p) + "p")
        if s: hand_parts.append("".join(s) + "s")
        if z_str: hand_parts.append(z_str)
        return "".join(hand_parts)

    def process_event(self, event):
        """イベントを処理して状態を更新"""
        event_type = event.get('type')
        if event_type == 'start_game':
            self.state = {'hands': [[] for _ in range(4)]}
        elif event_type == 'start_kyoku':
            self.state['hands'] = [sorted(h) for h in event['haipai']]
        elif event_type == 'tsumo':
            actor, pai = event['actor'], event['pai']
            self.state['hands'][actor].append(pai)
            self.state['hands'][actor].sort()
        elif event_type == 'dahai':
            actor, pai = event['actor'], event['pai']
            if pai in self.state['hands'][actor]:
                self.state['hands'][actor].remove(pai)
        elif event_type == 'naki':
            actor, m = event['actor'], event['m']
            meld = self.parser._decode_meld(m)
            if meld.type:
                # ▼▼▼【最終修正】ここから▼▼▼
                # mahjongライブラリの仕様:
                # 「加槓(kakan)」という行為は Meld.CHANKAN で表現される。
                # (役の「槍槓(chankan)」と定数を共有しているため紛らわしい)
                if meld.type == Meld.CHANKAN:
                    # 加槓の場合、手牌から1枚だけ消える
                    chakan_tile = meld.called_tile
                    tile_to_remove_34 = chakan_tile // 4
                    # 手牌から同じ種類の牌を1枚探して削除
                    for tile_in_hand in self.state['hands'][actor]:
                        if tile_in_hand // 4 == tile_to_remove_34:
                            self.state['hands'][actor].remove(tile_in_hand)
                            break
                else:
                    # それ以外の鳴き(チー,ポン,大明槓)は鳴きに使った手牌を除く
                    tiles_to_remove = [t for t in meld.tiles if t != meld.called_tile]
                    for tile in tiles_to_remove:
                        if tile in self.state['hands'][actor]:
                            self.state['hands'][actor].remove(tile)
                # ▲▲▲【最終修正】ここまで▲▲▲
            else:
                print(f"[SIMULATOR_WARNING] Failed to decode meld data m={m} for actor {actor}")

    def print_state(self):
        """現在の状態を分かりやすく表示"""
        print("--- Simulator State (After Event) ---")
        for i in range(4):
            hand_str = self._convert_hand_to_str(self.state['hands'][i])
            print(f"  P{i} Hand ({len(self.state['hands'][i])}枚): {hand_str}")
        print("------------------------------------------\n")

def main(log_path):
    """メイン処理"""
    print("=" * 50)
    print(f"INVESTIGATION REPORT FOR: {os.path.basename(log_path)}")
    print("=" * 50)
    
    parser = TransformerParser()
    events = parser.generate_simple_events(log_path)

    if not events or len(events) <= 2:
        print("\n[ERROR] Failed to parse events from the log file using TransformerParser.")
        return

    print(f"\nSuccessfully parsed. Found {len(events)} events. Starting simulation...")
    
    simulator = GameStateSimulator()

    for i, event in enumerate(events):
        print(f"==================== Event {i+1} ====================")
        pprint(event)
        print("")
        simulator.process_event(event)
        simulator.print_state()

if __name__ == '__main__':
    try:
        DATA_DIR = '/content/data'
        TEST_FILE = '20061028gm-0001-0000-134c5173&tw=0.mjlog' 
        test_file_path = os.path.join(DATA_DIR, TEST_FILE)

        if not os.path.exists(test_file_path):
            try:
                test_file_path = next(f.path for f in os.scandir(DATA_DIR) if f.is_file())
            except StopIteration:
                print(f"No log files found in {DATA_DIR}.")
                exit()
        main(test_file_path)
    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()


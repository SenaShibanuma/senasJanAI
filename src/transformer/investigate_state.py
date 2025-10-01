# -*- coding: utf-8 -*-
import os
import copy
import gzip
from pprint import pprint
from mahjong.tile import TilesConverter
from mahjong.meld import Meld

# 最新のv5パーサーを参照
from src.transformer.transformer_parser import TransformerParser

def get_raw_log_context(raw_log_text, event):
    """
    イベント情報から検索文字列を生成し、rawログ内の該当箇所とその前の5タグを抽出する。
    """
    try:
        search_str = ""
        tag_type = event.get('type')

        # イベントタイプに応じて、ログ内でユニークに特定できる文字列を生成
        if tag_type == 'naki':
            search_str = f'm="{event.get("m")}"'
        elif tag_type == 'dahai':
            actor_char = "DEFG"[event.get('actor')]
            search_str = f'<{actor_char}{event.get("pai")}'
        elif tag_type == 'tsumo':
            actor_char = "TUVW"[event.get('actor')]
            search_str = f'<{actor_char}{event.get("pai")}'
        elif tag_type == 'start_kyoku':
            haipai_str = ",".join(map(str, event.get("haipai")[0]))
            search_str = f'hai0="{haipai_str}"'
        else:
            return "[Context search not supported for this event type]"

        pos = raw_log_text.find(search_str)
        if pos == -1:
            return f"[Could not find event in raw log. Search string: {search_str}]"

        # エラー箇所から遡って、開始タグを6つ見つける（エラー箇所＋その前の5タグ）
        start_pos = pos
        for _ in range(6):
            temp_pos = raw_log_text.rfind('<', 0, start_pos)
            if temp_pos != -1:
                start_pos = temp_pos
            else:
                start_pos = 0
                break
        
        # エラータグの終了位置を見つける
        end_pos = raw_log_text.find('>', pos)
        if end_pos == -1:
            end_pos = pos + len(search_str) + 20 # 見つからない場合のフォールバック

        return raw_log_text[start_pos : end_pos + 1].strip()

    except Exception as e:
        return f"[Failed to extract raw log context: {e}]"


class GameStateSimulator:
    """
    mjaiのイベントを元に、簡易的なゲーム状態をシミュレートするクラス。
    """
    def __init__(self):
        self.parser = TransformerParser()
        self.state = {}
        self.pai_map = self._create_pai_map()

    def _create_pai_map(self):
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

    def process_event(self, event, last_discard):
        event_type = event.get('type')
        try:
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
                else:
                    return False, f"Dahai error: Player {actor} tried to discard tile {pai}, which is not in their hand."
            elif event_type == 'naki':
                actor, m = event['actor'], event['m']
                meld = self.parser._decode_meld(m)
                if not meld.type:
                    return False, f"Failed to decode meld data m={m} for actor {actor}"
                
                meld.called_tile = last_discard # 直前の捨て牌を鳴き牌として設定

                if meld.type == Meld.KAN: # Ankan
                    kan_tile_34 = meld.tiles[0] // 4
                    removed_count = 0
                    for tile_in_hand in reversed(self.state['hands'][actor][:]):
                        if tile_in_hand // 4 == kan_tile_34 and removed_count < 4:
                            self.state['hands'][actor].remove(tile_in_hand)
                            removed_count += 1
                    if removed_count != 4: return False, f"Ankan error: Could not find 4 matching tiles for {kan_tile_34}."
                
                elif meld.type == Meld.CHANKAN: # Kakan
                    chakan_tile_34 = meld.called_tile // 4
                    found = False
                    for tile_in_hand in reversed(self.state['hands'][actor][:]):
                        if tile_in_hand // 4 == chakan_tile_34:
                            self.state['hands'][actor].remove(tile_in_hand)
                            found = True; break
                    if not found: return False, f"Chakan error: Could not find tile {chakan_tile_34}."
                
                else: # Chi, Pon, Daiminkan
                    tiles_to_remove = [t for t in meld.tiles if t // 4 != meld.called_tile // 4]
                    for r_tile in tiles_to_remove:
                        found = False
                        for h_tile in self.state['hands'][actor]:
                            if h_tile // 4 == r_tile // 4:
                                self.state['hands'][actor].remove(h_tile)
                                found = True; break
                        if not found: return False, f"Naki error: Could not find tile {r_tile} to remove from player {actor}'s hand."
            return True, None
        except Exception as e:
            return False, f"An unexpected error occurred: {e}"

    def print_state(self, state_to_print=None):
        state = state_to_print if state_to_print is not None else self.state
        if not state or 'hands' not in state or not state['hands']:
             print("  [No valid state to print]")
             return
             
        print("--- Simulator State ---")
        for i in range(4):
            hand_136 = state['hands'][i]
            hand_str = self._convert_hand_to_str(hand_136)
            print(f"  P{i} Hand ({len(hand_136)}枚): {hand_str}")
        print("------------------------------------------")

def main(log_path):
    raw_log_text = ""
    try:
        try:
            with gzip.open(log_path, 'rb') as f: raw_log_text = f.read().decode('utf-8')
        except (gzip.BadGzipFile, OSError):
            with open(log_path, 'r', encoding='utf-8') as f: raw_log_text = f.read()
    except Exception as e:
        print(f"Error reading log file for context: {e}")

    parser = TransformerParser()
    events = parser.generate_simple_events(log_path)

    if not events or len(events) <= 2:
        print(f"\n[ERROR] Failed to parse any events from: {os.path.basename(log_path)}")
        return

    simulator = GameStateSimulator()
    last_discarded_tile = None
    
    for i, event in enumerate(events):
        state_before_event = copy.deepcopy(simulator.state)
        
        success, error_message = simulator.process_event(event, last_discarded_tile)

        if event.get('type') == 'dahai':
            last_discarded_tile = event.get('pai')
        
        if not success:
            print("\n" + "="*25 + " SIMULATION ERROR! " + "="*25)
            print(f"File: {os.path.basename(log_path)}")
            print(f"Error at Event {i+1}: {error_message}\n")
            
            print("--- State BEFORE The Failed Event ---")
            simulator.print_state(state_to_print=state_before_event)
            
            print("\n--- Failed Event Details ---")
            pprint(event)

            if raw_log_text:
                print("\n--- Raw .mjlog Context (around the error) ---")
                context = get_raw_log_context(raw_log_text, event)
                print(context)
            
            print("="*67 + "\n")
            return

    print(f"--- SUCCESS: Simulation for {os.path.basename(log_path)} completed without errors. ---")


if __name__ == '__main__':
    import sys
    try:
        if len(sys.argv) > 1:
            test_file_path = sys.argv[1]
            if os.path.exists(test_file_path):
                 main(test_file_path)
            else:
                print(f"Error: File not found at {test_file_path}")
        else:
            print("Usage: python -m src.transformer.investigate_state <path_to_log_file>")

    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()


# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import gzip
import urllib.parse
import numpy as np
import os
import random
from glob import glob
from sklearn.model_selection import train_test_split
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.meld import Meld

# --- 定数定義 ---
EVENT_VECTOR_SIZE = 100
CHOICE_VECTOR_SIZE = 100
MAX_CONTEXT_LENGTH = 150
MAX_CHOICES = 50

class TenhouDataGenerator:
    def __init__(self):
        self.shanten_calculator = Shanten()
        self.reset_game_state()

    def reset_game_state(self):
        self.game_state = {}
        self.reset_round_state()

    def reset_round_state(self):
        self.round_state = {
            'events': [], 'round': 0, 'honba': 0, 'riichi_sticks': 0,
            'dora_indicators': [], 'scores': [25000] * 4,
            'hands_136': [[] for _ in range(4)], 'rivers_136': [[] for _ in range(4)],
            'melds': [[] for _ in range(4)], 'is_riichi': [False] * 4,
            'oya_player_id': 0, 'bakaze': 27, 'jikaze': [27, 28, 29, 30],
            'turn': 0,
        }
        self.training_samples = []
        self.pending_sample_data = None

    def parse_log_file(self, filepath):
        self.reset_game_state()
        try:
            root_elements = self._read_and_parse_xml(filepath)
            if not root_elements: return []
            for element in root_elements:
                self.process_tag(element)
        except Exception as e:
            # print(f"Error processing file {filepath}: {e}")
            pass
        return self.training_samples

    def _read_and_parse_xml(self, filepath):
        try:
            with gzip.open(filepath, 'rb') as f: xml_content = f.read()
        except (gzip.BadGzipFile, OSError):
            with open(filepath, 'rb') as f: xml_content = f.read()
        log_text = xml_content.decode('utf-8').strip()
        repaired_text = f"<dummytoplevel>{log_text}</dummytoplevel>"
        root = ET.fromstring(repaired_text)
        mjloggm_tag = root.find('mjloggm')
        return list(mjloggm_tag) if mjloggm_tag is not None else list(root)

    def process_tag(self, element):
        tag, attrib = element.tag, element.attrib

        if tag == 'INIT':
            self._process_init(attrib)
        elif tag[0] in "TUVW" and tag[1:].isdigit():
            self._process_draw(tag)
        elif tag[0] in "DEFG" and tag[1:].isdigit():
            self._process_discard(tag)
        elif tag == 'DORA':
            self.round_state['dora_indicators'].append(int(attrib.get('hai')))
            self._add_event('DORA', player=-1, tile=int(attrib.get('hai')))
        elif tag == 'REACH':
            if int(attrib.get('step')) == 2:
                player = int(attrib.get('who'))
                self.round_state['is_riichi'][player] = True
                self._add_event('RIICHI', player=player)
        elif tag in ['AGARI', 'RYUUKYOKU']:
            # End of round, clear pending data
            self.pending_sample_data = None

    def _add_event(self, event_type, player, tile=None):
        event = {
            'type': event_type, 'player': player, 'tile': tile,
            'turn': self.round_state['turn']
        }
        self.round_state['events'].append(event)

    def _process_init(self, attrib):
        self.reset_round_state()
        seed = [int(s) for s in attrib.get('seed').split(',')]
        oya = int(attrib.get('oya'))
        self.round_state.update({
            'round': seed[0], 'honba': seed[1], 'riichi_sticks': seed[2],
            'dora_indicators': [seed[5]], 'oya_player_id': oya,
            'scores': [int(s) for s in attrib.get('ten').split(',')],
            'bakaze': 27 + seed[0] // 4,
            'jikaze': [(27 + (i - oya + 4) % 4) for i in range(4)],
            'turn': 0
        })
        for i in range(4):
            hai_str = attrib.get(f'hai{i}', '')
            if hai_str:
                self.round_state['hands_136'][i] = sorted([int(p) for p in hai_str.split(',')])
        self._add_event('INIT', player=-1)

    def _process_draw(self, tag):
        player = "TUVW".find(tag[0])
        tile = int(tag[1:])
        if player == -1: return

        if player == 0: # For simplicity, only learn from player 0
            self.round_state['turn'] += 1

        self.round_state['hands_136'][player].append(tile)
        self.round_state['hands_136'][player].sort()
        self._add_event('DRAW', player=player, tile=tile)

        hand = self.round_state['hands_136'][player]
        if len(hand) % 3 == 2: # Check for valid hand for discard
            choices = sorted(list(set(hand)))
            if 0 < len(choices) <= MAX_CHOICES:
                self.pending_sample_data = {
                    'player_pov': player,
                    'events': self.round_state['events'].copy(),
                    'hand': hand.copy(),
                    'choices': choices
                }

    def _process_discard(self, tag):
        player = "DEFG".find(tag[0])
        tile = int(tag[1:])
        if player == -1: return

        if self.pending_sample_data and self.pending_sample_data['player_pov'] == player:
            self._generate_training_sample(tile)
        
        self.pending_sample_data = None # Clear after use or if player doesn't match

        if tile in self.round_state['hands_136'][player]:
            self.round_state['hands_136'][player].remove(tile)
        self.round_state['rivers_136'][player].append(tile)
        self._add_event('DISCARD', player=player, tile=tile)

    def _generate_training_sample(self, actual_discard_tile):
        data = self.pending_sample_data
        player_pov = data['player_pov']
        choices = data['choices']
        
        label_index = -1
        for i, choice_tile in enumerate(choices):
            if choice_tile // 4 == actual_discard_tile // 4:
                # Find the first instance of the tile type
                label_index = i
                break
        
        # If the discarded tile is not in the choices for some reason, skip
        if label_index == -1: return

        # 1. Context Vector
        context_vectors = [self._create_event_vector(e) for e in data['events']]
        context_input = np.zeros((MAX_CONTEXT_LENGTH, EVENT_VECTOR_SIZE), dtype=np.float32)
        context_len = min(len(context_vectors), MAX_CONTEXT_LENGTH)
        if context_len > 0:
            context_input[:context_len] = context_vectors[-context_len:]

        # 2. Choice Vectors
        choice_vectors = [self._create_choice_vector(t, data['hand'], player_pov) for t in choices]
        choices_input = np.zeros((MAX_CHOICES, CHOICE_VECTOR_SIZE), dtype=np.float32)
        choices_input[:len(choice_vectors)] = choice_vectors

        # 3. Mask and Label
        mask_input = np.zeros(MAX_CHOICES, dtype=np.float32)
        mask_input[:len(choices)] = 1.0
        labels = np.zeros(MAX_CHOICES, dtype=np.float32)
        labels[label_index] = 1.0

        self.training_samples.append({
            'context_input': context_input,
            'choices_input': choices_input,
            'mask_input': mask_input,
            'label': labels
        })

    def _create_event_vector(self, event):
        vec = np.zeros(EVENT_VECTOR_SIZE, dtype=np.float32)
        event_type_map = {'INIT': 0, 'DRAW': 1, 'DISCARD': 2, 'RIICHI': 3, 'DORA': 4}
        if event['type'] in event_type_map: vec[event_type_map[event['type']]] = 1.0
        if event['player'] != -1: vec[5 + event['player']] = 1.0
        if event['tile'] is not None: vec[9 + (event['tile'] // 4)] = 1.0
        for dora_indicator in self.round_state['dora_indicators']: vec[43 + (dora_indicator // 4)] = 1.0
        for i in range(4): vec[77 + i] = self.round_state['scores'][i] / 50000.0
        for i in range(4):
            if self.round_state['is_riichi'][i]: vec[81 + i] = 1.0
        vec[85] = self.round_state['turn'] / 35.0
        vec[86] = self.round_state['round'] / 12.0
        vec[87] = self.round_state['honba'] / 5.0
        vec[88] = self.round_state['riichi_sticks'] / 4.0
        for i in range(4): vec[89 + i] = (self.round_state['jikaze'][i] - 27) / 4.0
        vec[93] = (self.round_state['bakaze'] - 27) / 4.0
        return vec

    def _create_choice_vector(self, choice_tile, hand, player_pov):
        vec = np.zeros(CHOICE_VECTOR_SIZE, dtype=np.float32)
        choice_tile_34 = choice_tile // 4
        vec[choice_tile_34] = 1.0
        # is_aka, is_dora
        dora_tiles_34 = [(dora // 4) for dora in self.round_state['dora_indicators']]
        if choice_tile_34 in dora_tiles_34: vec[35] = 1.0
        temp_hand = hand.copy()
        temp_hand.remove(choice_tile)
        try:
            hand_34 = TilesConverter.to_34_array(temp_hand)
            shanten = self.shanten_calculator.calculate_shanten(hand_34)
            shanten_idx = min(shanten + 1, 8)
            vec[36 + shanten_idx] = 1.0
        except: vec[36 + 8] = 1.0
        return vec

def preprocess(raw_logs_dir, processed_dir, test_size, validation_size):
    print("Finding log files...")
    filepaths = glob(os.path.join(raw_logs_dir, '*.mjlog'))
    if not filepaths: print(f"No .mjlog files found in {raw_logs_dir}"); return
    print(f"Found {len(filepaths)} log files.")

    if len(filepaths) < 3:
        print("Not enough files to create train/validation/test splits. Need at least 3 files.")
        return

    train_files, test_files = train_test_split(filepaths, test_size=test_size, random_state=42)
    val_split_size = validation_size / (1.0 - test_size)
    if len(train_files) > 1:
        train_files, val_files = train_test_split(train_files, test_size=val_split_size, random_state=42)
    else:
        val_files = []

    datasets = {'train': train_files, 'validation': val_files, 'test': test_files}
    generator = TenhouDataGenerator()

    for name, files in datasets.items():
        if not files: continue
        print(f"\nProcessing {name} dataset ({len(files)} files)...")
        all_samples = []
        for i, f in enumerate(files):
            samples = generator.parse_log_file(f)
            all_samples.extend(samples)
        
        if not all_samples: print(f"No training samples generated for {name} dataset. Skipping."); continue

        print(f"Generated {len(all_samples)} samples for {name} dataset.")
        context_inputs = np.array([s['context_input'] for s in all_samples])
        choices_inputs = np.array([s['choices_input'] for s in all_samples])
        mask_inputs = np.array([s['mask_input'] for s in all_samples])
        labels = np.array([s['label'] for s in all_samples])

        output_path = os.path.join(processed_dir, f'{name}.npz')
        np.savez_compressed(output_path, context_input=context_inputs, choices_input=choices_inputs, mask_input=mask_inputs, labels=labels)
        print(f"Saved {name} dataset to {output_path}")

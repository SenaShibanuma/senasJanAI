# SenasJanAI - 麻雀AI開発プロジェクト

<<<<<<< HEAD
これは、天鳳の牌譜データを活用して最強の麻雀AIを開発するプロジェクトの記録です。

## プロジェクトの目標

PCでプレイされている麻雀ゲームの画面をリアルタイムに認識し、AIが最適な打牌を提案するツールの開発を目指します。最終的には、人間のトッププロの思考を模倣し、さらに自己対戦（強化学習）によって人間を超える独自の戦略を発見するAIを構築します。

## 現在の進捗

-   [x] **Phase 0: 計画** - プロジェクト全体の技術選定と開発計画の策定
-   [x] **Phase 1: MVP開発** - 牌譜パーサーを実装し、シンプルなAIモデル(v1)を構築
-   [x] **Phase 2.1: 特徴量追加** - 河、ドラ、リーチ状況など、盤面全体の情報をAIに与えたモデル(v2)を構築
-   [x] **Phase 2.2: シャンテン数追加** - 「アガリまでの距離」をAIに教え、戦略的な判断の基礎を導入
-   [x] **Phase 2.3: モデル構造の強化** - AIの「脳」をより深くし、高度な情報を処理できるモデル(v3)を構築
-   [ ] **Phase 2.4: 最終学習** - 全プレイヤーのデータを学習させ、CNNモデルの性能を最大化する **(現在ここ -> New!)**
-   [ ] **Phase 3: 複数判断能力の実装** - 打牌選択だけでなく、リーチ・副露判断AIを開発
-   [ ] **Phase 4: 自己対戦（強化学習）** - AI同士を対戦させ、人間を超える戦略を発見させる
-   [ ] **Phase 5: Transformerモデルへの移行** - 時間的文脈を理解する最先端モデルを導入

## 各スクリプトの役割

-   `src/mjlog_parser.py`: 天鳳の`.mjlog`ファイルを解析し、AIの学習用データセットを生成します。
-   `src/train_model.py`: `mjlog_parser.py`を使ってデータセットを準備し、AIモデルの学習と保存を行います。
-   `src/predict.py`: 学習済みのAIモデルを読み込み、指定された手牌や状況に対して最適な打牌を予測・提案します。

## 実行方法

1.  **データ準備**: `data`フォルダを作成し、解析したい`.mjlog`ファイルを入れます。
2.  **ライブラリインストール**: `pip install tensorflow mahjong scikit-learn`
3.  **AIの学習**: `python -m src.train_model`
4.  **AIによる予測**: `python -m src.predict`
```eof
```python:天鳳牌譜パーサー (最終版):senasJanAI/src/mjlog_parser.py
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import gzip
import numpy as np
import urllib.parse
from mahjong.shanten import Shanten
import os

class MjlogParser:
    """
    天鳳の.mjlogファイルを解析し、AIの学習データを生成するクラス (最終版)
    """
    def __init__(self):
        self.shanten_calculator = Shanten()
        self.reset_game_state()

    def reset_game_state(self):
        """対局の状態を初期化する"""
        self.game_state = {
            'players': [], 'round': 0, 'honba': 0, 'riichi_sticks': 0,
            'dora_indicators': [], 'scores': [25000] * 4,
            'hands': [[] for _ in range(4)], 'rivers': [[] for _ in range(4)],
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
                if element.tag in ['root', 'mjloggm']: continue
                self.process_tag(element)
        except Exception as e:
            # print(f"Error processing file {os.path.basename(filepath)}: {e}")
            return []
        return self.training_data

    def process_tag(self, element):
        """XMLタグに応じて処理を振り分ける"""
        tag, attrib = element.tag, element.attrib
        if tag == 'UN':
            self.game_state['players'] = [urllib.parse.unquote(attrib.get(f'n{i}', f'P{i}')) for i in range(4)]
        elif tag == 'INIT':
            self.initialize_round(attrib)
        elif tag[0] in "TUVW" and tag[1:].isdigit():
            self.process_draw(tag)
        elif tag[0] in "DEFG" and tag[1:].isdigit():
            self.process_discard(tag)

    def initialize_round(self, attrib):
        """INITタグから局情報を初期化"""
        seed = [int(s) for s in attrib.get('seed', '0,0,0,0,0,0').split(',')]
        self.game_state.update({
            'round': seed[0], 'honba': seed[1], 'riichi_sticks': seed[2],
            'dora_indicators': [seed[5] // 4],
            'scores': [int(s) for s in attrib.get('ten', '25000,25000,25000,25000').split(',')]
        })
        for i in range(4):
            hand_str = attrib.get(f'hai{i}', '')
            self.game_state['hands'][i] = sorted([int(p) for p in hand_str.split(',')]) if hand_str else []
            self.game_state['rivers'][i] = []
            self.game_state['is_riichi'][i] = False

    def process_draw(self, tag):
        """ツモ処理"""
        player = "TUVW".find(tag[0])
        tile = int(tag[1:])
        if player != -1: self.game_state['hands'][player].append(tile)

    def process_discard(self, tag):
        """打牌処理"""
        player = "DEFG".find(tag[0])
        tile = int(tag[1:])
        
        # 【最終変更点】全プレイヤーの打牌を学習データにする
        feature_tensor = self.create_feature_tensor(player)
        action_label = tile // 4
        self.training_data.append((feature_tensor, action_label))

        # 手牌から捨て牌を削除 (ツモ切りなどで手牌にない場合も考慮)
        try:
            self.game_state['hands'][player].remove(tile)
        except ValueError:
            pass # 手牌にない場合は何もしない
        self.game_state['rivers'][player].append(tile)

    def create_feature_tensor(self, player_id):
        """CNN用の特徴量テンソルを生成"""
        NUM_CHANNELS, NUM_TILE_TYPES = 15, 34
        tensor = np.zeros((NUM_CHANNELS, NUM_TILE_TYPES), dtype=np.float32)

        # ch0: 自分の手牌
        for tile in self.game_state['hands'][player_id]:
            tensor[0, tile // 4] += 1
        
        for i in range(4):
            p_idx = (player_id + i) % 4
            if i == 0: # 自分
                for tile in self.game_state['rivers'][p_idx]: tensor[1, tile // 4] = 1
                if self.game_state['is_riichi'][p_idx]: tensor[3, :] = 1
            else: # 他家
                offset = 4 + (i - 1) * 3
                for tile in self.game_state['rivers'][p_idx]: tensor[offset, tile // 4] = 1
                if self.game_state['is_riichi'][p_idx]: tensor[offset + 2, :] = 1

        # ch13: ドラ表示牌
        for dora in self.game_state['dora_indicators']: tensor[13, dora] = 1

        # ch14: シャンテン数
        hand_counts = np.bincount([t // 4 for t in self.game_state['hands'][player_id]], minlength=NUM_TILE_TYPES)
        if sum(hand_counts) % 3 == 2:
            shanten = self.shanten_calculator.calculate_shanten(hand_counts.tolist())
            tensor[14, :] = min(shanten, 8) / 8.0
            
        return tensor
```eof
```python:AIモデルの学習スクリプト (最終版):senasJanAI/src/train_model.py
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from src.mjlog_parser import MjlogParser

def prepare_dataset(data_dir='data', processed_data_dir='processed_data'):
    """データセットを準備"""
    dataset_path = os.path.join(processed_data_dir, 'training_dataset_final.npz')
    
    if os.path.exists(dataset_path):
        print(f"Loading final dataset from {dataset_path}...")
        data = np.load(dataset_path)
        return data['features'], data['labels']

    print(f"Final dataset not found. Generating from logs in {data_dir}...")
    os.makedirs(processed_data_dir, exist_ok=True)
    
    log_files = glob.glob(os.path.join(data_dir, '**', '*.mjlog'), recursive=True)
    if not log_files: return None, None
    print(f"Found {len(log_files)} log files.")
    
    parser = MjlogParser()
    all_data = []
    
    for i, file_path in enumerate(log_files):
        print(f"\rProcessing file {i+1}/{len(log_files)}...", end="")
        data = parser.parse_log_file(file_path)
        if data: all_data.extend(data)
    print("\nFile processing finished.")

    if not all_data: return None, None

    features = np.array([item[0] for item in all_data])
    labels = np.array([item[1] for item in all_data])
    
    print(f"Saving final dataset to {dataset_path}...")
    np.savez_compressed(dataset_path, features=features, labels=labels)
    return features, labels

def build_cnn_model_v3(input_shape):
    """強化版CNNモデル(v3)を構築"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(34, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    """メイン実行関数"""
    print("--- Step 1: Preparing final dataset... ---")
    features, labels = prepare_dataset()
    if features is None: return

    print(f"\nData ready. Total samples: {len(features)}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.1, random_state=42
    )

    print("\n--- Step 2: Building the final CNN model (v3)... ---")
    model = build_cnn_model_v3(input_shape=(features.shape[1], features.shape[2]))
    model.summary()

    print("\n--- Step 3: Final training... ---")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,  #【最終変更点】エポック数を50に増やす
        batch_size=512, # バッチサイズを大きくして学習を安定化・高速化
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping]
    )

    print("\n--- Step 4: Saving final model... ---")
    model_path = os.path.join('models', 'senas_jan_ai_v3_final.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
```eof
```python:AIによる打牌予測スクリプト (最終版):senasJanAI/src/predict.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from mahjong.shanten import Shanten

# --- 定数と牌の変換用データ ---
TILE_TYPES = [
    '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
    '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
    '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
    'E', 'S', 'W', 'N', 'H', 'G', 'C'
]
TILE_MAP = {s: i for i, s in enumerate(TILE_TYPES)}

def parse_hand_string(hand_str):
    tiles = []
    current_suit = ''
    # 文字列を逆から読むことで、"123m"のような表記に対応
    for char in reversed(hand_str.lower()):
        if char.isalpha():
            current_suit = char
            # 字牌の処理
            if current_suit in 'eswnhgc':
                tile_name = current_suit.upper()
                if tile_name in TILE_MAP:
                    tiles.append(TILE_MAP[tile_name])
        elif char.isdigit():
            # 数牌の処理
            if current_suit in 'mps':
                tile_name = char + current_suit
                if tile_name in TILE_TYPES:
                    tiles.append(TILE_MAP[tile_name])
    tiles.reverse() # 最後に順番を元に戻す
    return tiles

def create_feature_tensor_for_prediction(hand_tiles, dora_indicators, rivers, is_riichi_list):
    """予測用の盤面状態から入力テンソルを生成する"""
    NUM_CHANNELS, NUM_TILE_TYPES = 15, 34
    tensor = np.zeros((NUM_CHANNELS, NUM_TILE_TYPES), dtype=np.float32)

    # ch0: 自分の手牌
    for tile_type in hand_tiles:
        tensor[0, tile_type] += 1
    
    # ch1, ch3: 自分の河とリーチ状態
    for tile_type in rivers[0]: tensor[1, tile_type] = 1
    if is_riichi_list[0]: tensor[3, :] = 1

    # 他家3人の情報
    for i in range(1, 4):
        offset = 4 + (i - 1) * 3
        for tile_type in rivers[i]: tensor[offset, tile_type] = 1
        if is_riichi_list[i]: tensor[offset + 2, :] = 1

    # ch13: ドラ表示牌
    for tile_type in dora_indicators: tensor[13, tile_type] = 1
    
    # ch14: シャンテン数
    shanten_calculator = Shanten()
    hand_counts = np.bincount(hand_tiles, minlength=NUM_TILE_TYPES)
    if sum(hand_counts) % 3 == 2:
        shanten = shanten_calculator.calculate_shanten(hand_counts.tolist())
        tensor[14, :] = min(shanten, 8) / 8.0
             
    return tensor

def main():
    """メインの実行関数"""
    model_path = os.path.join('models', 'senas_jan_ai_v3_final.keras')
    print("--- Loading final trained model (v3)... ---")
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return

    model = tf.keras.models.load_model(model_path)

    # --- 予測したい状況をここに設定 ---
    test_hand_str = "234m555p678sWWNNC" # 例: 七対子のイーシャンテン
    dora_str = "S"
    rivers_str = ["", "W", "1m2m", "9s"]
    is_riichi_list = [False, False, False, False]
    
    print(f"\nAnalyzing
=======
天鳳の牌譜データを活用して最強の麻雀AIを開発するプロジェクトの記録です。

---

## プロジェクトの目標

- PCでプレイされている麻雀ゲームの画面をリアルタイムに認識し、AIが最適な打牌を提案するツールの開発を目指します。
- 人間のトッププロの思考を模倣し、自己対戦（強化学習）によって人間を超える独自の戦略を発見するAIを構築します。

---

## 現在の進捗

- [x] **Phase 1:** 計画 - 技術選定と開発計画の策定
- [x] **Phase 2:** CNNモデル - 初期モデル群の構築・検証
- [x] **Phase 3:** Transformerモデルへの移行準備 - 時間的文脈を理解できるモデル設計
- [x] **Phase 4:** 牌譜パーサーの全面改修 - 新データ形式対応のtransformer_parser.py開発
- [x] **Phase 4.5:** 徹底的なデバッグ - 新パーサーとシミュレーターの動作検証・修正（完了！）
- [ ] **Phase 5:** Transformerモデルの学習 - 全データで本格的な学習（現在ここ）
- [ ] **Phase 6:** 自己対戦（強化学習） - AI同士の対戦による戦略発見

---

## 各スクリプトの役割（`src/transformer/` ディレクトリ）

- **transformer_parser.py**  
  天鳳の `.mjlog` ファイルを解析し、ゲームの出来事を時系列イベントデータに変換する心臓部。

- **generate_data.py**  
  `transformer_parser.py` を使って全牌譜ログを処理し、AI学習用の最終データセット（ベクトル化テンソル）を生成。

- **train_transformer.py**  
  生成したデータセットを読み込み、Transformerモデルの学習と保存を行う。

- **predict_transformer.py**  
  学習済みAIモデルを読み込み、与えられた状況に対して最適な行動を予測・提案。

---

## 実行方法

1. **データ準備**  
   `data` フォルダを作成し、解析したい `.mjlog` ファイルを入れる

2. **ライブラリインストール**  
   ```sh
   pip install tensorflow mahjong scikit-learn
   ```

3. **データセット生成**  
   ```sh
   python -m src.transformer.generate_data
   ```

4. **AIの学習**  
   ```sh
   python -m src.transformer.train_transformer
   ```

5. **AIによる予測**  
   ```sh
   python -m src.transformer.predict_transformer
   ```
>>>>>>> f0fb94ae28795f6be74cf4f7044ef4fda6955a09

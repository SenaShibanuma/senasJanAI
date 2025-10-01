# SenasJanAI - 麻雀AI開発プロジェクト

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
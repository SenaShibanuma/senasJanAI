SenasJanAI - 麻雀AI開発プロジェクト
これは、天鳳の牌譜データを活用して最強の麻雀AIを開発するプロジェクトの記録です。

プロジェクトの目標
PCでプレイされている麻雀ゲームの画面をリアルタイムに認識し、AIが最適な打牌を提案するツールの開発を目指します。最終的には、人間のトッププロの思考を模倣し、さらに自己対戦（強化学習）によって人間を超える独自の戦略を発見するAIを構築します。

現在の進捗
[x] Phase 1: 計画 - プロジェクト全体の技術選定と開発計画の策定。

[x] Phase 2: CNNモデル - CNNをベースにした初期モデル群を構築・検証。

[x] Phase 3: Transformerモデルへの移行準備 - より高度な判断を実現するため、時間的文脈を理解できるTransformerモデルの設計に着手。

[x] Phase 4: 牌譜パーサーの全面改修 - Transformerモデル用の新しいデータ形式に対応するため、transformer_parser.pyを新規開発。

[x] Phase 4.5: 徹底的なデバッグ - 新パーサーとシミュレーターの動作を検証し、あらゆる牌譜形式に対応できるよう修正を完了。 (完了！)

[ ] Phase 5: Transformerモデルの学習 - 全データを用いて、Transformerモデルの本格的な学習を開始する。 (現在ここ)

[ ] Phase 6: 自己対戦（強化学習） - AI同士を対戦させ、人間を超える戦略を発見させる。


各スクリプトの役割 (src/transformer/ ディレクトリ)
プロジェクトの主要なロジックは src/transformer/ 内に集約されています。

transformer_parser.py: 天鳳の.mjlogファイルを解析し、ゲームの出来事を時系列イベントデータに変換する、プロジェクトの心臓部。

generate_data.py: transformer_parser.pyを使って全ての牌譜ログを処理し、AIが学習できる形式の最終的なデータセット（ベクトル化されたテンソル）を生成します。

train_transformer.py: generate_data.pyが生成したデータセットを読み込み、Transformerモデルの学習と保存を行います。

predict_transformer.py: 学習済みのAIモデルを読み込み、与えられた状況に対して最適な行動を予測・提案します。

実行方法
データ準備: dataフォルダを作成し、解析したい.mjlogファイルを入れます。

ライブラリインストール: pip install tensorflow mahjong scikit-learn

データセット生成: python -m src.transformer.generate_data

AIの学習: python -m src.transformer.train_transformer

AIによる予測: python -m src.transformer.predict_transformer
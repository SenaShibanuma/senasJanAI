# 麻雀AI 事前学習パイプライン

このプロジェクトは、麻雀の牌譜ログから人間の打牌選択を学習する、シンプルなAIモデルを構築するためのコマンドラインツールです。

天鳳形式の牌譜ログ（`.mjlog`）を解析・ベクトル化し、TensorFlowを用いてモデルの事前学習を行います。

## 主な機能

*   **データ前処理**: 天鳳形式の牌譜ログ（`.mjlog`または`.xml`）を解析し、モデルが学習可能な形式（`.npy`ファイル群）に変換します。
*   **モデル学習**: 前処理済みのデータセットを使い、打牌選択を予測するモデルを学習します。
*   **シンプルなCLI**: `main.py`を通じて、すべての操作をコマンドで実行できます。

## セットアップ

### 1. 仮想環境の構築

プロジェクトの依存関係を管理するため、Pythonの仮想環境を作成します。

```shell
# Python 3.10以降を推奨
python -m venv .venv
```

### 2. 依存ライブラリのインストール

作成した仮想環境を有効化し、必要なライブラリをインストールします。

**Windowsの場合:**
```shell
.venv\Scripts\activate
```

**macOS / Linuxの場合:**
```shell
source .venv/bin/activate
```

**インストール実行:**
```shell
pip install -r requirements.txt
```

### 3. 学習データの準備

学習に使用したい天鳳形式の牌譜ログ（`.mjlog`または解凍済みの`.xml`ファイル）を、`data/raw_logs/`ディレクトリに配置してください。

## 使い方

すべての操作は`main.py`を通じて行います。

### データの前処理

`data/raw_logs/`内の牌譜を解析し、学習用のデータを`data/processed/`ディレクトリ内に`train`, `validation`, `test`のサブディレクトリとして生成します。

```shell
python main.py preprocess
```

### モデルの学習

前処理済みのデータ（`data/processed/`内のファイル）を使ってモデルを学習します。学習の進捗は`models/logs/`に保存され、最も性能の良いモデルが`models/tenho_model.keras`として保存されます。

```shell
python main.py pretrain
```

### 全工程の実行

データの前処理とモデルの学習を連続して実行します。

```shell
python main.py all
```

## ディレクトリ構成

```
C:\Work\senasJanAI
├── data/
│   ├── raw_logs/      # (入力) .mjlog ファイルをここに配置
│   └── processed/     # (出力) 前処理済みのデータが保存される
├── models/            # (出力) 学習済みモデルとログが保存される
├── src/               # ソースコード
├── main.py            # メイン実行ファイル
├── preprocess.py      # データ前処理スクリプト
├── pretrain.py        # モデル学習スクリプト
└── requirements.txt   # 依存ライブラリ
```

## 注意事項

*   モデルの性能は、学習に使用するデータの量と質に大きく依存します。強力なAIを育成するには、数万局以上の大規模なデータセットが必要です。
*   `main.py`の各コマンドには、`--test-size`や`--validation-size`などのオプション引数があります。詳細は`python main.py --help`で確認できます。
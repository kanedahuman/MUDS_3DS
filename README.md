# sherd-3d-merge

土器破片の表/裏トレイスキャンを統合し、個別の三次元モデルを生成する。

## セットアップ
    python -m venv .venv
    .venv\Scripts\activate
    pip install -e ".[dev]"

## 実行
    sherd-merge --config config.yaml

## テスト
    pytest -v

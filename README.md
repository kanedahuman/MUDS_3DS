# sherd-3d-merge

土器破片の表/裏トレイスキャンを統合し、個別の三次元モデルを生成する。

## セットアップ

Open3D は Windows では Python 3.12 までの対応のため、conda で 3.12 環境を作る。

    conda create -y -n sherd python=3.12
    conda activate sherd
    pip install -e ".[dev]"

## 実行
    sherd-merge --config config.yaml

## テスト
    pytest -v

# 実スキャン受け入れテスト

合成データの自動テストとは別に、実際のスキャンで end-to-end を確認する手順。

## 手順

1. 5〜10 片を**互いに離して**トレイに並べ、表面をスキャン → `input/tray_front.ply`
2. 各破片を**同じ位置で左右反転**（縦軸まわりに裏返す）して裏面をスキャン → `input/tray_back.ply`
   - 破片の XY 位置はできるだけ動かさない（位置事前情報が照合を助ける）。
3. `config.yaml` を用意（`config.example.yaml` をコピーし、`target_unit_mm` と各閾値を実測スケールに合わせる）。
4. 実行：
   ```
   conda activate sherd
   sherd-merge --config config.yaml
   ```
5. 確認：
   - `output/fragments/*.ply` が破片数だけ生成されるか。
   - `output/report.csv` の `needs_review=True` の件数が許容範囲か。
   - `work/` の照合ペアを CloudCompare で目視確認。
   - `icp_rmse` 列が目標（スキャン解像度に応じ、目安 < 0.3mm 相当）に収まるか。

## 調整すべきパラメータ

| 症状 | 調整 |
|------|------|
| 破片が分離されない／併合される | `segment.cluster_eps`（点間隔に合わせる）、`cluster_min_points` |
| トレイ面が残る／破片が削れる | `preprocess.plane_dist_threshold` |
| 表裏の対応を取り違える | `match.position_prune_radius`（反転時の重心移動量に合わせる）、`cost_threshold` |
| 統合がずれる／RMSE が大きい | `registration.icp_max_corr_dist`、`icp_max_iter`、`boundary_quantile`（縁の巻き込み量に合わせる） |

調整した値と所見はこの README に追記し、`config.yaml` をコミットして再現可能にすること。

## 注意

- 実データでの最終的な受け入れ基準（照合正解率・統合 RMSE・要確認率）はスキャン解像度に依存するため、最初のパイロットで実測して確定する。

---

## 実データ受け入れ結果（2026-06-22, Phase 2）

対象：`MUDS-3DCtest/test5/Aside.ply`（表）/ `Bside.ply`（裏）。4破片トレイ、各約120万点、ターンテーブル上。設定＝`config.real.yaml`。

- **前処理**：トレイ平面（ターンテーブル）が点群の約71%。RANSAC 平面除去で除去後、約34万点の破片群。`voxel_size: 0.4` で約7千点に間引き、DBSCAN・ICP が実用速度に。
- **セグメンテーション**：`cluster_eps: 2.0` / `cluster_min_points: 300` で表裏とも4破片を抽出。
- **照合**：本データは**トレイ全体を裏返した**ため表裏の破片は共在しない。`position_prune_radius: 500`（実質無効化）とし、輪郭ベースコストで4破片すべてを 1:1 割当。
- **統合結果**：

  | 個体 | 対応(裏) | 照合コスト | icp_rmse | 要確認 |
  |------|---------|-----------|----------|--------|
  | fragment_001 | back_002 | 0.115 | 0.85 | No |
  | fragment_002 | back_001 | 0.054 | 0.47 | No |
  | fragment_003 | back_004 | 0.074 | 0.40 | No |
  | fragment_004 | back_003 | 0.142 | 0.37 | No |

  4個体とも薄い湾曲した殻（表裏2面＋厚み）として統合され、要確認フラグなし。

### 所見・既知の制約
- スケールは座標系1単位≒1mm 想定（`target_unit_mm: 1.0`）。破片の実寸が判れば確定する。
- 面積依存の旧コストでは表裏の被覆差で照合が乱れたが、輪郭（極半径シグネチャ）ベースに変更して安定した。
- トレイ全体反転のケースでは位置事前情報が使えないため枝刈りを無効化する。各破片を同じ位置で個別反転すれば位置事前情報が有効になり、より頑健になる。

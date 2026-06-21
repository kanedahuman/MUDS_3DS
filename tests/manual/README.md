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

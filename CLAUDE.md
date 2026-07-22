# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

Irodori-TTS: 日本語NSFW音声作品（ASMR等）を学習データとするTTSの学習・推論・データセット構築コードベース。ベースモデル `Aratako/Irodori-TTS-500M-v3`、音声表現は `Aratako/Semantic-DACVAE-Japanese-32dim`（48kHz / hop 1920 = **25fps** / 32次元latent）のRectified Flow。

## 環境・コマンド

- Windows + uv 管理。同期は `uv sync --extra cu128 --all-groups`（**cu128 extraを忘れるとCPU版torchに置き換わる**）
- 実行は `.\.venv\Scripts\python.exe`。GPU: RTX 5060 Ti 16GB ×2、CPU: Xeon 52コア/104スレッド
- transformers 5.x / huggingface-hub 1.x / faster-whisper（CTranslate2）
- テストスイートは無い。検証は各CLIの `--max-files` / `--max-rows` / `--dry-run` でのスモーク実行

```powershell
# データセット全再作成（全ステージ再開可能・キャッシュ済みは自動スキップ）
python -m dataset.cli.prepare_all           # --dry-run / --start-at / --stop-after / --force-stage
# 学習（resume自動判別コマンドは train/configs/train_command_windows.txt）
python -m train.cli.train --config train\configs\train_500m_v3_full.yaml --manifest dataset\data\manifests\train.jsonl --output-dir outputs\... --device cuda
# チェックポイント→safetensors（EMA重みを使うなら --use-ema 必須）
python -m inference.cli.convert_checkpoint <ckpt.pt> --use-ema
```

## データパイプライン（dataset/ — 6ステージ、完全ローカル）

クラウドSTT・音響分類器は無い。**Grok CLIはLLMとしてテキスト校正+分類にのみ**使う（`~/.grok` にサブスクログイン済みであること。モデルは grok-4.5 にピン留め）。オーケストレータは `prepare_all.py`、work dirは `dataset/data/pipeline/`、ログは `full_pipeline/logs/`、実行ログ `run.log`。

1. **speech** (`speech_pipeline.py`): Silero VAD → 発話を**5〜20秒**にパック → FLACクリップ + `all.jsonl`（text空）。動的プロセスプール24並列（CPUバウンド）。VAD結果は `vad_responses/*.json` にキャッシュ（キーに `speech_pad_ms` 等のVAD設定を含む — nonverbal側と**同一設定必須**）
2. **nonverbal** (`nonverbal_pipeline.py`): VADが拾わなかったギャップ（≥5秒）を `acoustic_segmentation.py` の自然境界で切り、無音除去して5〜20秒イベントクリップ化。行形式はspeechと同一（`origin: vad_complement`）。囁き・喘ぎ・フェラ音がここから回収される
3. **transcribe**（旧称 anime_whisper）: speech+nonverbalを1本に連結（`all_clips.jsonl`）し、faster-whisper + `TransWithAI/whisper-ja-1.5B-ct2` で文字起こし。**CT2直叩きのクロスクリップバッチ推論**（`local_asr.FasterWhisperTranscriber`、反復抑制: repetition_penalty 1.1 + condition_on_previous_text=False）。GPUあたり2ワーカー×バッチ16をシャード分割（`--shard-index/count`）→ID順マージ
4. **context_correction** (`transcript_correction.py`): Grok-4.5が同一音源タイムライン文脈で校正+**カテゴリ分類**。`category ∈ {speech, aegi, chupa, mixed, other}` が唯一のラベル権威。`other`（ノイズ）はreview行き。安全ゲート（類似度・長さ比）を通らない修正は棄却。バッチ結果は `text_correction/` にキャッシュ。並列128が実績上限（**256はgrok CLIがタイムアウト連鎖で崩壊**、timeout 600s）。出力は `all_corrected.jsonl` / `train.jsonl` / `review.jsonl`（入力の`all.jsonl`は上書きしない）
5. **latents**: `prepare_manifest` がDACVAEエンコード（2GPU分散、`category`/`cluster_token` フィールドをパススルー）
6. **publish**: `manifest_merge.py` で検証（latent重複・ID重複・実在）→ `dataset/data/manifests/train.jsonl` へ原子的置換（旧版は `train_before_full_*` にバックアップ）

### 重要な不変条件

- 最終train.jsonl行: `text`（発話内容のみ、ラッパー無し）/ `latent_path` / `num_frames`(25fps) / `speaker_id` / `id` / `source_uid` / `category`
- `speaker_id` は**RJ作品単位**（publish時に `RJ\d+` へ正規化。1作品=1CV前提。RJコードが無いソースはファイル単位IDのまま）。中間成果物はファイル単位IDを保持 — publish以外で正規化するとキャッシュ連鎖でlatent全再エンコードになるため触らない
- クリップはcontent-addressed（idにstart/end ms）、各キャッシュ（VAD/ASR/校正）はモデル・設定メタデータがキー — 設定変更で自動無効化、再実行は差分のみ
- 空textの行を学習マニフェストに入れない（latentステージの行数検証が落ちる）。ASR不能行は `aw_transcript_unusable` でreview行き
- 学習側 `max_latent_steps: 750`（=30秒）に対しデータは20秒上限なので切り詰めは発生しない設計

### 長時間実行の運用

- パイプラインは `Start-Process` でデタッチ起動する（Claude Code終了の巻き添えで死んだ実績あり）。落ちても再起動すればキャッシュから続行
- mpg123の `dequantization failed` / id3警告はソースmp3の壊れフレームに対する無害ログ。監視フィルタから除外すること

## 学習・推論（train/ core/ inference/）

- `core/`: モデル本体（RF/DiT、DACVAEラッパ、tokenizer、LoRA、watermark）
- EMA有効（`ema_device: cpu`）。**推論品質はEMA重み前提** — エクスポート時 `--use-ema` を忘れない
- 学習configは `train/configs/train_500m_v3_full.yaml` の一本のみ（起動コマンドは `train_command_windows.txt`）。valid分割なし（`valid_ratio: 0`）、毎epochサンプル推論
- マニフェスト差し替え時は `train.jsonl.irodori_index.pt`（loaderインデックス）が自動再構築される（キャッシュキーにバージョン番号 `_MANIFEST_INDEX_CACHE_VERSION` — indexへのフィールド追加時は必ずインクリメント）
- **resume時**、yaml/CLIのoptimizerハイパラ（`learning_rate`/`weight_decay`/`muon_momentum`/`adam_beta*`/`adam_eps`/`muon_adjust_lr_fn`）はcheckpoint値を上書きして反映（moment・stepカウントは保持）。`weight_decay`はdecayグループのみに適用（param_groupの`irodori_decay`マーカーで判別。旧checkpointは`wd>0`フォールバック）
- 学習精度は **FP32 固定**（`precision: fp32`）。`precision=bf16` にすると CUDA autocast 混合精度。pure-bf16 実験経路は削除済み

### category を使った学習時サンプリング（train/dataset.py + train/sampler.py）

- **参照音声（speaker条件のref latent）は「同一speaker＋同一category」の別クリップから毎回ランダム選択**。自分自身は絶対に参照しない。同カテゴリに他候補が無い行はspeakerフォールバックせず**Dataset構築時に学習対象から除外**（`style_filtered_count`、マニフェスト自体は不変更）
- **カテゴリ均等サンプリング**（`category_balancing: true` デフォルト）: 毎epoch、各カテゴリから最小カテゴリ件数を非復元でランダム抽出（アンダーサンプリング）。1epoch = 最小カテゴリ件数×カテゴリ数で、epochごとにサブセットが入れ替わる。refプールは**除外前でなくロード済み全サンプル**から構築されるため、この間引きでrefフォールバックは発生しない
- 1epochが実データより大幅に短くなるので `save_every_epochs` / `sample_every_epochs` の実質頻度に注意
- `sample_training_checkpoint.py`（epoch毎サンプル推論）もテキストのカテゴリに一致するrefクリップを作品ごとに選ぶ（無ければspeechにフォールバック、`metadata.json` の `reference_category` に記録）

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

Irodori-TTS: 日本語NSFW音声作品（ASMR等）を学習データとするTTSの学習・推論・データセット構築コードベース。ベースモデルは `Aratako/Irodori-TTS-500M-v3`、音声表現は `Aratako/Semantic-DACVAE-Japanese-32dim`（48kHz / hop 1920 = **25fps** / 32次元latent）のRectified Flow。

## 環境・コマンド

- Windows + uv 管理。venv同期: `uv sync --extra cu128 --all-groups`（**cu128 extraを忘れるとCPU版torchに置き換わる**）
- 実行は `.\.venv\Scripts\python.exe`（GPU: RTX 5060 Ti 16GB ×2）
- transformers は 5.x 系（`>=5.4`、hub 1.x）。ASRは faster-whisper + `TransWithAI/whisper-ja-1.5B-ct2`
- テストスイートは無い。検証は各CLIの `--max-rows` / `--max-files` / `--dry-run` でのスモーク実行

```powershell
# データセット全再作成（VAD→分類→文字起こし→Grok校正→latent→publish、再開可能）
python -m dataset.cli.prepare_all            # --dry-run でpreflightのみ / --start-at/--stop-after/--force-stage
# 学習（resume自動判別コマンドは train/configs/train_command_windows.txt）
python -m train.cli.train --config train\configs\train_500m_v3_full.yaml --manifest dataset\data\manifests\train.jsonl --output-dir outputs\... --device cuda
# チェックポイント→safetensors（EMA重みを使うなら --use-ema 必須）
python -m inference.cli.convert_checkpoint <ckpt.pt> --use-ema
```

## データパイプライン（dataset/）

**クラウドSTTは使わない**（Grok STTは全廃済み）。`prepare_all.py` が全ステージをfingerprint付きで統括し、`dataset/data/pipeline/`（work dir）に再開可能な状態を持つ。ログ: `dataset/data/pipeline/full_pipeline/logs/`。

1. **speech** (`speech_pipeline.py` / `prepare_speech`): ソース走査 → Silero VAD → 発話を**5〜20秒**にパック → FLACクリップ + `all.jsonl`（この時点で text は空）。VAD結果は `vad_responses/*.json` にキャッシュされ、nonverbal側も同じファイルを読む
2. **nonverbal**: `acoustic_segmentation.py`（VAD外領域の自然区切り）→ BEATs埋め込み → `ero_voice_classifier.py`（aegi/chupa分類、pyannote動的import）
3. **anime_whisper ステージ**: 実体は faster-whisper（`local_asr.FasterWhisperTranscriber`、反復抑制: repetition_penalty 1.1 + compression_ratio 2.4 + condition_on_previous_text=False）。speech/nonverbal両方を文字起こしし、`--shard-index/--shard-count` でGPU台数分のプロセスに分割→ID順マージ。speechは `--replace-text` で text を確定
4. **context_correction** (`transcript_correction.py`): Grok CLI（LLMとしてのみ使用可。codex/claudeもフォールバック可）が同一音源タイムライン文脈で校正+**カテゴリ分類**（speech/aegi/chupa/mixed/other）。`other` は review 行きで学習除外。安全ゲート（類似度・長さ比）を通らない修正は棄却
5. **manifests → latents → publish**: text は**発話内容のみ**（`喘ぎ声。…ラベルc0000。` 形式は廃止）。分類は `category` / `cluster_token` の別フィールドで `prepare_manifest`（DACVAEエンコード、マルチGPU対応）を通って最終 `dataset/data/manifests/train.jsonl` まで保持される

### 重要な不変条件

- 最終train.jsonlの行: `text` / `latent_path` / `num_frames`(25fps) / `speaker_id` / `id` / `source_uid` / `category` / `cluster_token`
- クリップは content-addressed（idにstart/end ms埋め込み）で再実行時に再利用。キャッシュ類（VAD/ASR/校正）はモデル・設定のメタデータでキーされ、変更時に自動無効化
- nonverbalイベントの適格レンジも5〜20秒（`nonverbal_training_manifest.py` と `acoustic_segmentation.py` の上限を揃えること）
- LLM分類がnonverbalで `speech`/`other` を返した行、文字起こし不能行（`missing_transcript`）は review へ。空textの行を学習マニフェストに入れない（latent段の行数検証が壊れる）

## 学習・推論（train/ core/ inference/）

- `core/`: モデル本体（RF/DiT、DACVAEコーデックラッパ、tokenizer、LoRA、watermark）
- 学習設定は `train/configs/train_500m_v3_full.yaml`。`max_latent_steps: 750`（=30秒）を超える行はlatentだけ切り詰められてtext不一致になるため、データ側で20秒上限を守る
- EMA有効（`ema_device: cpu` でVRAM消費なし、update毎10step）。**推論品質はEMA重み前提**なので、エクスポート時の `--use-ema` を忘れない
- 学習マニフェスト差し替え時は `train.jsonl.irodori_index.pt`（loaderのインデックスキャッシュ）が自動再構築される

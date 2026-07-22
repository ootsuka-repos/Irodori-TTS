"""DACVAEデコーダのみをローカルASMR音源でファインチューニングする。

エンコーダ・in_proj（latent空間）は完全凍結。TTS側のlatent・学習済みモデルは
そのまま使える。学習対象は quantizer.out_proj + decoder.model +
decoder.wm_model.encoder_block.pre（推論時のalpha=0パススルー出力経路）のみ。

学習ペアはsource音源からのランダム切り出しをオンザフライで
「凍結エンコード（決定的mean）→デコード→再構成損失」する。
損失はDACレシピ準拠: multi-scale mel + MS-STFT + MPD/MRD adversarial + feature matching。
精度は FP32 固定（重み・勾配・AdamW・mel/STFT 損失とも float32。autocast なし）。

例:
  python -m train.cli.finetune_dacvae_decoder --output-dir outputs\dacvae_decoder_ft --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
SAMPLE_RATE = 48000
HOP = 1920


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def ffmpeg_excerpt(path: Path, offset: float, seconds: float) -> np.ndarray:
    out = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-ss", f"{offset:.3f}", "-t", f"{seconds + 0.05:.3f}",
            "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "pipe:1",
        ],
        capture_output=True,
    )
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


class ExcerptDataset(Dataset):
    """source音源からランダム切り出し。1エポック=steps_per_epoch仮想長。"""

    def __init__(self, files: list[tuple[str, float]], excerpt_samples: int, virtual_len: int, seed: int):
        self.files = files
        self.excerpt_samples = excerpt_samples
        self.virtual_len = virtual_len
        self.seed = seed
        weights = np.array([d for _, d in files], dtype=np.float64)
        self.probs = weights / weights.sum()

    def __len__(self) -> int:
        return self.virtual_len

    def __getitem__(self, index: int) -> torch.Tensor:
        # 例外でワーカーが死ぬとWindowsでは共有メモリ経由の復旧ができず学習全体が
        # 落ちるため、どんな失敗でも必ずTensorを返す
        try:
            rng = random.Random(self.seed * 1_000_003 + index)
            seconds = self.excerpt_samples / SAMPLE_RATE
            last: np.ndarray | None = None
            for _ in range(6):
                fi = rng.choices(range(len(self.files)), weights=self.probs, k=1)[0]
                path, dur = self.files[fi]
                if dur <= seconds + 1.0:
                    continue
                offset = rng.uniform(0.5, dur - seconds - 0.5)
                wav = ffmpeg_excerpt(Path(path), offset, seconds)
                if wav.shape[0] < self.excerpt_samples:
                    last = wav
                    continue
                wav = wav[: self.excerpt_samples]
                if float(np.sqrt(np.mean(wav**2))) > 1e-3:
                    return torch.from_numpy(wav)
                last = wav
            if last is None:
                last = np.zeros(self.excerpt_samples, dtype=np.float32)
            return torch.from_numpy(np.pad(last, (0, max(0, self.excerpt_samples - last.shape[0]))))
        except Exception as exc:  # noqa: BLE001
            print(f"[data] worker error at index={index}: {exc!r} -> zero excerpt", flush=True)
            return torch.zeros(self.excerpt_samples, dtype=torch.float32)


def normalize_batch(wav: torch.Tensor, target_db: float) -> torch.Tensor:
    """(B, T) → (B, 1, T)。audiotoolsでクリップ毎に-16dBラウドネス正規化（codec.pyと同一処理）。"""
    from audiotools import AudioSignal

    outs = []
    for w in wav:
        sig = AudioSignal(w.unsqueeze(0).unsqueeze(0), SAMPLE_RATE)
        sig.normalize(target_db)
        sig.ensure_max_of_audio()
        outs.append(sig.audio_data.squeeze(0))
    return torch.stack(outs, dim=0)


@torch.no_grad()
def encode_deterministic(model, wav: torch.Tensor) -> torch.Tensor:
    """(B,1,T) → latent mean (B, D, T_lat)。パイプラインのdeterministic encodeと同一。"""
    z = model.encoder(model._pad(wav))
    mean, _scale = model.quantizer.in_proj(z).chunk(2, dim=1)
    return mean


def decode_body(model, latent_mean: torch.Tensor) -> torch.Tensor:
    """alpha=0推論経路の再現: out_proj → decoder.model → wm encoder_block.forward_no_conv。"""
    x = model.quantizer.out_proj(latent_mean)
    for layer in model.decoder.model:
        x = layer(x)
    return model.decoder.wm_model.encoder_block.forward_no_conv(x)


def trainable_modules(model) -> list[torch.nn.Module]:
    return [model.quantizer.out_proj, model.decoder.model, model.decoder.wm_model.encoder_block.pre]


def collect_files(source_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in source_dirs:
        files.extend(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dirs", nargs="*", default=None,
                        help="音源ルート（default: dataset/collected/*/source）")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", default="Aratako/Semantic-DACVAE-Japanese-32dim",
                        help="初期重み（HF repo または .pth。resume時はfinetune済み.pthを指定）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--excerpt-frames", type=int, default=19, help="latentフレーム数（19=0.76秒）")
    parser.add_argument("--gen-lr", type=float, default=5e-5)
    parser.add_argument("--disc-lr", type=float, default=1e-4)
    parser.add_argument("--lr-gamma", type=float, default=0.999996)
    parser.add_argument("--lambda-mel", type=float, default=15.0)
    parser.add_argument("--lambda-stft", type=float, default=1.0)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-feat", type=float, default=2.0)
    parser.add_argument("--normalize-db", type=float, default=-16.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-latents", nargs="*", default=None,
                        help="保存毎にデコードして音を確認するlatent .pt（検証用）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_dirs:
        source_dirs = [Path(d) for d in args.source_dirs]
    else:
        source_dirs = sorted((REPO_ROOT / "dataset" / "collected").glob("*/source"))
    files = collect_files(source_dirs)
    if not files:
        raise SystemExit(f"no audio files under: {[str(d) for d in source_dirs]}")

    # ffprobe結果をキャッシュ（初回のみ全走査）
    dur_cache_path = args.output_dir / "durations.json"
    dur_cache: dict[str, float] = {}
    if dur_cache_path.exists():
        dur_cache = json.loads(dur_cache_path.read_text(encoding="utf-8"))
    missing = [f for f in files if str(f) not in dur_cache]
    if missing:
        from concurrent.futures import ThreadPoolExecutor

        print(f"[data] probing {len(missing)} files ...", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=32) as pool:
            for f, dur in zip(missing, pool.map(ffprobe_duration, missing)):
                dur_cache[str(f)] = dur
                done += 1
                if done % 2000 == 0:
                    print(f"[data] probed {done}/{len(missing)}", flush=True)
                    dur_cache_path.write_text(json.dumps(dur_cache, ensure_ascii=False), encoding="utf-8")
        dur_cache_path.write_text(json.dumps(dur_cache, ensure_ascii=False), encoding="utf-8")
    entries = [(str(f), dur_cache[str(f)]) for f in files if dur_cache[str(f)] > 3.0]
    total_hours = sum(d for _, d in entries) / 3600
    print(f"[data] {len(entries)} files, {total_hours:.1f}h from {len(source_dirs)} roots", flush=True)

    from core.codec import DACVAECodec
    from dacvae.model.discriminator import Discriminator
    from dacvae.nn.loss import GANLoss, MelSpectrogramLoss, MultiScaleSTFTLoss
    from audiotools import AudioSignal

    codec = DACVAECodec.load(
        repo_id=args.weights,
        device=str(device),
        dtype=torch.float32,
        normalize_db=args.normalize_db,
    )
    model = codec.model.float().to(device)
    assert codec.sample_rate == SAMPLE_RATE

    for p in model.parameters():
        p.requires_grad_(False)
    gen_params: list[torch.nn.Parameter] = []
    for m in trainable_modules(model):
        for p in m.parameters():
            p.requires_grad_(True)
            gen_params.append(p)
    model.decoder.train()
    model.encoder.eval()
    n_train = sum(p.numel() for p in gen_params)
    print(
        f"[model] trainable decoder params: {n_train/1e6:.1f}M "
        f"(precision=fp32, dtype={next(model.parameters()).dtype})",
        flush=True,
    )

    disc = Discriminator(sample_rate=SAMPLE_RATE).float().to(device)
    disc_ckpt = args.output_dir / "discriminator_last.pt"
    start_step = 0
    state_path = args.output_dir / "train_state.json"
    if disc_ckpt.exists():
        disc.load_state_dict(torch.load(disc_ckpt, map_location=device))
        if state_path.exists():
            start_step = int(json.loads(state_path.read_text())["step"])
        print(f"[resume] discriminator + step={start_step}", flush=True)

    mel_loss = MelSpectrogramLoss(
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin=[0.0] * 7,
        mel_fmax=[None] * 7,
        mag_weight=0.0,
        pow=1.0,
        clamp_eps=1e-5,
    )
    stft_loss = MultiScaleSTFTLoss()
    gan_loss = GANLoss(disc)

    opt_g = torch.optim.AdamW(gen_params, lr=args.gen_lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.disc_lr, betas=(0.8, 0.99))
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=args.lr_gamma)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=args.lr_gamma)

    excerpt_samples = args.excerpt_frames * HOP

    def make_loader(current_step: int) -> DataLoader:
        dataset = ExcerptDataset(
            entries,
            excerpt_samples,
            virtual_len=(args.steps - current_step) * args.batch_size,
            seed=args.seed + current_step,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            drop_last=True,
        )

    sample_latents = [Path(p) for p in (args.sample_latents or [])]

    def save_checkpoint(step: int) -> None:
        weights_path = args.output_dir / "weights_ft.pth"
        model.save(str(weights_path), package=False)
        torch.save(disc.state_dict(), disc_ckpt)
        state_path.write_text(json.dumps({"step": step}), encoding="utf-8")
        model.decoder.eval()
        with torch.inference_mode():
            for lp in sample_latents:
                latent = torch.load(lp, map_location=device).float().t().unsqueeze(0)  # (T,D)→(1,D,T)
                audio = decode_body(model, latent)
                out_wav = args.output_dir / f"sample_{lp.stem}_step{step}.wav"
                torchaudio.save(str(out_wav), audio[0].cpu().float(), SAMPLE_RATE)
        model.decoder.train()
        print(f"[save] step={step} -> {weights_path}", flush=True)

    step = start_step
    running: dict[str, float] = {}
    while step < args.steps:
        loader = make_loader(step)
        try:
          for batch in loader:
              if step >= args.steps:
                  break
              wav = normalize_batch(batch, args.normalize_db).to(device)  # (B,1,T)
              latent = encode_deterministic(model, wav)
              fake = decode_body(model, latent)
              n = min(fake.shape[-1], wav.shape[-1])
              fake_sig = AudioSignal(fake[..., :n], SAMPLE_RATE)
              real_sig = AudioSignal(wav[..., :n], SAMPLE_RATE)

              # discriminator
              opt_d.zero_grad(set_to_none=True)
              loss_d = gan_loss.discriminator_loss(fake_sig, real_sig)
              loss_d.backward()
              torch.nn.utils.clip_grad_norm_(disc.parameters(), 10.0)
              opt_d.step()
              sched_d.step()

              # generator
              opt_g.zero_grad(set_to_none=True)
              l_mel = mel_loss(fake_sig, real_sig)
              l_stft = stft_loss(fake_sig, real_sig)
              l_adv, l_feat = gan_loss.generator_loss(fake_sig, real_sig)
              loss_g = (
                  args.lambda_mel * l_mel
                  + args.lambda_stft * l_stft
                  + args.lambda_adv * l_adv
                  + args.lambda_feat * l_feat
              )
              loss_g.backward()
              torch.nn.utils.clip_grad_norm_(gen_params, 1e3)
              opt_g.step()
              sched_g.step()

              step += 1
              for k, v in {"mel": l_mel, "stft": l_stft, "adv": l_adv, "feat": l_feat, "disc": loss_d}.items():
                  running[k] = running.get(k, 0.0) + float(v.detach())
              if step % args.log_every == 0:
                  avg = {k: v / args.log_every for k, v in running.items()}
                  running = {}
                  msg = " ".join(f"{k}={v:.3f}" for k, v in avg.items())
                  print(f"[step {step}/{args.steps}] {msg} lr={sched_g.get_last_lr()[0]:.2e}", flush=True)
              if step % args.save_every == 0:
                  save_checkpoint(step)
        except RuntimeError as exc:
          # Windowsではワーカー死亡が共有メモリイベントのRuntimeErrorとして表面化する。
          # 学習状態はプロセス内にあるためワーカーだけ作り直して続行する
          print(f"[warn] DataLoader crashed at step {step}: {exc} -> restarting workers", flush=True)
          del loader

    if step % args.save_every != 0:
        save_checkpoint(step)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

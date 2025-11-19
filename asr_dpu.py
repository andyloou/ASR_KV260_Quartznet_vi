import os, sys, csv, time, argparse, unicodedata
from pathlib import Path
from datetime import datetime
import shutil

import numpy as np
import librosa
import onnxruntime as ort

try:
    from pyctcdecode import build_ctcdecoder
    PYCTCDECODE_AVAILABLE = True
except ImportError:
    PYCTCDECODE_AVAILABLE = False

BASE_TOKENS = [' ', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o',
               'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'à', 'á', 'â', 'ã', 'è',
               'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'đ',
               'ĩ', 'ũ', 'ơ', 'ư', 'ạ', 'ả', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ',
               'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị',
               'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ',
               'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ']
BASE_V = len(BASE_TOKENS) + 1

LABELS = [
    ' ', 'a','b','c','d','e','g','h','i','k','l','m','n','o',
    'p','q','r','s','t','u','v','x','y',
    '\u00e1','\u00e0','\u00e2','\u00e3','\u00e8','\u00e9','\u00ea','\u00ec','\u00ed',
    '\u00f2','\u00f3','\u00f4','\u00f5','\u00f9','\u00fa','\u00fd','\u0103','\u0111',
    '\u0129','\u0169','\u01a1','\u01b0',
    '\u1ea1','\u1ea3','\u1ea5','\u1ea7','\u1ea9','\u1eab','\u1ead',
    '\u1eaf','\u1eb1','\u1eb3','\u1eb5','\u1eb7',
    '\u1eb9','\u1ebb','\u1ebd','\u1ebf','\u1ec1','\u1ec3','\u1ec5','\u1ec7',
    '\u1ec9','\u1ecb','\u1ecd','\u1ecf','\u1ed1','\u1ed3','\u1ed5','\u1ed7','\u1ed9',
    '\u1edb','\u1edd','\u1edf','\u1ee1','\u1ee3',
    '\u1ee5','\u1ee7','\u1ee9','\u1eeb','\u1eed','\u1eef','\u1ef1',
    '\u1ef3','\u1ef5','\u1ef7','\u1ef9'
]
BLANK_ID = 0

def u_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")

def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        ca = a[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if ca == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n]

def cer(ref: str, hyp: str) -> float:
    ref = ref or ""
    if len(ref) == 0:
        return 1.0 if (hyp or "") != "" else 0.0
    return edit_distance(ref, hyp) / len(ref)

def edit_distance_list(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n]

def wer(ref: str, hyp: str) -> float:
    rw = u_nfc(ref).strip().split()
    hw = u_nfc(hyp).strip().split()
    if len(rw) == 0:
        return 1.0 if len(hw) > 0 else 0.0
    return edit_distance_list(rw, hw) / len(rw)

def make_logmel(wav_path: str, target_sr=16000, n_mels=64, n_fft=512, win_ms=0.02, hop_ms=0.01) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    win_length = int(sr * win_ms)
    hop_length = int(sr * hop_ms)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        n_mels=n_mels, power=2.0, center=True
    )
    mel = np.log(mel + 1e-6)
    mean = mel.mean(axis=1, keepdims=True)
    std  = mel.std(axis=1, keepdims=True) + 1e-6
    mel = (mel - mean) / std
    mel = mel[np.newaxis, :, :]
    return mel.astype(np.float32)

def ctc_decode_like_old(logits: np.ndarray, labels=LABELS) -> str:
    x = logits[0]
    x = x - np.max(x, axis=-1, keepdims=True)
    probs = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1)

    out = []
    prev = None
    for p in preds:
        if p != prev:
            if 0 <= p < len(labels):
                out.append(labels[p])
        prev = p
    return ''.join(out)

def to_TV(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 1:
        a, b = arr.shape[1], arr.shape[2]
        if b in (BASE_V, BASE_V - 1) or b <= 256:
            return arr[0]
        if a in (BASE_V, BASE_V - 1) or a <= 256:
            return arr[0].T
        return arr[0] if b <= a else arr[0].T
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected logits shape for to_TV: {arr.shape}")

def normalize_output_logits(out: np.ndarray, vocab_size: int = None) -> np.ndarray:
    arr = out

    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = np.squeeze(arr, axis=1)

    if vocab_size is None:
        if arr.ndim == 3:
            _, d1, d2 = arr.shape
            return arr if d1 >= d2 else np.transpose(arr, (0, 2, 1))
        if arr.ndim == 4 and arr.shape[1] == 1:
            return normalize_output_logits(np.squeeze(arr, axis=1))
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        raise ValueError(f"Unexpected logits shape: {arr.shape}")

    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Unexpected 3D logits with batch != 1: {arr.shape}")
        _, a, b = arr.shape
        if b == vocab_size:
            return arr
        elif a == vocab_size:
            return np.transpose(arr, (0, 2, 1))
        else:
            return np.transpose(arr, (0, 2, 1)) if a < b else arr

    if arr.ndim == 2:
        a, b = arr.shape
        if b == vocab_size:
            return arr[np.newaxis, ...]
        elif a == vocab_size:
            return np.transpose(arr, (1, 0))[np.newaxis, ...]
        else:
            return arr[np.newaxis, ...] if b < a else np.transpose(arr, (1, 0))[np.newaxis, ...]

    raise ValueError(f"Unexpected logits shape: {arr.shape}")


def detect_blank_index_from_logits(out_any: np.ndarray) -> int:
    tv = to_TV(out_any)
    col_means = tv.mean(axis=0)
    return int(np.argmax(col_means))

def build_labels_with_blank_at(blank_idx: int) -> list:
    labels = []
    bi = 0
    for i in range(len(BASE_TOKENS) + 1):
        if i == blank_idx:
            labels.append("")
        else:
            labels.append(BASE_TOKENS[bi])
            bi += 1
    return labels


class LMDecoder:
    def __init__(self, labels, lm_binary=None, beam_width=100, alpha=0.5, beta=1.0, verbose=True, blank_idx=None):
        self.ok = False
        self.verbose = verbose
        self.labels = list(labels)
        self.blank_idx = blank_idx
        self.lm_binary = lm_binary if (lm_binary and Path(lm_binary).exists()) else None
        self.decoder = None

        if not PYCTCDECODE_AVAILABLE:
            if self.verbose:
                pass
            return

        def _build_with_or_without_ctc_idx(use_ctc_idx: bool):
            if self.lm_binary is None:
                if use_ctc_idx:
                    return build_ctcdecoder(labels=self.labels, ctc_token_idx=self.blank_idx)
                return build_ctcdecoder(labels=self.labels)
            else:
                if use_ctc_idx:
                    return build_ctcdecoder(labels=self.labels, kenlm_model_path=self.lm_binary,
                                            alpha=alpha, beta=beta, ctc_token_idx=self.blank_idx)
                return build_ctcdecoder(labels=self.labels, kenlm_model_path=self.lm_binary,
                                        alpha=alpha, beta=beta)

        try:
            if self.blank_idx is not None:
                self.decoder = _build_with_or_without_ctc_idx(True)
            else:
                self.decoder = _build_with_or_without_ctc_idx(False)
            self.ok = True
            if self.verbose:
                pass
        except TypeError as e:
            if self.verbose:
                pass
            try:
                self.decoder = _build_with_or_without_ctc_idx(False)
                self.ok = True
                if self.verbose:
                    pass
            except Exception as e2:
                self.ok = False
                if self.verbose:
                    pass
        except Exception as e:
            self.ok = False
            if self.verbose:
                pass

    def decode(self, logits_1tv: np.ndarray, beam_width=100) -> str:
        if not self.ok or not PYCTCDECODE_AVAILABLE or self.decoder is None:
            return None
        try:
            tv = logits_1tv[0] if logits_1tv.ndim == 3 else logits_1tv
            tv = tv.astype(np.float32, copy=False)
            hyp = self.decoder.decode(tv, beam_width=beam_width)
            return hyp
        except Exception as e:
            if self.verbose:
                pass
            return None

def load_ground_truth(prompts_file: Path):
    gt = {}
    if not prompts_file.exists():
        return gt
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split(' ', 1)
            if len(parts) == 2:
                audio_id, text = parts
                gt[audio_id] = u_nfc(text)
    return gt

def find_audio_files(test_dir: Path):
    waves = test_dir / 'waves'
    items = []
    if not waves.is_dir():
        return items
    for spk in sorted(waves.iterdir()):
        if not spk.is_dir():
            continue
        for wav in sorted(spk.glob('*.wav')):
            items.append((wav.stem, str(wav)))
    return items

def prepare_input_tensor(mel_nct: np.ndarray, layout: str) -> np.ndarray:
    layout = layout.upper()
    if layout == 'NCT':  return mel_nct
    if layout == 'NTC':  return np.transpose(mel_nct, (0, 2, 1))
    if layout == 'NCHW': return mel_nct[:, np.newaxis, :, :]
    raise ValueError(f"Unsupported layout: {layout}")

def create_vitisai_session(model_path: str, vaip_config: str):
    so = ort.SessionOptions()
    try:
        sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": vaip_config}],
            disable_fallback=True,
        )
    except TypeError:
        sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": vaip_config}],
        )
    return sess

def main():
    ap = argparse.ArgumentParser(description="ASR on DPU with optional LM Beam Search (PyCTCDecode)")
    ap.add_argument('--model', default='quan_quartz.onnx')
    ap.add_argument('--test-dir', default='test')
    ap.add_argument('--input-name', default=None)
    ap.add_argument('--layout', default='NCT', choices=['NCT','NTC','NCHW'])
    ap.add_argument('--vaip-config', default='/usr/bin/vaip_config.json')
    ap.add_argument('--n-mels', type=int, default=64)
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--win-ms', type=float, default=0.02)
    ap.add_argument('--hop-ms', type=float, default=0.01)
    ap.add_argument('--out-csv', default='/tmp/results.csv')
    ap.add_argument('--preview', type=int, default=10)
    ap.add_argument('--copy-to-boot', action='store_true', default=False)

    ap.add_argument('--lm-binary', default= "5-gram-lm.binary")
    ap.add_argument('--beam-width', type=int, default=100, help='Beam width cho decode')
    ap.add_argument('--alpha', type=float, default=0.5, help='LM weight (0.3-0.7)')
    ap.add_argument('--beta', type=float, default=1.0, help='Word insertion penalty (0.5-1.5)')
    ap.add_argument('--use-lm', action='store_true', help='Force use LM even if not specified')

    args = ap.parse_args()

    try:
        session = create_vitisai_session(args.model, args.vaip_config)
        print("Using DPU (VitisAIExecutionProvider)")
    except Exception as e:
        print("Chi tiết:", repr(e))
        sys.exit(1)

    if args.input_name is None:
        inps = session.get_inputs()
        if not inps:
            sys.exit(1)
        input_name = inps[0].name
    else:
        input_name = args.input_name
    print("Using input:", input_name, "| layout:", args.layout)

    test_dir = Path(args.test_dir)
    gt = load_ground_truth(test_dir / 'prompts.txt')
    items = find_audio_files(test_dir)
    print(f"Found {len(items)} wav files | GT entries: {len(gt)}\n")

    use_lm = args.lm_binary is not None or args.use_lm
    lm_decoder = None
    dynamic_labels = None
    vocab_size = None

    if use_lm:
        probe_wav = None
        for audio_id, wav in items:
            if audio_id in gt:
                probe_wav = wav
                break
        if probe_wav is None:
            sys.exit(1)

        mel_probe = make_logmel(probe_wav, target_sr=args.sr, n_mels=args.n_mels,
                                n_fft=512, win_ms=args.win_ms, hop_ms=args.hop_ms)
        x_probe = prepare_input_tensor(mel_probe, args.layout)
        out_probe = session.run(None, {input_name: x_probe})[0]
        
        tv_probe = to_TV(out_probe)
        T_probe, V_probe = tv_probe.shape
        print(f"Model output: T={T_probe}, V={V_probe}, Expected V={BASE_V}")

        blank_idx = detect_blank_index_from_logits(out_probe)
        print(f"Detected blank_idx = {blank_idx}")

        dynamic_labels = build_labels_with_blank_at(blank_idx)
        vocab_size = len(dynamic_labels)
        print(f"Built dynamic labels: {vocab_size} tokens")

        lm_decoder = LMDecoder(
            labels=dynamic_labels,
            lm_binary=args.lm_binary,
            beam_width=args.beam_width,
            alpha=args.alpha,
            beta=args.beta,
            verbose=True,
            blank_idx=blank_idx
        )

        if not lm_decoder.ok:
            use_lm = False
            lm_decoder = None
    else:
        pass

    results = []
    total_infer = 0.0
    total_audio = 0.0
    processed = 0

    for audio_id, wav in items:
        if audio_id not in gt:
            continue
        ref = gt[audio_id]

        mel = make_logmel(wav, target_sr=args.sr, n_mels=args.n_mels,
                          n_fft=512, win_ms=args.win_ms, hop_ms=args.hop_ms)
        x = prepare_input_tensor(mel, args.layout)

        t0 = time.time()
        out = session.run(None, {input_name: x})[0]
        infer_time = time.time() - t0

        if use_lm and lm_decoder and lm_decoder.ok:
            logits = normalize_output_logits(out, vocab_size=vocab_size)
            hyp_raw = lm_decoder.decode(logits, beam_width=args.beam_width)
            if hyp_raw is None:
                logits_old = normalize_output_logits(out)
                hyp_raw = u_nfc(ctc_decode_like_old(logits_old, labels=LABELS))
                decode_method = "CTC_fallback"
            else:
                decode_method = f"LM_beam{args.beam_width}"
        else:
            logits_old = normalize_output_logits(out)
            hyp_raw = u_nfc(ctc_decode_like_old(logits_old, labels=LABELS))
            decode_method = "CTC_greedy"

        hyp = u_nfc(hyp_raw.upper())
        
        audio_dur = mel.shape[2] * args.hop_ms
        cer_val = cer(ref, hyp)
        wer_val = wer(ref, hyp)

        if processed < args.preview:
            print(f"[{processed+1:04d}] {audio_id}")
            print("  REF:", ref)
            print("  HYP:", hyp)
            print(f"  CER: {cer_val:.3f} | WER: {wer_val:.3f} | Method: {decode_method}\n")

        results.append((audio_id, ref, hyp, cer_val, wer_val, infer_time, audio_dur,
                        (infer_time / audio_dur) if audio_dur > 0 else 0.0, decode_method))

        total_infer += infer_time
        total_audio += audio_dur
        processed += 1

        if processed % 50 == 0:
            print(f"Processed {processed} files")

    avg_cer = (np.mean([r[3] for r in results]) if results else 1.0)
    avg_wer = (np.mean([r[4] for r in results]) if results else 1.0)
    avg_rtf = (total_infer / total_audio) if total_audio > 0 else 0.0
    avg_speed = (total_audio / total_infer) if total_infer > 0 else 0.0

    print("\nEVALUATION RESULTS")
    print("==================")
    print(f"Total files processed: {processed}")
    print(f"Total audio duration: {total_audio:.2f} s")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
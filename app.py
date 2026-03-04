import os, time, threading, queue
from pathlib import Path
from datetime import datetime
import subprocess
import collections
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from asr_dpu import (
    LABELS, BLANK_ID, u_nfc, make_logmel, make_logmel_from_array, ctc_decode_like_old,
    prepare_input_tensor, normalize_output_logits, create_vitisai_session,
    to_TV, detect_blank_index_from_logits, build_labels_with_blank_at,
    LMDecoder, BASE_TOKENS, BASE_V
)

AUDIO_HW = "hw:1,0"
REC_SR = 48000
REC_CH = 2
REC_FMT = "S16_LE"

SAVE_DIR = "/home/petalinux/asr_webappQ3/asr_webapp/file_audio"
VAD_MODEL = "/home/petalinux/VAD_project/vad_marblenet_full1.onnx"
ASR_MODEL = "quan_quartz_1.onnx"
VAIP_CONFIG = "/usr/bin/vaip_config.json"

TARGET_SR = 16000
VAD_THRESHOLD = 0.45
CHUNK_DURATION = 0.4

VAD_MIN_SPEECH_DURATION = 0.5
VAD_MIN_SILENCE_DURATION = 0.9
VAD_SPEECH_PAD_MS = 500
_PRE_BUFFER_CHUNKS = max(1, int((VAD_SPEECH_PAD_MS / 1000) / CHUNK_DURATION))
LM_BINARY_DEFAULT = os.environ.get("LM_BINARY", "5-gram-lm.binary")
LM_BEAM = int(os.environ.get("LM_BEAM", "100"))
LM_ALPHA = float(os.environ.get("LM_ALPHA", "0.5"))
LM_BETA = float(os.environ.get("LM_BETA", "1.0"))

HUM_HZ = 50.0
TRIM_DB = 25.0
PEAK_DBFS = -3.0
TARGET_RMS_DBFS = -21.0

def _resolve_lm_path(lm_name_or_path: str):
    if not lm_name_or_path:
        return None
    p = Path(lm_name_or_path)
    if p.is_file():
        return str(p.resolve())
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / lm_name_or_path
    if candidate.is_file():
        return str(candidate.resolve())
    return None
from scipy.signal import iirnotch, lfilter

def biquad_notch(y, fs, f0, Q=35.0):
    b, a = iirnotch(f0, Q, fs)
    return lfilter(b, a, y)

def dc_block(y, R=0.995):
    return lfilter([1, -1], [1, -R], y)
'''
def dc_block(y: np.ndarray, R: float = 0.995) -> np.ndarray:
    if y.size == 0:
        return y
    y_out = np.empty_like(y)
    x1 = 0.0; y1 = 0.0
    for i, x in enumerate(y):
        y0 = x - x1 + R * y1
        y_out[i] = y0
        x1, y1 = x, y0
    return y_out

def biquad_notch(y: np.ndarray, fs: float, f0: float, Q: float = 35.0) -> np.ndarray:
    if y.size == 0:
        return y
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * Q)
    b0, b1, b2 = 1.0, -2.0 * np.cos(w0), 1.0
    a0, a1, a2 = 1.0 + alpha, -2.0 * np.cos(w0), 1.0 - alpha
    b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0
    y_out = np.empty_like(y)
    x1 = x2 = y1 = y2 = 0.0
    for i, x0 in enumerate(y):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y_out[i] = y0
        x2, x1, y2, y1 = x1, x0, y1, y0
    return y_out
def apply_cleaning_and_resample(input_wav: str, out_wav: str = None):
    p = Path(input_wav)
    if out_wav is None:
        out_wav = str(p.with_name(p.stem + "_clean16k.wav"))

    y, sr = librosa.load(input_wav, sr=None, mono=True)
    y = biquad_notch(y, fs=sr, f0=HUM_HZ, Q=35.0)
    y = dc_block(y, R=0.995)
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)

    if y.size:
        peak = float(np.max(np.abs(y))) + 1e-12
        rms = float(np.sqrt(np.mean(y**2))) + 1e-12
        rms_db = 20*np.log10(rms)
        gain_db = TARGET_RMS_DBFS - rms_db
        max_gain_db = 20*np.log10((10**(-1/20)) / peak)
        gain_db = min(gain_db, max_gain_db)
        y = y * (10**(gain_db/20))

        peak = float(np.max(np.abs(y))) + 1e-12
        target_peak = 10**(PEAK_DBFS/20)
        if peak > target_peak:
            y *= (target_peak / peak)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="polyphase")
        sr = TARGET_SR

    sf.write(out_wav, y, sr, subtype="PCM_16")
    return out_wav
'''
from scipy.signal import iirnotch, lfilter

def biquad_notch(y, fs, f0, Q=35.0):
    b, a = iirnotch(f0, Q, fs)
    return lfilter(b, a, y)

def dc_block(y, R=0.995):
    return lfilter([1, -1], [1, -R], y)
def apply_cleaning_and_resample_ram(y, sr):
    # Nhận trực tiếp mảng dữ liệu âm thanh (numpy array) thay vì đọc từ file
    y = biquad_notch(y, fs=sr, f0=HUM_HZ, Q=35.0)
    y = dc_block(y, R=0.995)
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)

    if y.size:
        peak = float(np.max(np.abs(y))) + 1e-12
        rms = float(np.sqrt(np.mean(y**2))) + 1e-12
        rms_db = 20*np.log10(rms)
        gain_db = TARGET_RMS_DBFS - rms_db
        max_gain_db = 20*np.log10((10**(-1/20)) / peak)
        gain_db = min(gain_db, max_gain_db)
        y = y * (10**(gain_db/20))

        peak = float(np.max(np.abs(y))) + 1e-12
        target_peak = 10**(PEAK_DBFS/20)
        if peak > target_peak:
            y *= (target_peak / peak)


    return y

def compute_logmel_vad(y, sr=16000, n_mels=80, eps=1e-10):
    win = int(round(0.025 * sr))
    hop = int(round(0.010 * sr))
    n_fft = 512

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
        window="hann", center=True, power=2.0,
        n_mels=n_mels, fmin=0.0, fmax=None
    )
    S = np.log(np.maximum(S, eps))
    return S

class VADProcessor:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        print(f"VAD Model loaded")

    def process_chunk(self, audio_chunk):
        mel = compute_logmel_vad(audio_chunk, sr=TARGET_SR, n_mels=80)
        feat = mel[None, :, :].astype(np.float32)

        logits = self.session.run(None, {self.input_name: feat})[0]

        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = np.squeeze(logits, 0)
        if logits.ndim == 2 and logits.shape[0] == 2:
            logits = logits.T

        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(np.clip(logits - logits_max, -40.0, 40.0))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs[:, 1]

class ASRProcessor:
    def __init__(self, model_path, vaip_config):
        self.model_path = model_path
        self.vaip_config = vaip_config
        self.session = None
        self.input_name = None

        self.lm_binary_path = _resolve_lm_path(LM_BINARY_DEFAULT)
        self.lm_beam = LM_BEAM
        self.lm_alpha = LM_ALPHA
        self.lm_beta = LM_BETA
        self.want_lm = bool(self.lm_binary_path)

        self.lm_ready = False
        self.blank_idx = None
        self.dynamic_labels = None
        self.vocab_size = None
        self.lm_decoder = None

        self.load_model()
        self.warmup()

    def load_model(self):
        try:
            self.session = create_vitisai_session(self.model_path, self.vaip_config)
            ins = self.session.get_inputs()
            self.input_name = ins[0].name if ins else 'input'
            print(f"ASR Model loaded. Input: {self.input_name}")
        except Exception as e:
            print(f"Failed to load ASR: {e}")
            self.session = None
    def warmup(self):
        """Chạy một inference giả (1 giây im lặng) để khởi động DPU và load LM trước"""
        print("warmupDPU, load LM")
        try:
            # Tạo 1 giây âm thanh hoàn toàn im lặng (mảng số 0) với tần số 16kHz
            dummy_audio = np.zeros(TARGET_SR, dtype=np.float32)

            # Chạy qua luồng xử lý audio như một câu nói bình thường
            self.process_audio(dummy_audio)

            print("Done, warmup sucess")
        except Exception as e:
            print(f"warmup error")

    def _maybe_init_lm_from_logits(self, out_probe):
        if self.lm_ready or not self.want_lm:
            return

        try:
            tv = to_TV(out_probe)
            T, V = tv.shape

            self.blank_idx = detect_blank_index_from_logits(out_probe)
            print(f"Detected blank_idx = {self.blank_idx}")

            self.dynamic_labels = build_labels_with_blank_at(self.blank_idx)
            self.vocab_size = len(self.dynamic_labels)
            print(f"Built dynamic labels: {self.vocab_size} tokens")

            self.lm_decoder = LMDecoder(
                labels=self.dynamic_labels,
                lm_binary=self.lm_binary_path,
                beam_width=self.lm_beam,
                alpha=self.lm_alpha,
                beta=self.lm_beta,
                verbose=True,
                blank_idx=self.blank_idx
            )
            self.lm_ready = bool(self.lm_decoder and self.lm_decoder.ok)
            if self.lm_ready:
                print(f"LM ready (beam={self.lm_beam})")
        except Exception as e:
            self.lm_ready = False
            print(f"LM init failed: {e}")
    def process_audio(self, audio_data):
        if not self.session:
            return "[MODEL ERROR]", "error"

        try:
        # 1. Dọn dẹp nhiễu trực tiếp trên RAM
            clean_audio = apply_cleaning_and_resample_ram(audio_data, TARGET_SR)

        # 2. Tính mel trực tiếp từ array — KHÔNG ghi file nữa
            mel = make_logmel_from_array(clean_audio, sr=TARGET_SR, n_mels=64)
            x = prepare_input_tensor(mel, 'NCT')

        # 3. Inference DPU
            out = self.session.run(None, {self.input_name: x})[0]

        # 4. Init LM lần đầu nếu cần
            if self.want_lm and not self.lm_ready:
                self._maybe_init_lm_from_logits(out)

        # 5. Decode
            decode_method = "CTC_greedy"
            if self.lm_ready and self.lm_decoder is not None:
                logits_lm = normalize_output_logits(out, vocab_size=self.vocab_size)
                hyp_raw = self.lm_decoder.decode(logits_lm, beam_width=self.lm_beam)
                if hyp_raw is not None:
                    text_raw = u_nfc(hyp_raw)
                    decode_method = f"LM_beam{self.lm_beam}"
                else:
                    logits_old = normalize_output_logits(out)
                    text_raw = u_nfc(ctc_decode_like_old(logits_old, labels=LABELS))
                    decode_method = "CTC_fallback"
            else:
                logits_old = normalize_output_logits(out)
                text_raw = u_nfc(ctc_decode_like_old(logits_old, labels=LABELS))
                decode_method = "CTC_greedy"

            text = u_nfc(text_raw.upper())
            return text, decode_method

        except Exception as e:
            print(f"ASR error: {e}")
            import traceback
            traceback.print_exc()
            return f"[ERROR: {e}]", "error"
    '''
    def process_audio(self, audio_data):
        if not self.session:
            return "[MODEL ERROR]", "error"

        try:
            temp_raw = "/tmp/temp_raw.wav"
            sf.write(temp_raw, audio_data, TARGET_SR, subtype="PCM_16")
            wav_clean = apply_cleaning_and_resample(temp_raw)

            mel = make_logmel(wav_clean, target_sr=TARGET_SR, n_mels=64)
            x = prepare_input_tensor(mel, 'NCT')

            out = self.session.run(None, {self.input_name: x})[0]

            if self.want_lm and not self.lm_ready:
                self._maybe_init_lm_from_logits(out)

            decode_method = "CTC_greedy"
            if self.lm_ready and self.lm_decoder is not None:
                logits_lm = normalize_output_logits(out, vocab_size=self.vocab_size)
                hyp_raw = self.lm_decoder.decode(logits_lm, beam_width=self.lm_beam)
                if hyp_raw is not None:
                    text_raw = u_nfc(hyp_raw)
                    decode_method = f"LM_beam{self.lm_beam}"
                else:
                    logits_old = normalize_output_logits(out)
                    text_raw = u_nfc(ctc_decode_like_old(logits_old, labels=LABELS))
                    decode_method = "CTC_fallback"
            else:
                logits_old = normalize_output_logits(out)
                text_raw = u_nfc(ctc_decode_like_old(logits_old, labels=LABELS))
                decode_method = "CTC_greedy"

            text = u_nfc(text_raw.upper())
            return text, decode_method

        except Exception as e:
            print(f"ASR error: {e}")
            import traceback
            traceback.print_exc()
            return f"[ERROR: {e}]", "error"

    def process_audio(self, audio_data):
        if not self.session:
            return "[MODEL ERROR]", "error"

        try:
            # 1. Dọn dẹp nhiễu trực tiếp trên RAM
            clean_audio = apply_cleaning_and_resample_ram(audio_data, TARGET_SR)

            # 2. Thủ thuật RAM Disk cho hàm make_logmel
            # Ghi file tạm vào /dev/shm để tránh đọc/ghi thẻ SD vật lý.
            ram_wav_path = "/dev/shm/temp_asr_input.wav"
            sf.write(ram_wav_path, clean_audio, TARGET_SR, subtype="PCM_16")

            # 3. Trích xuất đặc trưng và đưa vào model
            mel = make_logmel(ram_wav_path, target_sr=TARGET_SR, n_mels=64)
            x = prepare_input_tensor(mel, 'NCT')

            out = self.session.run(None, {self.input_name: x})[0]

            if self.want_lm and not self.lm_ready:
                self._maybe_init_lm_from_logits(out)

            decode_method = "CTC_greedy"
            if self.lm_ready and self.lm_decoder is not None:
                logits_lm = normalize_output_logits(out, vocab_size=self.vocab_size)
                hyp_raw = self.lm_decoder.decode(logits_lm, beam_width=self.lm_beam)
                if hyp_raw is not None:
                    text_raw = u_nfc(hyp_raw)
                    decode_method = f"LM_beam{self.lm_beam}"
                else:
                    logits_old = normalize_output_logits(out)
                    text_raw = u_nfc(ctc_decode_like_old(logits_old, labels=LABELS))
                    decode_method = "CTC_fallback"
            else:
                logits_old = normalize_output_logits(out)
                text_raw = u_nfc(ctc_decode_like_old(logits_old, labels=LABELS))
                decode_method = "CTC_greedy"

            text = u_nfc(text_raw.upper())
            return text, decode_method

        except Exception as e:
            print(f"ASR error: {e}")
            import traceback
            traceback.print_exc()
            return f"[ERROR: {e}]", "error"
    '''
class RealtimeRecorder:
    def __init__(self, vad_proc, asr_proc, socketio_instance):
        self.vad      = vad_proc
        self.asr      = asr_proc
        self.socketio = socketio_instance

        self.is_recording = False

        self.audio_queue: queue.Queue = queue.Queue()

        self.asr_queue: queue.Queue   = queue.Queue()

        self.pre_buffer: collections.deque = collections.deque(
            maxlen=_PRE_BUFFER_CHUNKS
        )

        # State VAD
        self.speech_buffer      = []
        self.in_speech          = False
        self.speech_start_time  = 0.0
        self.silence_duration   = 0.0
        self.last_speech_time   = 0.0

        # Pad cuối đoạn speech (số chunk)
        self._end_pad_chunks = max(1, int((VAD_SPEECH_PAD_MS / 1000) / CHUNK_DURATION))
        self._end_pad_countdown = 0   # đếm ngược khi đang pad


    def start_recording(self):
        self.is_recording = True
        self._reset_state()

        # Thread 1: đọc mic
        t_rec = threading.Thread(target=self._record_audio, daemon=True)
        # Thread 2: VAD + phân đoạn
        t_vad = threading.Thread(target=self._process_audio, daemon=True)
        # Thread 3: ASR inference  ← FIX 2 (thread riêng biệt)
        t_asr = threading.Thread(target=self._asr_worker, daemon=True)

        t_rec.start()
        t_vad.start()
        t_asr.start()

        print("[Recorder] Started: rec / VAD / ASR threads running independently")

    def stop_recording(self):
        self.is_recording = False

       
        if self.in_speech and self.speech_buffer:
            self._enqueue_segment()

        
        self.asr_queue.put(None)
        print("[Recorder] Stopped")


    def _reset_state(self):
        self.speech_buffer      = []
        self.in_speech          = False
        self.silence_duration   = 0.0
        self.last_speech_time   = 0.0
        self._end_pad_countdown = 0
        self.pre_buffer.clear()

    def _enqueue_segment(self):
       
        if not self.speech_buffer:
            return
        speech_audio = np.concatenate(self.speech_buffer)
        duration     = len(speech_audio) / TARGET_SR
        self.asr_queue.put((speech_audio, duration))
        print(f"[VAD] Enqueued segment {duration:.2f}s  |  ASR queue size={self.asr_queue.qsize()}")


    def _record_audio(self):
        chunk_samples = int(CHUNK_DURATION * REC_SR)

        cmd = [
            "arecord", "-D", AUDIO_HW,
            "-f", REC_FMT, "-c", str(REC_CH), "-r", str(REC_SR),
            "-t", "raw", "-"
        ]

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

            while self.is_recording:
                raw = proc.stdout.read(chunk_samples * REC_CH * 2)
                if not raw:
                    break

                audio = (np.frombuffer(raw, dtype=np.int16)
                           .astype(np.float32) / 32768.0)

                if REC_CH == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)

                if REC_SR != TARGET_SR:
                    audio = librosa.resample(audio,
                                             orig_sr=REC_SR,
                                             target_sr=TARGET_SR,
                                             res_type="polyphase")

                self.audio_queue.put(audio)

            proc.terminate()
            proc.wait()

        except Exception as e:
            print(f"[Record] Error: {e}")
            self.socketio.emit('error', {'message': f'Recording error: {e}'})


    def _process_audio(self):
        while self.is_recording or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # qq VAD qq
            probs            = self.vad.process_chunk(chunk)
            is_speech        = bool(np.mean(probs) > VAD_THRESHOLD)
            confidence       = float(np.mean(probs))

            self.socketio.emit('vad_status', {
                'is_speech':  is_speech,
                'confidence': confidence
            })

            now = time.time()


            if is_speech:
                # q CÓ tiếng nói q
                self.last_speech_time   = now
                self.silence_duration   = 0.0
                self._end_pad_countdown = 0   # reset pad counter

                if not self.in_speech:
                   
                    self.in_speech         = True
                    self.speech_start_time = now
                    self.speech_buffer     = list(self.pre_buffer) + [chunk]
                    print(f"[VAD] Speech START  (pre-pad {len(self.pre_buffer)} chunks)")
                else:
                    self.speech_buffer.append(chunk)

            else:
               
                if self.in_speech:
                    self.silence_duration = now - self.last_speech_time

                    if self.silence_duration < VAD_MIN_SILENCE_DURATION:
                       
                        if self._end_pad_countdown < self._end_pad_chunks:
                            self.speech_buffer.append(chunk)
                            self._end_pad_countdown += 1
                    else:
                       
                        self.in_speech = False
                        total_dur = now - self.speech_start_time

                        if total_dur >= VAD_MIN_SPEECH_DURATION:
                            self._enqueue_segment()
                        else:
                            print(f"[VAD] Skipped short segment ({total_dur:.2f}s)")

                        self.speech_buffer      = []
                        self.silence_duration   = 0.0
                        self._end_pad_countdown = 0
                self.pre_buffer.append(chunk)


    def _asr_worker(self):
        print("[ASR Worker] Ready and waiting for segments…")
        while True:
            item = self.asr_queue.get()

            # Sentinel → thoát
            if item is None:
                print("[ASR Worker] Received stop signal, exiting.")
                break

            speech_audio, duration = item
            print(f"[ASR Worker] Processing {duration:.2f}s  "
                  f"(queue remaining={self.asr_queue.qsize()})")

            try:
                t0 = time.time()
                text, decode_method = self.asr.process_audio(speech_audio)
                inf_time = time.time() - t0

                self.socketio.emit('asr_result', {
                    'text':           text,
                    'duration':       f"{duration:.2f}s",
                    'inference_time': f"{inf_time:.3f}s",
                    'timestamp':      datetime.now().strftime("%H:%M:%S"),
                    'decode_method':  decode_method
                })

                print(f"[ASR] {text}  ({inf_time:.3f}s, {decode_method})")

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.socketio.emit('error', {'message': f'ASR error: {e}'})


app = Flask(__name__,
            static_folder='static',      
            static_url_path='')          
app.config['SECRET_KEY'] = 'vad-asr-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

vad_processor = VADProcessor(VAD_MODEL)
asr_processor = ASRProcessor(ASR_MODEL, VAIP_CONFIG)
recorder = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logo_ITI.jpg')
def logo():
    return send_from_directory('static', 'logo_ITI.jpg')

@socketio.on('start_recording')
def handle_start_recording():
    global recorder
    try:
        recorder = RealtimeRecorder(vad_processor, asr_processor, socketio)
        recorder.start_recording()
        emit('recording_started', {'status': 'success', 'lm_ready': asr_processor.lm_ready})
    except Exception as e:
        emit('error', {'message': f'Failed to start: {e}'})

@socketio.on('stop_recording')
def handle_stop_recording():
    global recorder
    if recorder:
        recorder.stop_recording()
        recorder = None
    emit('recording_stopped', {'status': 'success'})

@app.route('/status')
def status():
    return jsonify({
        "vad_loaded": vad_processor.session is not None,
        "asr_loaded": asr_processor.session is not None,
        "audio_device": AUDIO_HW,
        "lm_ready": asr_processor.lm_ready,
        "lm_binary": asr_processor.lm_binary_path
    })

if __name__ == '__main__':
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True) 
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n" + "="*60)
    print("VAD + ASR Real-time Server - VNU ITI Edition")
    print("="*60)
    print(f"Audio device: {AUDIO_HW}")
    print(f"VAD model: {VAD_MODEL}")
    print(f"ASR model: {ASR_MODEL}")
    print(f"LM binary: {LM_BINARY_DEFAULT}")
    print(f"LM ready: {asr_processor.lm_ready}")
    print(f"Access: http://{local_ip}:5000")
    print("="*60 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

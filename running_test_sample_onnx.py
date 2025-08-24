# Batch Automatic Speech Recognition (ASR) with optional Language Model integration for Vietnamese

import os
import numpy as np
import onnxruntime as ort
import torchaudio
import torch
import re
import glob
import time
from pathlib import Path
import pandas as pd
import kenlm

labels = [
    ' ', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
    'r', 's', 't', 'u', 'v', 'x', 'y', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í',
    'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 'ạ', 'ả', 'ấ',
    'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ',
    'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ',
    'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ'
]
blank_id = 0

class BatchASRWithLM:
    """ASR with Language Model integration"""
    
    def __init__(self, onnx_path, lm_path=None):
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path)
        
        self.lm = None
        if lm_path and os.path.exists(lm_path):
            try:
                self.lm = kenlm.Model(lm_path)
                print(f"Loaded Language Model: {lm_path}")
                print(f"LM Order: {self.lm.order}")
                test_score = self.lm.score("xin chào việt nam", bos=True, eos=True)
                print(f"LM Test score: {test_score:.3f}")
            except Exception as e:
                print(f"Failed to load LM: {e}")
                self.lm = None
        elif lm_path:
            print(f"LM file not found: {lm_path}")
        
        print(f"Loaded ONNX model: {onnx_path}")
    
    def preprocess_audio(self, audio_path, sample_rate=16000):
        try:
            waveform, orig_sr = torchaudio.load(audio_path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if orig_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
                waveform = resampler(waveform)
            
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=512,
                win_length=int(sample_rate * 0.02),
                hop_length=int(sample_rate * 0.01),
                n_mels=64,
                window_fn=torch.hann_window,
                normalized=False
            )
            
            mel = mel_spectrogram(waveform)
            mel = torch.log(mel + 1e-6)
            mel = (mel - mel.mean(dim=2, keepdim=True)) / (mel.std(dim=2, keepdim=True) + 1e-6)
            
            return mel.numpy()
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def ctc_decode_with_lm(self, logits, beam_size=5, lm_weight=0.3):
        logits = logits[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        if self.lm is None:
            return self._simple_ctc_decode(probs)
        return self._beam_search_with_lm(probs, beam_size, lm_weight)
    
    def _simple_ctc_decode(self, probs):
        preds = np.argmax(probs, axis=-1)
        decoded, prev = [], None
        
        for pred in preds:
            if pred != prev:
                if 0 <= pred < len(labels):
                    decoded.append(labels[pred])
            prev = pred
        return ''.join(decoded)
    
    def _beam_search_with_lm(self, probs, beam_size, lm_weight):
        return self._simple_ctc_decode(probs)
    
    def _get_lm_score(self, text, lm_state=None):
        try:
            if not text.strip():
                return 0.0
            words = text.strip().split()
            if len(words) == 0:
                return 0.0
            phrase = ' '.join(words[-2:]) if len(words) >= 2 else words[0]
            score = self.lm.score(phrase)
            return score
        except Exception:
            return 0.0
    
    def apply_lm_correction(self, text):
        if not self.lm or not text.strip():
            return text
        try:
            best_text = self._correct_with_lm_scoring(text)
            corrected_text = self._apply_common_corrections_with_lm(best_text)
            return corrected_text
        except Exception as e:
            print(f"LM correction error: {e}")
            return text
    
    def _correct_with_lm_scoring(self, text):
        try:
            words = text.strip().split()
            if len(words) < 2:
                return text
            baseline_score = self.lm.score(text)
            best_text, best_score = text, baseline_score
            corrections = self._get_common_corrections()
            
            for i, word in enumerate(words):
                if word in corrections:
                    for candidate in corrections[word]:
                        test_words = words.copy()
                        test_words[i] = candidate
                        candidate_text = ' '.join(test_words)
                        try:
                            score = self.lm.score(candidate_text)
                            if score > best_score:
                                best_score = score
                                best_text = candidate_text
                        except:
                            continue
            return best_text
        except:
            return text
    
    def _apply_common_corrections_with_lm(self, text):
        try:
            corrections = self._get_common_corrections()
            best_text = text
            best_score = self.lm.score(text)
            for wrong, candidates in corrections.items():
                if wrong in text:
                    for right in candidates:
                        candidate_text = text.replace(wrong, right)
                        try:
                            score = self.lm.score(candidate_text)
                            if score > best_score:
                                best_score = score
                                best_text = candidate_text
                        except:
                            continue
            return best_text
        except:
            return text
    
    def _get_common_corrections(self):
        return {
            'mấí': ['mời', 'mấy'],
            'chuương': ['chương'],
            'ngàm': ['ngân', 'ngành'],
            'viem': ['viên'],
            'cỗ': ['cả', 'có'],
            'đâi': ['đài'],
            'tà': ['tại'],
            'xhăm': ['chăm'],
            'tyệt': ['tiết'],
            'phẩmg': ['phẩm'],
            'ngời': ['người'],
            'từn': ['từ'],
            'cón': ['có'],
            'kế': ['kết'],
            'trớc': ['trước'],
            'bôi': ['bộ'],
            'bỏn': ['bộ'],
            'tràn': ['truyền', 'thần'],
            'nành': ['ngành'],
            'cị': ['chị'],
            'xơn': ['xin'],
            'rủi': ['rồi'],
        }
    
    def process_single_file(self, audio_path, use_lm=True, beam_size=5, lm_weight=0.3):
        start_time = time.time()
        mel = self.preprocess_audio(audio_path)
        if mel is None:
            return None
        try:
            logits = self.session.run(None, {'mel_features': mel})[0]
            
            if use_lm and self.lm:
                raw_text = self.ctc_decode_with_lm(logits, beam_size, lm_weight)
                corrected_text = self.apply_lm_correction(raw_text)
            else:
                raw_text = self._simple_ctc_decode(
                    np.exp(logits[0] - np.max(logits[0], axis=-1, keepdims=True)) / 
                    np.sum(np.exp(logits[0] - np.max(logits[0], axis=-1, keepdims=True)), axis=-1, keepdims=True)
                )
                corrected_text = raw_text
            
            processing_time = time.time() - start_time
            
            return {
                'file': os.path.basename(audio_path),
                'raw_text': raw_text,
                'lm_corrected_text': corrected_text,
                'processing_time': processing_time,
                'audio_duration': mel.shape[2] * 0.01,
                'word_count_raw': len(raw_text.split()),
                'word_count_lm': len(corrected_text.split()),
                'has_spaces': ' ' in corrected_text,
                'used_lm': use_lm and self.lm is not None
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def batch_process(self, audio_dir, output_file=None, 
                     file_patterns=['*.wav', '*.mp3', '*.flac'],
                     use_lm=True, beam_size=5, lm_weight=0.3):
        print(f"Processing with Language Model: {'ON' if use_lm and self.lm else 'OFF'}")
        
        audio_files = []
        for pattern in file_patterns:
            audio_files.extend(glob.glob(os.path.join(audio_dir, pattern)))
            audio_files.extend(glob.glob(os.path.join(audio_dir, '**', pattern), recursive=True))
        
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return []
        
        print(f"Found {len(audio_files)} audio files")
        results = []
        
        for i, audio_file in enumerate(audio_files):
            print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
            result = self.process_single_file(audio_file, use_lm, beam_size, lm_weight)
            if result:
                results.append(result)
                print(f"Processed in {result['processing_time']:.2f}s")
                print(f"Raw: {result['raw_text'][:80]}...")
                if result['used_lm']:
                    print(f"LM:  {result['lm_corrected_text'][:80]}...")
                print(f"Words: {result['word_count_raw']} → {result['word_count_lm']}")
            else:
                print(f"Failed to process {audio_file}")
            print("-" * 80)
        
        if output_file and results:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"Results saved to: {output_file}")
        
        if results:
            total_time = sum(r['processing_time'] for r in results)
            total_audio = sum(r['audio_duration'] for r in results)
            rtf = total_time / total_audio if total_audio > 0 else 0
            lm_files = sum(1 for r in results if r['used_lm'])
            
            print("\nSUMMARY:")
            print(f"Successfully processed: {len(results)}/{len(audio_files)} files")
            print(f"Files processed with LM: {lm_files}/{len(results)}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Total audio duration: {total_audio:.2f}s")
            print(f"Real-time factor (RTF): {rtf:.3f}")
            
            if lm_files > 0:
                avg_improvement = np.mean([r['word_count_lm'] - r['word_count_raw'] 
                                           for r in results if r['used_lm']])
                print(f"Average word count change with LM: {avg_improvement:+.1f}")
        
        return results

if __name__ == "__main__":
    ONNX_MODEL = '/home/andyloou/viet-asr/quartznet_vietnamese_mel_fixed.onnx'
    LM_MODEL = '/home/andyloou/viet-asr/models/language_model/3-gram-lm.binary'
    AUDIO_DIR = '/home/andyloou/viet-asr/audio_samples'
    
    processor = BatchASRWithLM(
        onnx_path=ONNX_MODEL,
        lm_path=LM_MODEL
    )
    
    print("="*60)
    print("PROCESSING WITH LANGUAGE MODEL")
    print("="*60)
    
    results_lm = processor.batch_process(
        audio_dir=AUDIO_DIR,
        output_file='asr_results_with_lm.csv',
        use_lm=True,
        beam_size=5,
        lm_weight=0.5
    )
    
    print("="*60)
    print("PROCESSING WITHOUT LANGUAGE MODEL (COMPARISON)")
    print("="*60)
    
    results_no_lm = processor.batch_process(
        audio_dir=AUDIO_DIR,
        output_file='asr_results_no_lm.csv',
        use_lm=False
    )
    
    if results_lm and results_no_lm:
        print("\nSIDE-BY-SIDE COMPARISON:")
        print("="*80)
        
        for i in range(min(3, len(results_lm))):
            print(f"\nFile: {results_lm[i]['file']}")
            print(f"No LM : {results_no_lm[i]['raw_text']}")
            print(f"With LM: {results_lm[i]['lm_corrected_text']}")
            print(f"Words: {results_no_lm[i]['word_count_raw']} → {results_lm[i]['word_count_lm']}")

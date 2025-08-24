import torch
import torchaudio
import numpy as np
import os
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
import onnxruntime as ort
import librosa
from scipy.special import softmax
import time

# Sử dụng jiwer cho WER/CER calculation
try:
    from jiwer import wer, cer
except ImportError:
    print("Installing jiwer...")
    import subprocess
    subprocess.check_call(["pip", "install", "jiwer"])
    from jiwer import wer, cer

# Import NeMo components chỉ để lấy config
try:
    from omegaconf import OmegaConf
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
except ImportError as e:
    print(f"NeMo import error: {e}")
    print("Please install NeMo: pip install nemo_toolkit[asr]")
    exit(1)

class QuartzNetONNXEvaluator:
    def __init__(self, config_path: str, onnx_model_path: str, labels_path: str = None):
        """
        Khởi tạo evaluator cho QuartzNet ONNX model
        
        Args:
            config_path: Đường dẫn đến config file
            onnx_model_path: Đường dẫn đến model ONNX 
            labels_path: Đường dẫn đến file labels (optional)
        """
        self.config_path = config_path
        self.onnx_model_path = onnx_model_path
        self.labels_path = labels_path
        self.onnx_session = None
        self.labels = []
        self.preprocessor = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Kiểm tra ONNX Runtime providers
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Using CUDA for ONNX inference")
        else:
            self.providers = ['CPUExecutionProvider']
            print("Using CPU for ONNX inference")
        
    def load_labels(self):
        """Load labels từ config hoặc file riêng"""
        try:
            if self.labels_path and os.path.exists(self.labels_path):
                # Load từ file labels riêng
                if self.labels_path.endswith('.json'):
                    with open(self.labels_path, 'r', encoding='utf-8') as f:
                        labels_data = json.load(f)
                        if isinstance(labels_data, list):
                            self.labels = labels_data
                        elif isinstance(labels_data, dict) and 'labels' in labels_data:
                            self.labels = labels_data['labels']
                else:
                    # Text file, mỗi dòng một label
                    with open(self.labels_path, 'r', encoding='utf-8') as f:
                        self.labels = [line.strip() for line in f.readlines()]
            else:
                # Load từ config file
                if os.path.exists(self.config_path):
                    cfg = OmegaConf.load(self.config_path)
                    if 'labels' in cfg:
                        self.labels = cfg.labels
                    else:
                        # Default Vietnamese labels
                        self.labels = [
                            ' ', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 
                            'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 
                            'ạ', 'ả', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 
                            'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ'
                        ]
            
            # Tạo mapping
            self.char_to_idx = {char: idx for idx, char in enumerate(self.labels)}
            self.idx_to_char = {idx: char for idx, char in enumerate(self.labels)}
            
            print(f"Loaded {len(self.labels)} labels")
            print(f"Labels: {self.labels[:10]}..." if len(self.labels) > 10 else f"Labels: {self.labels}")
            
        except Exception as e:
            print(f"Error loading labels: {str(e)}")
            raise
    
    def setup_preprocessor(self):
        """Setup audio preprocessor từ config"""
        try:
            cfg = OmegaConf.load(self.config_path)
            
            # Default preprocessor config
            preprocessor_cfg = {
                'sample_rate': 16000,
                'window_size': 0.02,
                'window_stride': 0.01, 
                'window': 'hann',
                'normalize': 'per_feature',
                'n_fft': 512,
                'features': 64,
                'dither': 1e-5,
                'pad_to': 16,
                'stft_conv': False
            }
            
            # Override với config nếu có
            if 'preprocessor' in cfg:
                if 'AudioToMelSpectrogramPreprocessor' in cfg.preprocessor:
                    preprocessor_cfg.update(cfg.preprocessor.AudioToMelSpectrogramPreprocessor)
                elif hasattr(cfg.preprocessor, '_target_') and 'AudioToMelSpectrogramPreprocessor' in cfg.preprocessor._target_:
                    # Handle newer NeMo config format
                    for key, value in cfg.preprocessor.items():
                        if key != '_target_':
                            preprocessor_cfg[key] = value
            
            # Tạo preprocessor với kwargs thay vì cfg
            self.preprocessor = AudioToMelSpectrogramPreprocessor(
                sample_rate=preprocessor_cfg['sample_rate'],
                window_size=preprocessor_cfg['window_size'],
                window_stride=preprocessor_cfg['window_stride'],
                window=preprocessor_cfg['window'],
                normalize=preprocessor_cfg['normalize'],
                n_fft=preprocessor_cfg['n_fft'],
                features=preprocessor_cfg['features'],
                dither=preprocessor_cfg['dither'],
                pad_to=preprocessor_cfg['pad_to'],
                stft_conv=preprocessor_cfg.get('stft_conv', False)
            )
            
            print(f"Preprocessor setup completed")
            print(f"  Sample rate: {preprocessor_cfg['sample_rate']}")
            print(f"  Features: {preprocessor_cfg['features']}")
            print(f"  Window size: {preprocessor_cfg['window_size']}")
            
        except Exception as e:
            print(f"Error setting up preprocessor: {str(e)}")
            # Fallback: create preprocessor with default values
            print("Using fallback preprocessor with default settings...")
            try:
                self.preprocessor = AudioToMelSpectrogramPreprocessor(
                    sample_rate=16000,
                    window_size=0.02,
                    window_stride=0.01,
                    window='hann',
                    normalize='per_feature',
                    n_fft=512,
                    features=64,
                    dither=1e-5,
                    pad_to=16
                )
                print("Fallback preprocessor created successfully")
            except Exception as fallback_error:
                print(f"Fallback preprocessor also failed: {fallback_error}")
                raise
    
    def load_onnx_model(self):
        """Load ONNX model"""
        try:
            print(f"Loading ONNX model: {self.onnx_model_path}")
            
            if not os.path.exists(self.onnx_model_path):
                raise FileNotFoundError(f"ONNX model not found: {self.onnx_model_path}")
            
            # Tạo ONNX Runtime session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.onnx_session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options,
                providers=self.providers
            )
            
            # In thông tin về model
            print(f"Model loaded successfully")
            print(f"Input names: {[input.name for input in self.onnx_session.get_inputs()]}")
            print(f"Output names: {[output.name for output in self.onnx_session.get_outputs()]}")
            
            # In input shapes
            for input_meta in self.onnx_session.get_inputs():
                print(f"  Input '{input_meta.name}': {input_meta.shape}")
            
            # In output shapes  
            for output_meta in self.onnx_session.get_outputs():
                print(f"  Output '{output_meta.name}': {output_meta.shape}")
                
        except Exception as e:
            print(f"Error loading ONNX model: {str(e)}")
            raise
    
    def load_model(self):
        """Load tất cả components"""
        print("Loading model components...")
        self.load_labels()
        self.setup_preprocessor() 
        self.load_onnx_model()
        print("All components loaded successfully!")
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocess audio file thành mel spectrogram"""
        try:
            # Load audio
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample nếu cần
            target_sr = 16000
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Chuẩn bị input cho preprocessor
            input_signal = waveform
            input_length = torch.tensor([waveform.shape[1]], dtype=torch.long)
            
            # Preprocess với NeMo preprocessor
            with torch.no_grad():
                processed, processed_length = self.preprocessor(
                    input_signal=input_signal,
                    length=input_length
                )
            
            # Convert to numpy
            processed_np = processed.cpu().numpy()
            processed_length_np = processed_length.cpu().numpy()
            
            return processed_np, processed_length_np
            
        except Exception as e:
            print(f"Error preprocessing audio {audio_path}: {str(e)}")
            raise
    
    def decode_predictions(self, logits: np.ndarray) -> str:
        """Decode CTC logits thành text sử dụng greedy decoding"""
        try:
            # Apply softmax
            probs = softmax(logits, axis=-1)
            
            # Greedy decoding
            predicted_ids = np.argmax(probs, axis=-1)
            
            # Remove consecutive duplicates và blank token (thường là index 0)
            decoded = []
            prev_id = -1
            
            for pred_id in predicted_ids[0]:  # Lấy sequence đầu tiên
                if pred_id != prev_id and pred_id != 0:  # 0 là blank token
                    if pred_id < len(self.labels):
                        decoded.append(self.labels[pred_id])
                prev_id = pred_id
            
            result = ''.join(decoded).strip()
            return result
            
        except Exception as e:
            print(f"Error decoding predictions: {str(e)}")
            return ""
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe một file audio và trả về kết quả + timing"""
        try:
            print(f"Transcribing: {os.path.basename(audio_path)}")
            
            start_time = time.time()
            
            # Preprocess audio
            preprocess_start = time.time()
            mel_spec, spec_length = self.preprocess_audio(audio_path)
            preprocess_time = time.time() - preprocess_start
            
            print(f"  Preprocessed shape: {mel_spec.shape}")
            print(f"  Preprocessing time: {preprocess_time:.3f}s")
            
            # ONNX inference
            inference_start = time.time()
            
            # Chuẩn bị input cho ONNX model
            input_names = [input.name for input in self.onnx_session.get_inputs()]
            
            if len(input_names) == 1:
                # Chỉ có audio input
                onnx_inputs = {input_names[0]: mel_spec}
            else:
                # Có cả audio và length
                onnx_inputs = {
                    input_names[0]: mel_spec,
                    input_names[1]: spec_length
                }
            
            # Run inference
            outputs = self.onnx_session.run(None, onnx_inputs)
            inference_time = time.time() - inference_start
            
            print(f"  ONNX inference time: {inference_time:.3f}s")
            print(f"  Output shape: {outputs[0].shape}")
            
            # Decode predictions
            decode_start = time.time()
            transcript = self.decode_predictions(outputs[0])
            decode_time = time.time() - decode_start
            
            total_time = time.time() - start_time
            
            print(f"  Decoding time: {decode_time:.3f}s")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Result: '{transcript}'")
            
            return {
                'transcript': transcript,
                'timing': {
                    'preprocess_time': preprocess_time,
                    'inference_time': inference_time,
                    'decode_time': decode_time,
                    'total_time': total_time
                },
                'audio_info': {
                    'mel_shape': mel_spec.shape,
                    'spec_length': spec_length.tolist() if hasattr(spec_length, 'tolist') else spec_length
                }
            }
            
        except Exception as e:
            print(f"  Error transcribing {audio_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'transcript': "",
                'error': str(e),
                'timing': {'total_time': 0}
            }
    
    def load_ground_truth(self, gt_file: str) -> Dict[str, str]:
        """Load ground truth từ file"""
        ground_truth = {}
        
        try:
            if not os.path.exists(gt_file):
                print(f"Ground truth file not found: {gt_file}")
                return ground_truth
                
            if gt_file.endswith('.json'):
                with open(gt_file, 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)
                print(f"Loaded {len(ground_truth)} ground truth entries from JSON")
            else:
                # Text format: filename|transcript
                with open(gt_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '|' in line:
                            filename, text = line.split('|', 1)
                            ground_truth[filename.strip()] = text.strip()
                        else:
                            print(f"Warning: Invalid format at line {line_num}: {line}")
                print(f"Loaded {len(ground_truth)} ground truth entries from text file")
                        
        except Exception as e:
            print(f"Error loading ground truth: {str(e)}")
            
        return ground_truth
    
    def evaluate_on_dataset(self, audio_dir: str, ground_truth_file: str = None) -> Dict:
        """Đánh giá model trên dataset"""
        if self.onnx_session is None:
            print("Loading model first...")
            self.load_model()
        
        # Tìm tất cả file audio
        audio_files = []
        supported_formats = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        
        print(f"Searching for audio files in: {audio_dir}")
        for ext in supported_formats:
            found_files = list(Path(audio_dir).glob(ext))
            audio_files.extend(found_files)
            if found_files:
                print(f"  Found {len(found_files)} {ext} files")
        
        audio_files = sorted(audio_files)
        print(f"Total audio files found: {len(audio_files)}")
        
        if len(audio_files) == 0:
            print("No audio files found!")
            return {'error': 'No audio files found'}
        
        # Load ground truth nếu có
        ground_truth = {}
        if ground_truth_file:
            ground_truth = self.load_ground_truth(ground_truth_file)
        
        # Khởi tạo results
        results = {
            'transcriptions': {},
            'metrics': {},
            'timing_stats': {},
            'errors': [],
            'config': {
                'onnx_model': self.onnx_model_path,
                'audio_dir': audio_dir,
                'ground_truth_file': ground_truth_file,
                'num_files': len(audio_files),
                'providers': self.providers,
                'num_labels': len(self.labels)
            }
        }
        
        predictions = []
        references = []
        all_timings = {
            'preprocess_times': [],
            'inference_times': [],
            'decode_times': [],
            'total_times': []
        }
        
        print(f"\nStarting transcription of {len(audio_files)} files...")
        print("=" * 60)
        
        for i, audio_file in enumerate(audio_files):
            filename = audio_file.name
            print(f"\n[{i+1}/{len(audio_files)}] Processing: {filename}")
            
            try:
                # Transcribe
                result = self.transcribe_audio(str(audio_file))
                transcript = result['transcript']
                
                results['transcriptions'][filename] = {
                    'transcript': transcript,
                    'timing': result.get('timing', {}),
                    'audio_info': result.get('audio_info', {})
                }
                
                # Lưu timing stats
                if 'timing' in result:
                    timing = result['timing']
                    all_timings['preprocess_times'].append(timing.get('preprocess_time', 0))
                    all_timings['inference_times'].append(timing.get('inference_time', 0))
                    all_timings['decode_times'].append(timing.get('decode_time', 0))
                    all_timings['total_times'].append(timing.get('total_time', 0))
                
                # So sánh với ground truth nếu có
                if filename in ground_truth:
                    gt_text = ground_truth[filename]
                    predictions.append(transcript)
                    references.append(gt_text)
                    
                    # Tính WER và CER cho file này
                    file_wer = wer(gt_text, transcript) * 100
                    file_cer = cer(gt_text, transcript) * 100
                    
                    results['metrics'][filename] = {
                        'wer': file_wer,
                        'cer': file_cer,
                        'ground_truth': gt_text,
                        'prediction': transcript,
                        'timing': result.get('timing', {})
                    }
                    
                    print(f"  Ground truth: {gt_text}")
                    print(f"  Prediction:   {transcript}")
                    print(f"  WER: {file_wer:.2f}%, CER: {file_cer:.2f}%")
                else:
                    if ground_truth:
                        print(f"  Warning: No ground truth for {filename}")
                    
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                results['errors'].append(error_msg)
                print(f"  ERROR: {error_msg}")
        
        # Tính timing statistics
        if all_timings['total_times']:
            results['timing_stats'] = {
                'preprocess': {
                    'mean': np.mean(all_timings['preprocess_times']),
                    'std': np.std(all_timings['preprocess_times']),
                    'min': np.min(all_timings['preprocess_times']),
                    'max': np.max(all_timings['preprocess_times'])
                },
                'inference': {
                    'mean': np.mean(all_timings['inference_times']),
                    'std': np.std(all_timings['inference_times']),
                    'min': np.min(all_timings['inference_times']),
                    'max': np.max(all_timings['inference_times'])
                },
                'decode': {
                    'mean': np.mean(all_timings['decode_times']),
                    'std': np.std(all_timings['decode_times']),
                    'min': np.min(all_timings['decode_times']),
                    'max': np.max(all_timings['decode_times'])
                },
                'total': {
                    'mean': np.mean(all_timings['total_times']),
                    'std': np.std(all_timings['total_times']),
                    'min': np.min(all_timings['total_times']),
                    'max': np.max(all_timings['total_times']),
                    'total_time': np.sum(all_timings['total_times'])
                }
            }
        
        # Tính overall metrics
        if predictions and references:
            print(f"\n" + "=" * 60)
            print("CALCULATING OVERALL METRICS...")
            
            overall_wer = wer(references, predictions) * 100
            overall_cer = cer(references, predictions) * 100
            
            results['overall_metrics'] = {
                'wer': overall_wer,
                'cer': overall_cer,
                'num_files_with_gt': len(predictions),
                'num_total_files': len(audio_files),
                'num_errors': len(results['errors'])
            }
            
            print(f"Files with ground truth: {len(predictions)}/{len(audio_files)}")
            print(f"Overall WER: {overall_wer:.2f}%")
            print(f"Overall CER: {overall_cer:.2f}%")
            print(f"Processing errors: {len(results['errors'])}")
            
            # Thống kê chi tiết
            if len(predictions) > 1:
                wer_values = [results['metrics'][f]['wer'] for f in results['metrics']]
                cer_values = [results['metrics'][f]['cer'] for f in results['metrics']]
                
                print(f"\nDetailed Statistics:")
                print(f"WER - Min: {min(wer_values):.2f}%, Max: {max(wer_values):.2f}%, Mean: {np.mean(wer_values):.2f}%, Std: {np.std(wer_values):.2f}%")
                print(f"CER - Min: {min(cer_values):.2f}%, Max: {max(cer_values):.2f}%, Mean: {np.mean(cer_values):.2f}%, Std: {np.std(cer_values):.2f}%")
        else:
            print(f"\n" + "=" * 60)
            print(f"TRANSCRIPTION COMPLETED")
            print(f"Total files processed: {len(audio_files)}")
            print(f"Successful transcriptions: {len([t for t in results['transcriptions'].values() if t['transcript']])}")
            print(f"Errors: {len(results['errors'])}")
        
        # In timing statistics
        if results['timing_stats']:
            print(f"\nTIMING STATISTICS:")
            stats = results['timing_stats']
            print(f"Average inference time: {stats['inference']['mean']:.3f}s ± {stats['inference']['std']:.3f}s")
            print(f"Average total time: {stats['total']['mean']:.3f}s ± {stats['total']['std']:.3f}s")
            print(f"Total processing time: {stats['total']['total_time']:.2f}s")
            if len(audio_files) > 0:
                rtf = stats['total']['total_time'] / len(audio_files)  # Rough estimate
                print(f"Approximate RTF: {rtf:.3f}")
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Lưu kết quả ra file JSON"""
        try:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    def create_csv_report(self, results: Dict, output_file: str):
        """Tạo báo cáo CSV"""
        try:
            if 'metrics' not in results or not results['metrics']:
                # Chỉ có transcriptions
                df_data = []
                for filename, data in results['transcriptions'].items():
                    transcript = data['transcript'] if isinstance(data, dict) else data
                    timing_info = data.get('timing', {}) if isinstance(data, dict) else {}
                    
                    df_data.append({
                        'filename': filename,
                        'transcript': transcript,
                        'transcript_length': len(transcript) if transcript else 0,
                        'inference_time': timing_info.get('inference_time', 0),
                        'total_time': timing_info.get('total_time', 0),
                        'has_error': filename in [e.split(':')[0].replace('Error processing ', '') for e in results.get('errors', [])]
                    })
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('filename')
                
            else:
                # Có metrics
                df_data = []
                for filename, metrics in results['metrics'].items():
                    df_data.append({
                        'filename': filename,
                        'wer_percent': round(metrics['wer'], 2),
                        'cer_percent': round(metrics['cer'], 2),
                        'ground_truth': metrics['ground_truth'],
                        'prediction': metrics['prediction'],
                        'ground_truth_length': len(metrics['ground_truth']),
                        'prediction_length': len(metrics['prediction']),
                        'inference_time': metrics.get('timing', {}).get('inference_time', 0),
                        'total_time': metrics.get('timing', {}).get('total_time', 0)
                    })
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('wer_percent')  # Sort by WER
            
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"CSV report saved to: {output_file}")
            
            return df
            
        except Exception as e:
            print(f"Error creating CSV report: {str(e)}")
            return None

def compare_models(models_info: List[Dict], audio_dir: str, ground_truth_file: str = None):
    """So sánh nhiều model ONNX"""
    print("="*80)
    print("COMPARING MULTIPLE ONNX MODELS")
    print("="*80)
    
    all_results = {}
    
    for model_info in models_info:
        model_name = model_info['name']
        print(f"\n{'='*20} TESTING {model_name} {'='*20}")
        
        evaluator = QuartzNetONNXEvaluator(
            config_path=model_info['config_path'],
            onnx_model_path=model_info['onnx_path'],
            labels_path=model_info.get('labels_path', None)
        )
        
        results = evaluator.evaluate_on_dataset(audio_dir, ground_truth_file)
        all_results[model_name] = results
        
        # Lưu kết quả riêng cho từng model
        evaluator.save_results(results, f'results_{model_name}.json')
        evaluator.create_csv_report(results, f'results_{model_name}.csv')
    
    # So sánh tổng hợp
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_data = []
    for model_name, results in all_results.items():
        if 'overall_metrics' in results:
            metrics = results['overall_metrics']
            timing = results.get('timing_stats', {}).get('inference', {})
            
            comparison_data.append({
                'Model': model_name,
                'WER (%)': f"{metrics['wer']:.2f}",
                'CER (%)': f"{metrics['cer']:.2f}", 
                'Avg Inference Time (s)': f"{timing.get('mean', 0):.3f}",
                'Files Processed': f"{len(results.get('transcriptions', {}))}/{metrics['num_total_files']}",
                'Errors': len(results.get('errors', []))
            })
        else:
            timing = results.get('timing_stats', {}).get('inference', {})
            
            comparison_data.append({
                'Model': model_name,
                'WER (%)': 'N/A',
                'CER (%)': 'N/A', 
                'Avg Inference Time (s)': f"{timing.get('mean', 0):.3f}",
                'Files Processed': f"{len(results.get('transcriptions', {}))}/N/A",
                'Errors': len(results.get('errors', []))
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Lưu bảng so sánh
        df_comparison.to_csv('model_comparison.csv', index=False)
        print(f"\nComparison saved to: model_comparison.csv")
    
    return all_results

# Script chính
if __name__ == "__main__":
    # Cấu hình cho một model
    print("Choose evaluation mode:")
    print("1. Single model evaluation")
    print("2. Compare multiple models")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Đánh giá một model
        config_path = '/home/andyloou/viet-asr/quartznet12x1_vi_encoder_decoder_metadata.json'
        onnx_model_path = 'quartznet12x1_vi_encoder_decoder.onnx'  # Hoặc 'quan_quartz.onnx'
        labels_path = None  # Có thể chỉ định file labels riêng nếu cần
        
        print("Initializing QuartzNet ONNX Evaluator...")
        evaluator = QuartzNetONNXEvaluator(
            config_path=config_path,
            onnx_model_path=onnx_model_path,
            labels_path=labels_path
        )
        
        # Đánh giá trên dataset
        print("\nStarting evaluation...")
        results = evaluator.evaluate_on_dataset(
            audio_dir='audio_samples',
            ground_truth_file="ground_truth.txt"
        )
        
        # Lưu kết quả
        print("\nSaving results...")
        model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
        evaluator.save_results(results, f'evaluation_results_{model_name}.json')
        evaluator.create_csv_report(results, f'evaluation_results_{model_name}.csv')
        
        print(f"\nEvaluation completed for {model_name}!")
        
    elif choice == "2":
        # So sánh nhiều model
        models_to_compare = [
            {
                'name': 'original_fp32',
                'config_path': 'configs/quartznet12x1_vi.yaml',
                'onnx_path': '/home/andyloou/viet-asr/quartznet_asr_correct.onnx',
                'labels_path': None
            },
            {
                'name': 'quantized_int8', 
                'config_path': 'configs/quartznet12x1_vi.yaml',
                'onnx_path': 'quan_quartz.onnx',
                'labels_path': None
            }
        ]
        
        # Kiểm tra file tồn tại
        valid_models = []
        for model_info in models_to_compare:
            if os.path.exists(model_info['onnx_path']) and os.path.exists(model_info['config_path']):
                valid_models.append(model_info)
                print(f"✓ Found model: {model_info['name']} - {model_info['onnx_path']}")
            else:
                print(f"✗ Missing files for model: {model_info['name']}")
                if not os.path.exists(model_info['onnx_path']):
                    print(f"  Missing ONNX: {model_info['onnx_path']}")
                if not os.path.exists(model_info['config_path']):
                    print(f"  Missing config: {model_info['config_path']}")
        
        if len(valid_models) == 0:
            print("No valid models found!")
            exit(1)
        
        print(f"\nComparing {len(valid_models)} models...")
        all_results = compare_models(
            models_info=valid_models,
            audio_dir='audio_samples',
            ground_truth_file='ground_truth.txt'
        )
        
        print("\nComparison completed!")
        
    else:
        print("Invalid choice!")
        
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print("Files created:")
    print("- evaluation_results_*.json: Detailed results")
    print("- evaluation_results_*.csv: Tabular results")
    if choice == "2":
        print("- model_comparison.csv: Model comparison table")
    print("\nCheck the timing statistics to compare inference speed between models!")
    
    # Thêm hướng dẫn sử dụng
    print(f"\n{'='*60}")
    print("USAGE NOTES")
    print("="*60)
    print("1. Make sure you have the required files:")
    print("   - configs/quartznet12x1_vi.yaml")
    print("   - quartznet_asr_correct.onnx")
    print("   - quan_quartz.onnx")
    print("   - audio_samples/ directory with test audio files")
    print("   - ground_truth.txt (optional, for WER/CER calculation)")
    print("")
    print("2. Ground truth file format (text file):")
    print("   filename.wav|transcription text")
    print("   example.wav|xin chào thế giới")
    print("")
    print("3. Install required packages:")
    print("   pip install onnxruntime onnxruntime-gpu")
    print("   pip install jiwer librosa scipy")
    print("   pip install nemo_toolkit[asr]")
    print("")
    print("4. The quantized model should be faster but may have slightly lower accuracy")
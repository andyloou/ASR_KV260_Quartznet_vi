import torch
import torchaudio
import numpy as np
import os
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple

# Sử dụng jiwer cho WER/CER calculation
try:
    from jiwer import wer, cer
except ImportError:
    print("Installing jiwer...")
    import subprocess
    subprocess.check_call(["pip", "install", "jiwer"])
    from jiwer import wer, cer

# Import NeMo components
try:
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import EncDecCTCModel
    from nemo.collections.asr.modules import ConvASREncoder, ConvASRDecoder
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
except ImportError as e:
    print(f"NeMo import error: {e}")
    print("Please install NeMo: pip install nemo_toolkit[asr]")
    exit(1)

class QuartzNetEvaluatorFixed:
    def __init__(self, config_path: str, encoder_checkpoint: str, 
                 decoder_checkpoint: str, lm_path: str = None):
        """
        Khởi tạo evaluator cho QuartzNet model với config structure phẳng
        """
        self.config_path = config_path
        self.encoder_checkpoint = encoder_checkpoint
        self.decoder_checkpoint = decoder_checkpoint
        self.lm_path = lm_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def create_model_config(self, flat_cfg):
        """Tạo model config từ flat config"""
        
        # Lấy labels từ config
        labels = flat_cfg.labels if 'labels' in flat_cfg else [
            ' ', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 
            'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 
            'ạ', 'ả', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 
            'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ'
        ]
        
        # Tạo structured config
        model_cfg = OmegaConf.create({
            'sample_rate': 16000,
            'labels': labels,
            
            # Preprocessor config
            'preprocessor': {
                '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                'sample_rate': flat_cfg.AudioToMelSpectrogramPreprocessor.sample_rate if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 16000,
                'window_size': flat_cfg.AudioToMelSpectrogramPreprocessor.window_size if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 0.02,
                'window_stride': flat_cfg.AudioToMelSpectrogramPreprocessor.window_stride if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 0.01,
                'window': flat_cfg.AudioToMelSpectrogramPreprocessor.window if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 'hann',
                'normalize': flat_cfg.AudioToMelSpectrogramPreprocessor.normalize if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 'per_feature',
                'n_fft': flat_cfg.AudioToMelSpectrogramPreprocessor.n_fft if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 512,
                'features': flat_cfg.AudioToMelSpectrogramPreprocessor.features if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 64,
                'dither': flat_cfg.AudioToMelSpectrogramPreprocessor.dither if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 1e-5,
                'pad_to': flat_cfg.AudioToMelSpectrogramPreprocessor.pad_to if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 16,
                'stft_conv': flat_cfg.AudioToMelSpectrogramPreprocessor.stft_conv if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else False
            },
            
            # Spec augmentation (optional)
            'spec_augment': {
                '_target_': 'nemo.collections.asr.modules.SpectrogramAugmentation',
                'rect_masks': flat_cfg.SpectrogramAugmentation.rect_masks if 'SpectrogramAugmentation' in flat_cfg else 0,
                'rect_time': flat_cfg.SpectrogramAugmentation.rect_time if 'SpectrogramAugmentation' in flat_cfg else 120,
                'rect_freq': flat_cfg.SpectrogramAugmentation.rect_freq if 'SpectrogramAugmentation' in flat_cfg else 50,
            } if 'SpectrogramAugmentation' in flat_cfg else None,
            
            # Encoder config
            'encoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
                'feat_in': flat_cfg.AudioToMelSpectrogramPreprocessor.features if 'AudioToMelSpectrogramPreprocessor' in flat_cfg else 64,
                'activation': flat_cfg.JasperEncoder.activation if 'JasperEncoder' in flat_cfg else 'relu',
                'conv_mask': flat_cfg.JasperEncoder.conv_mask if 'JasperEncoder' in flat_cfg else True,
                'jasper': flat_cfg.JasperEncoder.jasper if 'JasperEncoder' in flat_cfg else []
            },
            
            # Decoder config
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': 1024,  # Từ layer cuối của jasper
                'num_classes': len(labels),
                'vocabulary': labels
            },
            
            # Loss function
            'loss': {
                '_target_': 'nemo.collections.asr.losses.CTCLoss',
                'num_classes': len(labels),
                'zero_infinity': True,
                'reduction': 'mean_batch'
            },
            
            # Optimization (not needed for inference nhưng có thể cần)
            'optim': {
                'name': 'novograd',
                'lr': 0.01,
                'betas': [0.8, 0.25],
                'weight_decay': 0.001
            }
        })
        
        return model_cfg
    
    def load_model(self):
        """Load model từ config và checkpoint"""
        try:
            print("Loading model configuration...")
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
            # Load flat config
            flat_cfg = OmegaConf.load(self.config_path)
            print("Config loaded successfully")
            print(f"Config keys: {list(flat_cfg.keys())}")
            
            # Tạo structured model config
            print("Creating structured model config...")
            model_cfg = self.create_model_config(flat_cfg)
            
            # Tạo model
            print("Creating model...")
            self.model = EncDecCTCModel(cfg=model_cfg)
            
            # Load checkpoints
            print("Loading encoder checkpoint...")
            if not os.path.exists(self.encoder_checkpoint):
                raise FileNotFoundError(f"Encoder checkpoint not found: {self.encoder_checkpoint}")
            
            encoder_state = torch.load(self.encoder_checkpoint, map_location=self.device)
            print(f"Encoder checkpoint keys (first 5): {list(encoder_state.keys())[:5]}")
            
            # Load encoder với strict=False để bỏ qua các keys không match
            missing_keys, unexpected_keys = self.model.encoder.load_state_dict(encoder_state, strict=False)
            if missing_keys:
                print(f"Missing encoder keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected encoder keys: {len(unexpected_keys)} keys")
            
            print("Loading decoder checkpoint...")
            if not os.path.exists(self.decoder_checkpoint):
                raise FileNotFoundError(f"Decoder checkpoint not found: {self.decoder_checkpoint}")
            
            decoder_state = torch.load(self.decoder_checkpoint, map_location=self.device)
            print(f"Decoder checkpoint keys (first 5): {list(decoder_state.keys())[:5]}")
            
            # Load decoder với strict=False
            missing_keys, unexpected_keys = self.model.decoder.load_state_dict(decoder_state, strict=False)
            if missing_keys:
                print(f"Missing decoder keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected decoder keys: {len(unexpected_keys)} keys")
            
            # Move model to device và set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe một file audio"""
        try:
            print(f"Transcribing: {os.path.basename(audio_path)}")
            
            # Load audio file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"  Original: {sample_rate}Hz, shape: {waveform.shape}")
            
            # Resample to 16kHz if needed
            target_sr = 16000
            if sample_rate != target_sr:
                print(f"  Resampling to {target_sr}Hz")
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                print("  Converting to mono")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to numpy array cho NeMo transcribe
            audio_signal = waveform.squeeze().cpu().numpy()
            
            # Transcribe
            with torch.no_grad():
                # Sử dụng method transcribe của model
                result = self.model.transcribe([audio_signal])
                
                # Extract text từ result
                if isinstance(result, list) and len(result) > 0:
                    transcript_obj = result[0]
                    
                    # Kiểm tra các cách extract text
                    if hasattr(transcript_obj, 'text'):
                        transcript = transcript_obj.text
                    elif hasattr(transcript_obj, 'prediction'):
                        transcript = transcript_obj.prediction
                    elif isinstance(transcript_obj, dict) and 'text' in transcript_obj:
                        transcript = transcript_obj['text']
                    elif isinstance(transcript_obj, str):
                        transcript = transcript_obj
                    else:
                        # In ra structure để debug
                        print(f"  Result structure: {type(transcript_obj)}")
                        if hasattr(transcript_obj, '__dict__'):
                            print(f"  Attributes: {list(transcript_obj.__dict__.keys())}")
                        transcript = str(transcript_obj)
                else:
                    transcript = ""
            
            transcript = transcript.strip() if transcript else ""
            print(f"  Result: '{transcript}'")
            return transcript
            
        except Exception as e:
            print(f"  Error transcribing {audio_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
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
        if self.model is None:
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
            'errors': [],
            'config': {
                'audio_dir': audio_dir,
                'ground_truth_file': ground_truth_file,
                'num_files': len(audio_files),
                'device': str(self.device)
            }
        }
        
        predictions = []
        references = []
        
        print(f"\nStarting transcription of {len(audio_files)} files...")
        print("=" * 60)
        
        for i, audio_file in enumerate(audio_files):
            filename = audio_file.name
            print(f"\n[{i+1}/{len(audio_files)}] Processing: {filename}")
            
            try:
                # Transcribe
                transcript = self.transcribe_audio(str(audio_file))
                results['transcriptions'][filename] = transcript
                
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
                        'prediction': transcript
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
            print(f"Successful transcriptions: {len(results['transcriptions'])}")
            print(f"Errors: {len(results['errors'])}")
        
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
                for filename, transcript in results['transcriptions'].items():
                    df_data.append({
                        'filename': filename,
                        'transcript': transcript,
                        'transcript_length': len(transcript),
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
                        'prediction_length': len(metrics['prediction'])
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

# Script chính
if __name__ == "__main__":
    # Cấu hình paths
    config_path = 'configs/quartznet12x1_vi.yaml'
    encoder_checkpoint = 'models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt'
    decoder_checkpoint = 'models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt'
    lm_path = 'models/language_model/3-gram-lm.binary'
    
    # Tạo evaluator
    print("Initializing QuartzNet Evaluator...")
    evaluator = QuartzNetEvaluatorFixed(
        config_path=config_path,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
        lm_path=lm_path
    )
    
    # Đánh giá trên dataset
    print("\nStarting evaluation...")
    results = evaluator.evaluate_on_dataset(
        audio_dir='audio_samples',
        ground_truth_file="ground_truth.txt"  # Thay bằng đường dẫn file ground truth nếu có
    )
    
    # Lưu kết quả
    print("\nSaving results...")
    evaluator.save_results(results, 'evaluation_results.json')
    evaluator.create_csv_report(results, 'evaluation_results.csv')
    
    print("\nEvaluation completed!")
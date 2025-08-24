import torch
import torchaudio
import numpy as np
import os
import json
import yaml
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
from collections import OrderedDict

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

class QuartzNetModelAnalyzer:
    def __init__(self, config_path: str, encoder_path: str, decoder_path: str, lm_path: str = None):
        """
        Khởi tạo analyzer cho QuartzNet model từ các component riêng biệt
        """
        self.config_path = config_path
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.lm_path = lm_path
        self.model = None
        self.config = None
        self.encoder_checkpoint = None
        self.decoder_checkpoint = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def check_files_exist(self):
        """Kiểm tra xem tất cả các file cần thiết có tồn tại không"""
        files_to_check = {
            'Config': self.config_path,
            'Encoder': self.encoder_path,
            'Decoder': self.decoder_path
        }
        
        if self.lm_path:
            files_to_check['Language Model'] = self.lm_path
        
        print("Checking file existence...")
        missing_files = []
        
        for file_type, file_path in files_to_check.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"✓ {file_type}: {file_path} ({file_size:.2f} MB)")
            else:
                print(f"✗ {file_type}: {file_path} (NOT FOUND)")
                missing_files.append((file_type, file_path))
        
        if missing_files:
            print(f"\n⚠️  Missing {len(missing_files)} file(s):")
            for file_type, file_path in missing_files:
                print(f"   - {file_type}: {file_path}")
            return False
        
        print("✓ All required files found!")
        return True
        
    def load_config(self):
        """Load config từ YAML file với xử lý lỗi tốt hơn"""
        try:
            print(f"Loading config from: {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print("✓ Config loaded successfully")
            
            # Debug: In cấu trúc config
            print(f"Config type: {type(self.config)}")
            if isinstance(self.config, dict):
                print(f"Config top-level keys: {list(self.config.keys())}")
            else:
                print(f"Config content (first 200 chars): {str(self.config)[:200]}")
            
            # In thông tin cơ bản về config
            if isinstance(self.config, dict):
                if 'model' in self.config:
                    model_config = self.config['model']
                    print(f"Model config type: {type(model_config)}")
                    
                    if isinstance(model_config, dict):
                        print(f"Model config keys: {list(model_config.keys())}")
                        if 'labels' in model_config:
                            labels = model_config['labels']
                            if isinstance(labels, list):
                                print(f"  - Labels: {len(labels)} classes")
                                print(f"  - Sample labels: {labels[:10]}...")
                            else:
                                print(f"  - Labels type: {type(labels)}")
                        
                        if 'train_ds' in model_config and isinstance(model_config['train_ds'], dict):
                            if 'sample_rate' in model_config['train_ds']:
                                print(f"  - Sample rate: {model_config['train_ds']['sample_rate']}")
                    else:
                        print(f"  - Model config is not a dict: {type(model_config)}")
                else:
                    print("  - No 'model' key found in config")
                    print(f"  - Available keys: {list(self.config.keys())}")
            else:
                print("Config is not a dictionary!")
                    
        except Exception as e:
            print(f"✗ Error loading config: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_checkpoints(self):
        """Load encoder và decoder checkpoints"""
        try:
            # Load encoder checkpoint
            print(f"Loading encoder checkpoint: {self.encoder_path}")
            self.encoder_checkpoint = torch.load(self.encoder_path, map_location=self.device)
            print("✓ Encoder checkpoint loaded")
            
            # Load decoder checkpoint  
            print(f"Loading decoder checkpoint: {self.decoder_path}")
            self.decoder_checkpoint = torch.load(self.decoder_path, map_location=self.device)
            print("✓ Decoder checkpoint loaded")
            
            # In thông tin về checkpoints
            print(f"\nCheckpoint info:")
            print(f"  Encoder state dict keys: {len(self.encoder_checkpoint)}")
            print(f"  Decoder state dict keys: {len(self.decoder_checkpoint)}")
            
            # In một vài key đầu tiên để kiểm tra
            print(f"  Sample encoder keys: {list(self.encoder_checkpoint.keys())[:5]}")
            print(f"  Sample decoder keys: {list(self.decoder_checkpoint.keys())[:5]}")
            
        except Exception as e:
            print(f"✗ Error loading checkpoints: {str(e)}")
            raise
    
    def create_nemo_model(self):
        """Tạo NeMo model từ config và load weights với xử lý lỗi tốt hơn"""
        try:
            print("Creating NeMo model from config...")
            
            # Kiểm tra config structure
            if not isinstance(self.config, dict):
                raise ValueError(f"Config must be a dictionary, got {type(self.config)}")
            
            # Tìm model config
            model_config = None
            
            if 'model' in self.config:
                model_config = self.config['model']
                print("Found 'model' section in config")
            else:
                # Thử tìm config trực tiếp ở root level
                required_keys = ['encoder', 'decoder', 'preprocessor', 'labels']
                if any(key in self.config for key in required_keys):
                    model_config = self.config
                    print("Using root-level config as model config")
                else:
                    raise ValueError(f"No valid model config found. Available keys: {list(self.config.keys())}")
            
            if not isinstance(model_config, dict):
                raise ValueError(f"Model config must be a dictionary, got {type(model_config)}")
            
            print(f"Model config keys: {list(model_config.keys())}")
            
            # Convert sang OmegaConf
            nemo_config = OmegaConf.create(model_config)
            
            # Tạo model
            self.model = EncDecCTCModel(cfg=nemo_config)
            print("✓ NeMo model created")
            
            # Load encoder weights
            print("Loading encoder weights...")
            missing_keys, unexpected_keys = self.model.encoder.load_state_dict(
                self.encoder_checkpoint, strict=False
            )
            if missing_keys:
                print(f"  Missing encoder keys: {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    print(f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected encoder keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 10:
                    print(f"  Unexpected keys: {unexpected_keys}")
            print("✓ Encoder weights loaded")
            
            # Load decoder weights
            print("Loading decoder weights...")
            missing_keys, unexpected_keys = self.model.decoder.load_state_dict(
                self.decoder_checkpoint, strict=False
            )
            if missing_keys:
                print(f"  Missing decoder keys: {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    print(f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected decoder keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 10:
                    print(f"  Unexpected keys: {unexpected_keys}")
            print("✓ Decoder weights loaded")
            
            # Move to device và set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model ready on {self.device}")
            
        except Exception as e:
            print(f"✗ Error creating NeMo model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_model(self):
        """Load toàn bộ model (config + checkpoints + tạo NeMo model)"""
        print("="*50)
        print("LOADING QUARTZNET MODEL COMPONENTS")
        print("="*50)
        
        # Kiểm tra files
        if not self.check_files_exist():
            raise FileNotFoundError("Some required files are missing")
        
        # Load config
        self.load_config()
        
        # Load checkpoints
        self.load_checkpoints()
        
        # Tạo NeMo model
        self.create_nemo_model()
        
        print("="*50)
        print("✓ MODEL LOADING COMPLETED")
        print("="*50)
    
    def count_parameters(self, model):
        """Đếm số parameters của model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    def analyze_module_parameters(self, module, module_name="", max_depth=3, current_depth=0):
        """Phân tích parameters của từng module"""
        analysis = {}
        
        # Đếm parameters của module hiện tại
        module_params = self.count_parameters(module)
        analysis['parameters'] = module_params
        analysis['name'] = module_name if module_name else type(module).__name__
        analysis['type'] = type(module).__name__
        analysis['children'] = {}
        
        # Nếu chưa đạt độ sâu tối đa, phân tích các submodule
        if current_depth < max_depth and len(list(module.children())) > 0:
            for name, child in module.named_children():
                child_analysis = self.analyze_module_parameters(
                    child, 
                    name, 
                    max_depth, 
                    current_depth + 1
                )
                analysis['children'][name] = child_analysis
        
        return analysis
    
    def get_layer_details(self):
        """Lấy thông tin chi tiết về các layer"""
        if self.model is None:
            self.load_model()
        
        layer_details = {}
        
        # Phân tích preprocessor
        if hasattr(self.model, 'preprocessor'):
            preprocessor_analysis = self.analyze_module_parameters(self.model.preprocessor, "preprocessor")
            layer_details['preprocessor'] = preprocessor_analysis
        
        # Phân tích encoder
        if hasattr(self.model, 'encoder'):
            encoder_analysis = self.analyze_module_parameters(self.model.encoder, "encoder", max_depth=4)
            layer_details['encoder'] = encoder_analysis
        
        # Phân tích decoder
        if hasattr(self.model, 'decoder'):
            decoder_analysis = self.analyze_module_parameters(self.model.decoder, "decoder")
            layer_details['decoder'] = decoder_analysis
        
        return layer_details
    
    def print_layer_summary(self, analysis, indent=0, max_indent=3):
        """In tóm tắt về các layer"""
        prefix = "  " * indent
        
        total_params = analysis['parameters']['total']
        trainable_params = analysis['parameters']['trainable']
        
        print(f"{prefix}{analysis['name']} ({analysis['type']})")
        print(f"{prefix}  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        if indent < max_indent and analysis['children']:
            print(f"{prefix}  Submodules:")
            for child_name, child_analysis in analysis['children'].items():
                self.print_layer_summary(child_analysis, indent + 1, max_indent)
    
    def get_jasper_block_details(self):
        """Lấy thông tin chi tiết về các Jasper blocks"""
        if self.model is None:
            self.load_model()
        
        jasper_blocks = []
        
        if hasattr(self.model.encoder, 'encoder') and hasattr(self.model.encoder.encoder, '_modules'):
            for i, (block_name, block_module) in enumerate(self.model.encoder.encoder._modules.items()):
                block_info = {
                    'block_id': i,
                    'block_name': block_name,
                    'block_type': type(block_module).__name__,
                    'parameters': self.count_parameters(block_module)
                }
                
                # Phân tích chi tiết các layer trong block
                if hasattr(block_module, '_modules'):
                    layers = []
                    for layer_name, layer_module in block_module._modules.items():
                        if layer_module is not None:
                            layer_info = {
                                'name': layer_name,
                                'type': type(layer_module).__name__,
                                'parameters': self.count_parameters(layer_module)
                            }
                            
                            # Thêm thông tin shape nếu có
                            if hasattr(layer_module, 'weight') and layer_module.weight is not None:
                                layer_info['weight_shape'] = list(layer_module.weight.shape)
                            if hasattr(layer_module, 'bias') and layer_module.bias is not None:
                                layer_info['bias_shape'] = list(layer_module.bias.shape)
                            
                            layers.append(layer_info)
                    
                    block_info['layers'] = layers
                
                jasper_blocks.append(block_info)
        
        return jasper_blocks
    
    def analyze_config_structure(self):
        """Phân tích cấu trúc config file"""
        config_analysis = {}
        
        if self.config:
            def analyze_dict(d, path=""):
                result = {}
                if not isinstance(d, dict):
                    return {
                        'type': type(d).__name__,
                        'value': str(d)[:100] + "..." if len(str(d)) > 100 else str(d)
                    }
                
                for key, value in d.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, dict):
                        result[key] = {
                            'type': 'dict',
                            'keys_count': len(value),
                            'children': analyze_dict(value, current_path)
                        }
                    elif isinstance(value, list):
                        result[key] = {
                            'type': 'list',
                            'length': len(value),
                            'sample_items': value[:3] if len(value) > 0 else []
                        }
                    else:
                        result[key] = {
                            'type': type(value).__name__,
                            'value': str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        }
                return result
            
            config_analysis = analyze_dict(self.config)
        
        return config_analysis
    
    def check_language_model(self):
        """Kiểm tra language model nếu có"""
        lm_info = {}
        
        if self.lm_path and os.path.exists(self.lm_path):
            lm_info['exists'] = True
            lm_info['path'] = self.lm_path
            lm_info['size_mb'] = os.path.getsize(self.lm_path) / (1024 * 1024)
            
            # Thử đọc thông tin cơ bản về LM
            try:
                import kenlm
                lm_model = kenlm.Model(self.lm_path)
                lm_info['order'] = lm_model.order
                lm_info['type'] = 'KenLM'
                print(f"✓ Language Model: {self.lm_path}")
                print(f"  - Order: {lm_model.order}-gram")
                print(f"  - Size: {lm_info['size_mb']:.2f} MB")
            except ImportError:
                lm_info['type'] = 'Unknown (kenlm not available)'
                print(f"✓ Language Model file found: {self.lm_path}")
                print(f"  - Size: {lm_info['size_mb']:.2f} MB")
                print("  - Install kenlm for detailed analysis: pip install kenlm")
            except Exception as e:
                lm_info['error'] = str(e)
                print(f"⚠️  Language Model file found but cannot be analyzed: {e}")
        else:
            lm_info['exists'] = False
            if self.lm_path:
                print(f"✗ Language Model not found: {self.lm_path}")
            else:
                print("ℹ️  No language model path specified")
        
        return lm_info
    
    def analyze_checkpoints_only(self):
        """Phân tích chỉ checkpoint files khi không thể tạo NeMo model"""
        print("="*80)
        print("CHECKPOINT-ONLY ANALYSIS (No NeMo Model Creation)")
        print("="*80)
        
        # Load config và checkpoints
        if not self.check_files_exist():
            raise FileNotFoundError("Some required files are missing")
        
        self.load_config()
        self.load_checkpoints()
        
        # Phân tích config
        print(f"\n" + "="*50)
        print("CONFIG ANALYSIS")
        print("="*50)
        config_analysis = self.analyze_config_structure()
        
        # In một số thông tin config quan trọng
        if isinstance(self.config, dict):
            if 'model' in self.config:
                model_config = self.config['model']
                if isinstance(model_config, dict):
                    if 'labels' in model_config and isinstance(model_config['labels'], list):
                        print(f"  Number of output classes: {len(model_config['labels'])}")
                        print(f"  Sample labels: {model_config['labels'][:10]}...")
        
        # Phân tích encoder checkpoint
        print(f"\n" + "="*50)
        print("ENCODER CHECKPOINT ANALYSIS")
        print("="*50)
        
        encoder_param_count = 0
        encoder_shapes = {}
        
        for key, tensor in self.encoder_checkpoint.items():
            if isinstance(tensor, torch.Tensor):
                param_count = tensor.numel()
                encoder_param_count += param_count
                encoder_shapes[key] = {
                    'shape': list(tensor.shape),
                    'params': param_count,
                    'dtype': str(tensor.dtype)
                }
        
        print(f"Total encoder parameters: {encoder_param_count:,}")
        print(f"Number of encoder tensors: {len(encoder_shapes)}")
        
        # In một vài tensor shapes quan trọng
        print(f"\nSample encoder tensor shapes:")
        for i, (key, info) in enumerate(list(encoder_shapes.items())[:10]):
            print(f"  {key}: {info['shape']} ({info['params']:,} params)")
        
        # Phân tích decoder checkpoint
        print(f"\n" + "="*50)
        print("DECODER CHECKPOINT ANALYSIS")
        print("="*50)
        
        decoder_param_count = 0
        decoder_shapes = {}
        
        for key, tensor in self.decoder_checkpoint.items():
            if isinstance(tensor, torch.Tensor):
                param_count = tensor.numel()
                decoder_param_count += param_count
                decoder_shapes[key] = {
                    'shape': list(tensor.shape),
                    'params': param_count,
                    'dtype': str(tensor.dtype)
                }
        
        print(f"Total decoder parameters: {decoder_param_count:,}")
        print(f"Number of decoder tensors: {len(decoder_shapes)}")
        
        # In decoder tensor shapes
        print(f"\nDecoder tensor shapes:")
        for key, info in decoder_shapes.items():
            print(f"  {key}: {info['shape']} ({info['params']:,} params)")
        
        # Language Model
        lm_info = self.check_language_model()
        
        # Tổng hợp
        total_acoustic_params = encoder_param_count + decoder_param_count
        total_size_mb = total_acoustic_params * 4 / 1024 / 1024  # float32
        
        print(f"\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total acoustic model parameters: {total_acoustic_params:,}")
        print(f"  - Encoder: {encoder_param_count:,} ({encoder_param_count/total_acoustic_params*100:.1f}%)")
        print(f"  - Decoder: {decoder_param_count:,} ({decoder_param_count/total_acoustic_params*100:.1f}%)")
        print(f"Estimated model size: {total_size_mb:.2f} MB (float32)")
        
        if lm_info['exists']:
            print(f"Language model size: {lm_info['size_mb']:.2f} MB")
            print(f"Total system size: {total_size_mb + lm_info['size_mb']:.2f} MB")
        
        # Tạo summary report
        summary_report = {
            'analysis_type': 'checkpoint_only',
            'total_acoustic_parameters': total_acoustic_params,
            'encoder_parameters': encoder_param_count,
            'decoder_parameters': decoder_param_count,
            'model_size_mb': total_size_mb,
            'config_analysis': config_analysis,
            'language_model': lm_info,
            'encoder_tensors': encoder_shapes,
            'decoder_tensors': decoder_shapes
        }
        
        return summary_report
    
    def create_parameter_report(self):
        """Tạo báo cáo chi tiết về parameters"""
        try:
            if self.model is None:
                self.load_model()
            
            print("="*80)
            print("QUARTZNET MODEL PARAMETER ANALYSIS")
            print("="*80)
            
            # Tổng quan model
            total_params = self.count_parameters(self.model)
            print(f"\nModel Overview:")
            print(f"  Architecture: QuartzNet")
            print(f"  Total Parameters: {total_params['total']:,}")
            print(f"  Trainable Parameters: {total_params['trainable']:,}")
            print(f"  Non-trainable Parameters: {total_params['non_trainable']:,}")
            print(f"  Model Size: {total_params['total'] * 4 / 1024 / 1024:.2f} MB (float32)")
            
            # Thông tin về config
            print(f"\n" + "="*50)
            print("CONFIG ANALYSIS")
            print("="*50)
            
            config_analysis = self.analyze_config_structure()
            if 'model' in config_analysis:
                model_config = config_analysis['model']
                if 'children' in model_config:
                    for key, info in model_config['children'].items():
                        if key in ['labels', 'train_ds', 'validation_ds']:
                            print(f"  {key}: {info}")
            
            # Language Model
            lm_info = self.check_language_model()
            
            # Phân tích từng component
            print(f"\n" + "="*50)
            print("COMPONENT ANALYSIS")
            print("="*50)
            
            layer_details = self.get_layer_details()
            
            for component_name, component_analysis in layer_details.items():
                print(f"\n{component_name.upper()}:")
                self.print_layer_summary(component_analysis, indent=1, max_indent=3)
            
            # Phân tích Jasper blocks
            print(f"\n" + "="*50)
            print("JASPER BLOCKS ANALYSIS")
            print("="*50)
            
            jasper_blocks = self.get_jasper_block_details()
            
            for block in jasper_blocks:
                print(f"\nBlock {block['block_id']}: {block['block_name']} ({block['block_type']})")
                print(f"  Total parameters: {block['parameters']['total']:,}")
                
                if 'layers' in block:
                    print(f"  Layers:")
                    for layer in block['layers']:
                        layer_params = layer['parameters']['total']
                        print(f"    {layer['name']} ({layer['type']}): {layer_params:,} params")
                        if 'weight_shape' in layer:
                            print(f"      Weight shape: {layer['weight_shape']}")
                        if 'bias_shape' in layer:
                            print(f"      Bias shape: {layer['bias_shape']}")
            
            # Tạo DataFrame tóm tắt
            summary_data = []
            
            for component_name, component_analysis in layer_details.items():
                summary_data.append({
                    'Component': component_name,
                    'Type': component_analysis['type'],
                    'Total_Parameters': component_analysis['parameters']['total'],
                    'Trainable_Parameters': component_analysis['parameters']['trainable'],
                    'Percentage': (component_analysis['parameters']['total'] / total_params['total']) * 100
                })
            
            df = pd.DataFrame(summary_data)
            
            print(f"\n" + "="*50)
            print("PARAMETER SUMMARY TABLE")
            print("="*50)
            print(df.to_string(index=False, float_format='%.2f'))
            
            # Lưu chi tiết ra file
            detailed_report = {
                'analysis_type': 'full_model',
                'model_info': {
                    'architecture': 'QuartzNet',
                    'total_parameters': total_params,
                    'model_size_mb': total_params['total'] * 4 / 1024 / 1024,
                    'config_path': self.config_path,
                    'encoder_path': self.encoder_path,
                    'decoder_path': self.decoder_path,
                    'lm_path': self.lm_path
                },
                'config_analysis': config_analysis,
                'language_model': lm_info,
                'component_analysis': layer_details,
                'jasper_blocks': jasper_blocks,
                'parameter_summary': df.to_dict('records')
            }
            
            return detailed_report, df
            
        except Exception as e:
            print(f"⚠️  Failed to create full model analysis: {e}")
            print("Falling back to checkpoint-only analysis...")
            
            # Fallback to checkpoint-only analysis
            summary_report = self.analyze_checkpoints_only()
            
            # Tạo dummy DataFrame cho compatibility
            df = pd.DataFrame([
                {
                    'Component': 'encoder',
                    'Type': 'JasperEncoder',
                    'Total_Parameters': summary_report['encoder_parameters'],
                    'Trainable_Parameters': summary_report['encoder_parameters'],
                    'Percentage': summary_report['encoder_parameters'] / summary_report['total_acoustic_parameters'] * 100
                },
                {
                    'Component': 'decoder',
                    'Type': 'JasperDecoder',
                    'Total_Parameters': summary_report['decoder_parameters'],
                    'Trainable_Parameters': summary_report['decoder_parameters'],
                    'Percentage': summary_report['decoder_parameters'] / summary_report['total_acoustic_parameters'] * 100
                }
            ])
            
            return summary_report, df
    
    def save_analysis_results(self, detailed_report, df, output_dir="model_analysis"):
        """Lưu kết quả phân tích ra file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu detailed report
        with open(f"{output_dir}/detailed_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False, default=str)
        print(f"Detailed analysis saved to: {output_dir}/detailed_analysis.json")
        
        # Lưu summary table
        df.to_csv(f"{output_dir}/parameter_summary.csv", index=False)
        print(f"Parameter summary saved to: {output_dir}/parameter_summary.csv")
        
        # Tạo markdown report
        with open(f"{output_dir}/analysis_report.md", 'w', encoding='utf-8') as f:
            f.write("# QuartzNet Model Analysis Report\n\n")
            f.write(f"## Model Overview\n")
            
            if detailed_report['analysis_type'] == 'full_model':
                f.write(f"- **Architecture**: {detailed_report['model_info']['architecture']}\n")
                f.write(f"- **Total Parameters**: {detailed_report['model_info']['total_parameters']['total']:,}\n")
                f.write(f"- **Trainable Parameters**: {detailed_report['model_info']['total_parameters']['trainable']:,}\n")
                f.write(f"- **Model Size**: {detailed_report['model_info']['model_size_mb']:.2f} MB\n")
                f.write(f"- **Config Path**: {detailed_report['model_info']['config_path']}\n")
                f.write(f"- **Encoder Path**: {detailed_report['model_info']['encoder_path']}\n")
                f.write(f"- **Decoder Path**: {detailed_report['model_info']['decoder_path']}\n")
            else:
                f.write(f"- **Analysis Type**: Checkpoint-only (NeMo model creation failed)\n")
                f.write(f"- **Total Acoustic Parameters**: {detailed_report['total_acoustic_parameters']:,}\n")
                f.write(f"- **Encoder Parameters**: {detailed_report['encoder_parameters']:,}\n")
                f.write(f"- **Decoder Parameters**: {detailed_report['decoder_parameters']:,}\n")
                f.write(f"- **Estimated Model Size**: {detailed_report['model_size_mb']:.2f} MB\n")
            
            if detailed_report['language_model']['exists']:
                f.write(f"- **Language Model**: {detailed_report['language_model']['path']} ({detailed_report['language_model']['size_mb']:.2f} MB)\n")
            
            f.write("\n## Parameter Distribution\n\n")
            f.write(df.to_markdown(index=False, floatfmt='.2f'))
            f.write("\n\n")
            
            if detailed_report['analysis_type'] == 'full_model' and 'jasper_blocks' in detailed_report:
                f.write("## Jasper Blocks\n\n")
                for block in detailed_report['jasper_blocks']:
                    f.write(f"### Block {block['block_id']}: {block['block_name']}\n")
                    f.write(f"- **Type**: {block['block_type']}\n")
                    f.write(f"- **Parameters**: {block['parameters']['total']:,}\n")
                    if 'layers' in block:
                        f.write("- **Layers**:\n")
                        for layer in block['layers']:
                            f.write(f"  - {layer['name']} ({layer['type']}): {layer['parameters']['total']:,} params\n")
                    f.write("\n")
        
        print(f"Markdown report saved to: {output_dir}/analysis_report.md")

# Script chính
if __name__ == "__main__":
    # Đường dẫn các file
    config_path = 'configs/quartznet12x1_vi.yaml'
    encoder_path = 'models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt'
    decoder_path = 'models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt'
    lm_path = 'models/language_model/3-gram-lm.binary'
    
    # Tạo analyzer
    print("Initializing QuartzNet Model Analyzer...")
    analyzer = QuartzNetModelAnalyzer(
        config_path=config_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        lm_path=lm_path
    )
    
    # Thực hiện phân tích
    print("\nStarting model analysis...")
    detailed_report, summary_df = analyzer.create_parameter_report()
    
    # Lưu kết quả
    print("\nSaving analysis results...")
    analyzer.save_analysis_results(detailed_report, summary_df)
    
    print(f"\n{'='*80}")
    print("MODEL ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    
    if detailed_report['analysis_type'] == 'full_model':
        print(f"Total Parameters: {detailed_report['model_info']['total_parameters']['total']:,}")
        print(f"Model Size: {detailed_report['model_info']['model_size_mb']:.2f} MB")
    else:
        print(f"Total Acoustic Parameters: {detailed_report['total_acoustic_parameters']:,}")
        print(f"Model Size: {detailed_report['model_size_mb']:.2f} MB")
    
    print(f"Results saved in 'model_analysis/' directory")
    print(f"{'='*80}")
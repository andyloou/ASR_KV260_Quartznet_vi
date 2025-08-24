# QuartzNet Vietnamese: load weights, run inference, CTC decode (simple argmax collapse), and export to ONNX

import glob
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio

class QuartzNetVietnamese(nn.Module):
    def __init__(self, weights_dir=None, config_path=None):
        super(QuartzNetVietnamese, self).__init__()
        self._vars = {}
        
        # Load config if provided and determine vocab size
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                if 'labels' in self.config:
                    self.labels = self.config['labels']
                    print(f"Config labels ({len(self.labels)}): {self.labels}")
                else:
                    self.labels = [' ', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 'ạ', 'ả', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ', 'f', 'j', 'w', 'z']
                    print(f"Using full Vietnamese labels ({len(self.labels)})")
                config_vocab_size = len(self.labels)
                print(f"Config vocab size: {config_vocab_size}")
        else:
            self.config = None
            self.labels = [' ', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 'ạ', 'ả', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ', 'f', 'j', 'w', 'z']
            config_vocab_size = len(self.labels)
        
        # Mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.labels)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.labels)}
        self.blank_id = 0
        
        # Determine actual vocab size from checkpoint data
        if weights_dir:
            decoder_weight_files = glob.glob(os.path.join(weights_dir, "decoder_layers_0_weight.npy"))
            if decoder_weight_files:
                sample_weight = np.load(decoder_weight_files[0])
                checkpoint_vocab_size = sample_weight.shape[0]
                print(f"Checkpoint vocab size: {checkpoint_vocab_size}")
                self.vocab_size = checkpoint_vocab_size
            else:
                self.vocab_size = config_vocab_size if config_vocab_size else 95
        else:
            self.vocab_size = config_vocab_size if config_vocab_size else 95
        
        print(f"Final vocab size: {self.vocab_size}")
        
        # Load weights from .npy files
        if weights_dir:
            for b in glob.glob(os.path.join(weights_dir, "*.npy")):
                v = torch.from_numpy(np.load(b))
                self._vars[os.path.basename(b)[:-4]] = v
        
        self._build_layers()
    
    def preprocess_audio(self, audio_path, sample_rate=16000):
        print(f"Processing audio: {audio_path}")
        
        if isinstance(audio_path, str):
            waveform, orig_sr = torchaudio.load(audio_path)
        else:
            waveform = audio_path
            orig_sr = sample_rate
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)
        
        n_fft = 512
        win_length = int(sample_rate * 0.02)
        hop_length = int(sample_rate * 0.01)
        n_mels = 64
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            normalized=False
        )
        
        mel = mel_spectrogram(waveform)
        mel = torch.log(mel + 1e-6)
        mel = (mel - mel.mean(dim=2, keepdim=True)) / (mel.std(dim=2, keepdim=True) + 1e-6)
        
        print(f"Audio preprocessed: {waveform.shape} -> {mel.shape}")
        return mel
        
    def ctc_decode(self, logits, blank_id=0):
        """
        Greedy decode like code 1: argmax over time, collapse repeats, DO NOT drop id=0.
        This assumes labels[0] == ' ' (space) and there is no explicit blank class.
        """
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)  # [B, T]

        decoded_sequences = []
        for b in range(preds.shape[0]):
            out = []
            prev = None
            for p in preds[b].tolist():
                if p != prev:
                    if 0 <= p < len(self.labels):
                        out.append(self.labels[p])  # keep index 0 => space
                prev = p
            text = ''.join(out)

            # Optional light cleanup (giống tác dụng hiển thị của code 1):
            # - rút gọn nhiều khoảng trắng liên tiếp
            # - strip đầu/cuối
            text = ' '.join(text.split())

            decoded_sequences.append(text)
        return decoded_sequences
    
    def predict(self, audio_path):
        mel = self.preprocess_audio(audio_path)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(mel)
        texts = self.ctc_decode(logits, blank_id=self.blank_id)
        return texts[0] if texts else ""
    
    def _build_layers(self):
        """Build layers with selective residual connections to match parameter count"""
        self.encoder = nn.ModuleDict()
        self.encoder['0'] = self._create_block_0()
        residual_blocks = self._determine_residual_blocks()
        
        for i in range(1, 7):
            kernel_size = 33 if i in [1, 2, 3] else 39
            has_residual = i in residual_blocks
            self.encoder[str(i)] = self._create_regular_block(
                in_channels=256, out_channels=256, kernel_size=kernel_size, has_residual=has_residual
            )
        
        has_residual = 7 in residual_blocks
        self.encoder['7'] = self._create_regular_block(
            in_channels=256, out_channels=512, kernel_size=51, has_residual=has_residual
        )
        
        for i in range(8, 14):
            if i in [8, 9]:
                kernel_size = 51
            elif i in [10, 11, 12]:
                kernel_size = 63
            else:
                kernel_size = 75
            has_residual = i in residual_blocks
            self.encoder[str(i)] = self._create_regular_block(
                in_channels=512, out_channels=512, kernel_size=kernel_size, has_residual=has_residual
            )
        
        self.encoder['14'] = self._create_block_14()
        
        self.decoder = nn.ModuleDict()
        self.decoder['decoder_layers'] = nn.ModuleDict()
        self.decoder['decoder_layers']['0'] = nn.Conv1d(1024, self.vocab_size, kernel_size=1, bias=True)
        
        if self._vars:
            self._assign_weights()
        self._print_parameter_summary()
    
    def _create_block_0(self):
        block = nn.ModuleDict()
        block['mconv'] = nn.ModuleDict()
        block['mconv']['0'] = nn.ModuleDict()
        block['mconv']['0']['conv'] = nn.Conv1d(64, 64, kernel_size=33, stride=2, padding=16, groups=64, bias=False)
        block['mconv']['1'] = nn.ModuleDict()
        block['mconv']['1']['conv'] = nn.Conv1d(64, 256, kernel_size=1, bias=False)
        block['mconv']['2'] = nn.BatchNorm1d(256)
        return block
    
    def _determine_residual_blocks(self):
        base_encoder_params = self._calculate_base_encoder_params()
        expected_encoder = 5015872
        available_for_residuals = expected_encoder - base_encoder_params
        
        print(f"Base encoder parameters: {base_encoder_params:,}")
        print(f"Available for residuals: {available_for_residuals:,}")
        
        residual_blocks = set()
        remaining_budget = available_for_residuals
        priority_blocks = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        
        for block_idx in priority_blocks:
            if block_idx <= 6:
                cost = 256 * 256 + 256 * 2
            elif block_idx == 7:
                cost = 256 * 512 + 512 * 2
            else:
                cost = 512 * 512 + 512 * 2
            
            res_weight_key = f'encoder_{block_idx}_res_0_0_conv_weight'
            if res_weight_key in self._vars and cost <= remaining_budget:
                residual_blocks.add(block_idx)
                remaining_budget -= cost
                print(f"Enabling residual for block {block_idx} (cost: {cost:,})")
        
        print(f"Residual blocks: {sorted(residual_blocks)}")
        print(f"Remaining budget: {remaining_budget:,}")
        return residual_blocks
    
    def _calculate_base_encoder_params(self):
        params = 0
        params += 64*1*33 + 256*64*1 + 256*2
        
        for i in range(1, 7):
            kernel_size = 33 if i in [1, 2, 3] else 39
            params += 256*1*kernel_size + 256*256*1 + 256*2
        
        params += 256*1*51 + 512*256*1 + 512*2
        
        for i in range(8, 14):
            kernel_size = 51 if i in [8, 9] else (63 if i in [10, 11, 12] else 75)
            params += 512*1*kernel_size + 512*512*1 + 512*2
        
        params += 1024*512*1 + 1024*2
        return params
    
    def _create_regular_block(self, in_channels, out_channels, kernel_size, has_residual=True):
        block = nn.ModuleDict()
        block['mconv'] = nn.ModuleDict()
        padding = kernel_size // 2
        
        block['mconv']['0'] = nn.ModuleDict()
        block['mconv']['0']['conv'] = nn.Conv1d(
            in_channels, in_channels, kernel_size, stride=1, padding=padding, groups=in_channels, bias=False
        )
        block['mconv']['1'] = nn.ModuleDict()
        block['mconv']['1']['conv'] = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        block['mconv']['2'] = nn.BatchNorm1d(out_channels)
        
        if has_residual:
            block['res'] = nn.ModuleDict()
            block['res']['0'] = nn.ModuleDict()
            block['res']['0']['0'] = nn.ModuleDict()
            block['res']['0']['0']['conv'] = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            block['res']['0']['1'] = nn.BatchNorm1d(out_channels)
        return block
    
    def _create_block_14(self):
        block = nn.ModuleDict()
        block['mconv'] = nn.ModuleDict()
        block['mconv']['0'] = nn.ModuleDict()
        block['mconv']['0']['conv'] = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        block['mconv']['1'] = nn.BatchNorm1d(1024)
        return block
    
    def _assign_weights(self):
        print("\n=== WEIGHT ASSIGNMENT ===")
        for block_idx in range(15):
            self._assign_block_weights(block_idx)
        self._assign_decoder_weights()
    
    def _assign_block_weights(self, block_idx):
        if str(block_idx) not in self.encoder:
            return
        block = self.encoder[str(block_idx)]
        mconv = block['mconv'] if 'mconv' in block else {}
        
        if block_idx == 0:
            self._assign_conv_weight(mconv, '0', f'encoder_{block_idx}_mconv_0_conv_weight')
            self._assign_conv_weight(mconv, '1', f'encoder_{block_idx}_mconv_1_conv_weight')
            self._assign_bn_weights(mconv, '2', f'encoder_{block_idx}_mconv_2')
        elif block_idx == 14:
            self._assign_conv_weight(mconv, '0', f'encoder_{block_idx}_mconv_0_conv_weight')
            self._assign_bn_weights(mconv, '1', f'encoder_{block_idx}_mconv_1')
        else:
            self._assign_conv_weight(mconv, '0', f'encoder_{block_idx}_mconv_0_conv_weight')
            self._assign_conv_weight(mconv, '1', f'encoder_{block_idx}_mconv_1_conv_weight')
            self._assign_bn_weights(mconv, '2', f'encoder_{block_idx}_mconv_2')
            if 'res' in block:
                res_path = block['res']['0']
                self._assign_conv_weight(res_path, '0', f'encoder_{block_idx}_res_0_0_conv_weight')
                self._assign_bn_weights(res_path, '1', f'encoder_{block_idx}_res_0_1')
    
    def _assign_conv_weight(self, parent_dict, key, weight_key):
        if key in parent_dict and 'conv' in parent_dict[key]:
            if weight_key in self._vars:
                conv_layer = parent_dict[key]['conv']
                weight_tensor = self._vars[weight_key]
                if conv_layer.weight.shape == weight_tensor.shape:
                    conv_layer.weight.data = weight_tensor
                    print(f"Loaded {weight_key}: {weight_tensor.shape}")
                else:
                    print(f"Shape mismatch {weight_key}: {conv_layer.weight.shape} vs {weight_tensor.shape}")
            else:
                print(f"Missing weight: {weight_key}")
    
    def _assign_bn_weights(self, parent_dict, key, bn_prefix):
        if key in parent_dict and isinstance(parent_dict[key], nn.BatchNorm1d):
            self._assign_bn_weights_direct(parent_dict[key], bn_prefix)
    
    def _assign_bn_weights_direct(self, bn_layer, bn_prefix):
        if bn_layer is None:
            return
        for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
            param_key = f'{bn_prefix}_{param_name}'
            if param_key in self._vars:
                param_tensor = self._vars[param_key]
                if param_name in ['weight', 'bias']:
                    getattr(bn_layer, param_name).data = param_tensor
                elif param_name in ['running_mean', 'running_var']:
                    setattr(bn_layer, param_name, param_tensor)
                print(f"Loaded BN {param_key}")
    
    def _assign_decoder_weights(self):
        decoder_layer = self.decoder['decoder_layers']['0']
        weight_key = 'decoder_layers_0_weight'
        bias_key = 'decoder_layers_0_bias'
        
        if weight_key in self._vars:
            weight_tensor = self._vars[weight_key]
            target_shape = decoder_layer.weight.shape
            if weight_tensor.shape == target_shape:
                decoder_layer.weight.data = weight_tensor
                print(f"Loaded decoder weight: {weight_tensor.shape}")
            else:
                print(f"Decoder weight shape mismatch: {target_shape} vs {weight_tensor.shape}")
                min_vocab = min(weight_tensor.shape[0], target_shape[0])
                decoder_layer.weight.data[:min_vocab] = weight_tensor[:min_vocab]
        
        if bias_key in self._vars:
            bias_tensor = self._vars[bias_key]
            target_shape = decoder_layer.bias.shape
            if bias_tensor.shape == target_shape:
                decoder_layer.bias.data = bias_tensor
                print(f"Loaded decoder bias: {bias_tensor.shape}")
            else:
                print(f"Decoder bias shape mismatch: {target_shape} vs {bias_tensor.shape}")
                min_vocab = min(bias_tensor.shape[0], target_shape[0])
                decoder_layer.bias.data[:min_vocab] = bias_tensor[:min_vocab]
    
    def _print_parameter_summary(self):
        print("\n=== PARAMETER SUMMARY ===")
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for name, p in self.named_parameters() if 'encoder' in name)
        decoder_params = sum(p.numel() for name, p in self.named_parameters() if 'decoder' in name)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Encoder parameters: {encoder_params:,}")
        print(f"Decoder parameters: {decoder_params:,}")
        print(f"Expected total: 5,132,215")
        
        diff = total_params - 5132215
        if abs(diff) < 1000:
            print("Parameter count close to expected.")
        else:
            print(f"Parameter difference: {diff:,}")
        
        print("\n--- Per-Block Parameters ---")
        for block_idx in range(15):
            if str(block_idx) in self.encoder:
                block = self.encoder[str(block_idx)]
                block_params = sum(p.numel() for p in block.parameters())
                has_res = 'res' in block
                print(f"Block {block_idx}: {block_params:,} ({'with' if has_res else 'no'} residual)")
    
    def forward(self, mel_features):
        x = mel_features
        for block_idx in range(15):
            if str(block_idx) not in self.encoder:
                continue
            block = self.encoder[str(block_idx)]
            x = self._forward_block(x, block, block_idx)
        decoder_layer = self.decoder['decoder_layers']['0']
        x = decoder_layer(x)
        return x.permute(0, 2, 1)
    
    def _forward_block(self, x, block, block_idx):
        residual = x
        mconv = block['mconv'] if 'mconv' in block else {}
        
        if block_idx == 0:
            if '0' in mconv and 'conv' in mconv['0']:
                x = mconv['0']['conv'](x)
            if '1' in mconv and 'conv' in mconv['1']:
                x = mconv['1']['conv'](x)
            if '2' in mconv:
                x = mconv['2'](x)
                x = F.relu(x)
        elif block_idx == 14:
            if '0' in mconv and 'conv' in mconv['0']:
                x = mconv['0']['conv'](x)
            if '1' in mconv:
                x = mconv['1'](x)
                x = F.relu(x)
        else:
            if '0' in mconv and 'conv' in mconv['0']:
                x = mconv['0']['conv'](x)
            if '1' in mconv and 'conv' in mconv['1']:
                x = mconv['1']['conv'](x)
            if '2' in mconv:
                x = mconv['2'](x)
            if 'res' in block:
                res_path = block['res']['0']
                if '0' in res_path and 'conv' in res_path['0']:
                    residual = res_path['0']['conv'](residual)
                if '1' in res_path:
                    residual = res_path['1'](residual)
                x = x + residual
            x = F.relu(x)
        return x


def extract_and_save_weights(encoder_checkpoint, decoder_checkpoint, output_dir):
    """Extract weights from checkpoints and save as .npy files"""
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading encoder from: {encoder_checkpoint}")
    encoder_state = torch.load(encoder_checkpoint, map_location='cpu')
    
    print(f"Loading decoder from: {decoder_checkpoint}")
    decoder_state = torch.load(decoder_checkpoint, map_location='cpu')
    
    encoder_weights = encoder_state['state_dict'] if 'state_dict' in encoder_state else encoder_state
    decoder_weights = decoder_state['state_dict'] if 'state_dict' in decoder_state else decoder_state
    
    print(f"Found {len(encoder_weights)} encoder weights")
    print(f"Found {len(decoder_weights)} decoder weights")
    
    all_weights = {}
    all_weights.update(encoder_weights)
    all_weights.update(decoder_weights)
    
    for key, tensor in all_weights.items():
        clean_key = key.replace('.', '_')
        if clean_key.startswith('module_'):
            clean_key = clean_key[7:]
        if clean_key.startswith('decoder_decoder_'):
            clean_key = clean_key.replace('decoder_decoder_', 'decoder_')
        npy_path = os.path.join(output_dir, f"{clean_key}.npy")
        np.save(npy_path, tensor.cpu().numpy())
        print(f"Saved: {clean_key}.npy - Shape: {tensor.shape}")
    
    print(f"\nTotal weights saved: {len(all_weights)}")
    return all_weights.keys()


def test_model_with_proper_preprocessing():
    """Test model with proper mel spectrogram preprocessing"""
    config_path = 'configs/quartznet12x1_vi.yaml'
    weights_output_dir = 'variables/v_quartznet_vi'
    test_audio_dir = '/home/andyloou/viet-asr/audio_samples'
    ground_truth_file = 'ground_truth.txt'
    
    print("=" * 60)
    print("TESTING QuartzNet Vietnamese with PROPER PREPROCESSING")
    print("=" * 60)
    
    print("\n1. Loading model...")
    try:
        model = QuartzNetVietnamese(
            weights_dir=weights_output_dir,
            config_path=config_path
        )
        model.eval()
        print("Model loaded successfully")
        print(f"Labels: {model.labels}")
        print(f"Vocab size: {model.vocab_size}")
        print(f"Blank ID: {model.blank_id}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n2. Loading ground truth...")
    ground_truth = {}
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('|')
                    if len(parts) == 2:
                        filename, text = parts
                        ground_truth[filename] = text
                    else:
                        print(f"Skipping invalid line: {line}")
        print(f"Loaded {len(ground_truth)} ground truth entries")
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return
    
    print("\n3. Testing audio files...")
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(glob.glob(os.path.join(test_audio_dir, ext)))
    
    results = []
    total_time = 0.0
    
    for audio_file in audio_files[:5]:
        filename = os.path.basename(audio_file)
        print(f"\nProcessing: {filename}")
        try:
            gt_text = ground_truth.get(filename, "No ground truth")
            import time
            start = time.time()
            predicted_text = model.predict(audio_file)
            end = time.time()
            inference_time = end - start
            total_time += inference_time
            
            print(f"  Ground Truth: {gt_text}")
            print(f"  Prediction:   {predicted_text}")
            print(f"  Inference Time: {inference_time:.3f}s")
            
            results.append({
                'filename': filename,
                'ground_truth': gt_text,
                'prediction': predicted_text,
                'inference_time': inference_time
            })
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if results:
        avg_time = total_time / len(results)
        print(f"Total files processed: {len(results)}")
        print(f"Average inference time: {avg_time:.3f}s")
        print(f"Total inference time: {total_time:.3f}s")
        
        exact_matches = sum(1 for r in results if r['prediction'].strip().lower() == r['ground_truth'].strip().lower())
        accuracy = exact_matches / len(results) * 100
        print(f"Exact match accuracy: {accuracy:.1f}%")
        
        print("\n--- Sample Results ---")
        for i, result in enumerate(results[:3]):
            print(f"{i+1}. {result['filename']}")
            print(f"   GT: {result['ground_truth']}")
            print(f"   Pred: {result['prediction']}")
            print(f"   Time: {result['inference_time']:.3f}s")
    else:
        print("No results to analyze")


def export_quartznet_vietnamese_fixed():
    """Main export function with proper mel spectrogram input"""
    config_path = 'configs/quartznet12x1_vi.yaml'
    encoder_checkpoint = 'models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt'
    decoder_checkpoint = 'models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt'
    weights_output_dir = 'variables/v_quartznet_vi'
    
    print("=" * 60)
    print("QuartzNet Vietnamese Export (FIXED VERSION)")
    print("=" * 60)
    
    print("\n1. Extracting weights from checkpoints...")
    try:
        _ = extract_and_save_weights(
            encoder_checkpoint, decoder_checkpoint, weights_output_dir
        )
        print(f"Successfully extracted weights to {weights_output_dir}")
    except Exception as e:
        print(f"Error extracting weights: {e}")
        return
    
    print("\n2. Creating model...")
    try:
        model = QuartzNetVietnamese(
            weights_dir=weights_output_dir,
            config_path=config_path
        )
        model.eval()
        print("Model created successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n3. Testing model with mel input...")
    try:
        dummy_mel = torch.randn(1, 64, 100)
        with torch.no_grad():
            output = model(dummy_mel)
        print(f"Model test passed")
        print(f"Input shape: {dummy_mel.shape}")
        print(f"Output shape: {output.shape}")
        
        decoded = model.ctc_decode(output)
        print(f"CTC decode test passed")
        print(f"Decoded text: '{decoded[0]}'")
    except Exception as e:
        print(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4. Exporting to ONNX...")
    try:
        output_path = "quartznet_vietnamese_mel_fixed.onnx"
        import onnx
        torch.onnx.export(
            model,
            dummy_mel,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['mel_features'],
            output_names=['logits'],
            dynamic_axes={'mel_features': {2: 'time_steps'}, 'logits': {1: 'time_steps'}},
            verbose=False
        )
        print(f"ONNX export successful: {output_path}")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Export completed successfully!")
    print(f"Model file: quartznet_vietnamese_mel_fixed.onnx")
    print(f"Input: mel_features [batch, 64, time_steps]")
    print(f"Output: logits [batch, time_steps, {model.vocab_size}]")
    print(f"Labels: {model.labels}")
    print("=" * 60)
    
    print("\n5. Testing with real audio...")
    test_model_with_proper_preprocessing()


if __name__ == "__main__":
    export_quartznet_vietnamese_fixed()

import torch
import yaml
from collections import defaultdict

def debug_weight_mapping():
    """Enhanced debug script to understand weight structure and create correct mapping"""
    
    encoder_path = 'models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt'
    decoder_path = 'models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt'
    config_path = 'configs/quartznet12x1_vi.yaml'
    
    # Load config để hiểu architecture
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    jasper_config = config['JasperEncoder']['jasper']
    num_labels = len(config['labels'])
    
    print("=== ARCHITECTURE FROM CONFIG ===")
    print(f"Number of encoder blocks: {len(jasper_config)}")
    print(f"Number of labels: {num_labels}")
    print("\nEncoder blocks:")
    for i, block in enumerate(jasper_config):
        residual = "✓" if block.get('residual', False) else "✗"
        separable = "✓" if block.get('separable', False) else "✗"
        repeat = block.get('repeat', 1)
        print(f"  Block {i:2d}: {block['filters']:4d} filters, kernel={block['kernel']}, "
              f"repeat={repeat}, residual={residual}, separable={separable}")
    
    # Load checkpoints
    print("\n=== DETAILED WEIGHT ANALYSIS ===")
    try:
        encoder_state = torch.load(encoder_path, map_location='cpu')
        encoder_weights = encoder_state['state_dict'] if 'state_dict' in encoder_state else encoder_state
        
        decoder_state = torch.load(decoder_path, map_location='cpu')
        decoder_weights = decoder_state['state_dict'] if 'state_dict' in decoder_state else decoder_state
        
        # Analyze encoder structure
        print("ENCODER STRUCTURE:")
        encoder_blocks = defaultdict(list)
        total_encoder_params = 0
        
        for key, tensor in encoder_weights.items():
            total_encoder_params += tensor.numel()
            
            # Parse key to understand structure
            parts = key.split('.')
            if len(parts) >= 2:
                block_num = parts[1] if parts[1].isdigit() else 'unknown'
                encoder_blocks[block_num].append({
                    'key': key,
                    'shape': tensor.shape,
                    'params': tensor.numel(),
                    'parts': parts
                })
        
        # Print detailed block structure
        for block_num in sorted(encoder_blocks.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            block_weights = encoder_blocks[block_num]
            block_total_params = sum(w['params'] for w in block_weights)
            
            print(f"\n--- Block {block_num} ({block_total_params:,} params) ---")
            
            # Group by layer type
            layer_groups = defaultdict(list)
            for weight in block_weights:
                layer_key = '.'.join(weight['parts'][:4]) if len(weight['parts']) >= 4 else weight['key']
                layer_groups[layer_key].append(weight)
            
            for layer_name, layer_weights in layer_groups.items():
                print(f"  {layer_name}:")
                for w in layer_weights:
                    print(f"    {w['key']}: {w['shape']} ({w['params']:,} params)")
        
        # Analyze decoder structure
        print("\n--- DECODER STRUCTURE ---")
        total_decoder_params = 0
        for key, tensor in decoder_weights.items():
            total_decoder_params += tensor.numel()
            print(f"  {key}: {tensor.shape} ({tensor.numel():,} params)")
        
        print(f"\n=== PARAMETER SUMMARY ===")
        print(f"Total encoder parameters: {total_encoder_params:,}")
        print(f"Total decoder parameters: {total_decoder_params:,}")
        print(f"Grand total: {total_encoder_params + total_decoder_params:,}")
        
        # Generate weight mapping suggestions
        print(f"\n=== WEIGHT MAPPING ANALYSIS ===")
        print("Based on the structure, here's what we can infer:")
        
        # Look for patterns
        separable_blocks = []
        regular_blocks = []
        
        for block_num in sorted(encoder_blocks.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            if not block_num.isdigit():
                continue
                
            block_weights = encoder_blocks[block_num]
            
            # Count different types of weights
            conv_weights = [w for w in block_weights if 'conv.weight' in w['key']]
            bn_weights = [w for w in block_weights if any(x in w['key'] for x in ['weight', 'bias', 'running_mean', 'running_var']) and 'conv' not in w['key']]
            
            print(f"\nBlock {block_num}: {len(conv_weights)} conv layers, {len(bn_weights)} BN params")
            
            # Analyze convolution structure
            mconv_weights = [w for w in conv_weights if 'mconv' in w['key']]
            res_weights = [w for w in conv_weights if 'res' in w['key']]
            
            print(f"  Main conv (mconv): {len(mconv_weights)} layers")
            print(f"  Residual (res): {len(res_weights)} layers")
            
            for w in mconv_weights:
                conv_shape = w['shape']
                is_depthwise = len(conv_shape) == 3 and conv_shape[0] == conv_shape[1]
                is_pointwise = len(conv_shape) == 3 and conv_shape[2] == 1
                conv_type = "depthwise" if is_depthwise else "pointwise" if is_pointwise else "regular"
                print(f"    {w['key']}: {conv_shape} ({conv_type})")
        
        # Create corrected model template
        print(f"\n=== MODEL STRUCTURE RECOMMENDATIONS ===")
        print("Your model should have these key characteristics:")
        print(f"1. Total parameters should be ~{total_encoder_params + total_decoder_params:,}")
        print(f"2. Decoder output size should be {num_labels}")
        print("3. Use separable convolutions (depthwise + pointwise) for most blocks")
        print("4. Match the exact layer structure seen in the weights")
        
        return encoder_weights, decoder_weights, config
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def analyze_separable_conv_structure():
    """Analyze how separable convolutions should be implemented"""
    
    print("\n=== SEPARABLE CONVOLUTION ANALYSIS ===")
    print("QuartzNet uses 1D Time-Channel Separable Convolutions:")
    print("1. Depthwise Conv1D: operates along time dimension for each channel separately")
    print("2. Pointwise Conv1D: 1x1 conv that mixes channels")
    print("3. BatchNorm after the pointwise convolution")
    print("4. ReLU activation")
    print("5. Optional residual connection")
    
    # Example of correct separable convolution
    print("\nCorrect PyTorch Implementation:")
    print("""
    # Depthwise convolution (groups=in_channels)
    depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                         padding=padding, groups=in_channels, bias=False)
    
    # Pointwise convolution (1x1)
    pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
    
    # BatchNorm after pointwise
    bn = nn.BatchNorm1d(out_channels)
    
    # Forward pass:
    x = depthwise(x)
    x = pointwise(x)  
    x = bn(x)
    x = F.relu(x)
    """)

def create_minimal_test_model():
    """Create a minimal model to test parameter matching"""
    
    print("\n=== CREATING MINIMAL TEST MODEL ===")
    
    import torch.nn as nn
    
    class MinimalQuartzNet(nn.Module):
        def __init__(self, vocab_size=91):
            super().__init__()
            
            # Try to match the first few blocks exactly
            # Block 0: Based on typical QuartzNet first block
            self.block0_conv1 = nn.Conv1d(1, 64, 33, stride=2, padding=16, bias=False)
            self.block0_conv2 = nn.Conv1d(64, 256, 1, bias=False) 
            self.block0_bn = nn.BatchNorm1d(256)
            
            # Block 1: Separable conv with residual
            self.block1_dw = nn.Conv1d(256, 256, 33, padding=16, groups=256, bias=False)
            self.block1_pw = nn.Conv1d(256, 256, 1, bias=False)
            self.block1_bn = nn.BatchNorm1d(256)
            self.block1_res = nn.Conv1d(256, 256, 1, bias=False)
            self.block1_res_bn = nn.BatchNorm1d(256)
            
            # Simple decoder for testing
            self.decoder = nn.Conv1d(256, vocab_size, 1, bias=True)
        
        def forward(self, x):
            # Block 0
            x = self.block0_conv1(x)
            x = self.block0_conv2(x)
            x = self.block0_bn(x)
            x = torch.relu(x)
            
            # Block 1 with residual
            residual = self.block1_res(x)
            residual = self.block1_res_bn(residual)
            
            x = self.block1_dw(x)
            x = self.block1_pw(x)
            x = self.block1_bn(x)
            x = x + residual
            x = torch.relu(x)
            
            # Decoder
            x = self.decoder(x)
            return x.permute(0, 2, 1)
    
    # Test parameter count
    model = MinimalQuartzNet()
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Minimal model parameters: {total_params:,}")
    print("\nParameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} = {param.numel():,}")
    
    return model

def validate_weight_loading():
    """Test weight loading with actual checkpoint files"""
    
    print("\n=== WEIGHT LOADING VALIDATION ===")
    
    encoder_path = 'models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt'
    decoder_path = 'models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt'
    
    try:
        # Load the actual weights
        encoder_state = torch.load(encoder_path, map_location='cpu')
        decoder_state = torch.load(decoder_path, map_location='cpu')
        
        encoder_weights = encoder_state['state_dict'] if 'state_dict' in encoder_state else encoder_state
        decoder_weights = decoder_state['state_dict'] if 'state_dict' in decoder_state else decoder_state
        
        print("Sample encoder weight keys:")
        encoder_keys = list(encoder_weights.keys())
        for i, key in enumerate(encoder_keys[:10]):
            tensor = encoder_weights[key]
            print(f"  {key}: {tensor.shape}")
        
        print("\nSample decoder weight keys:")
        decoder_keys = list(decoder_weights.keys())
        for key in decoder_keys:
            tensor = decoder_weights[key]
            print(f"  {key}: {tensor.shape}")
        
        # Check for common patterns
        print("\nWeight key patterns:")
        patterns = {
            'mconv': [k for k in encoder_keys if 'mconv' in k],
            'res': [k for k in encoder_keys if 'res' in k],
            'conv.weight': [k for k in encoder_keys if 'conv.weight' in k],
            'running_mean': [k for k in encoder_keys if 'running_mean' in k],
            'running_var': [k for k in encoder_keys if 'running_var' in k],
        }
        
        for pattern, keys in patterns.items():
            print(f"  {pattern}: {len(keys)} matches")
            if keys:
                print(f"    Example: {keys[0]}")
        
        return encoder_weights, decoder_weights
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None, None

def main():
    """Main debug function"""
    
    print("QuartzNet Vietnamese - Enhanced Weight Mapping Debug")
    print("=" * 60)
    
    # Step 1: Analyze config and weight structure
    encoder_weights, decoder_weights, config = debug_weight_mapping()
    
    if encoder_weights is None:
        print("Failed to load weights, stopping debug")
        return
    
    # Step 2: Analyze separable convolution structure
    analyze_separable_conv_structure()
    
    # Step 3: Create and test minimal model
    minimal_model = create_minimal_test_model()
    
    # Step 4: Validate weight loading patterns
    validate_weight_loading()
    
    # Step 5: Provide final recommendations
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATIONS:")
    print("=" * 60)
    
    print("\n1. PARAMETER COUNT:")
    print("   - Your model currently creates ~10.2M parameters")
    print("   - Target should be ~5.1M parameters")
    print("   - Issue: Double-counting layers or wrong architecture")
    
    print("\n2. ARCHITECTURE ISSUES:")
    print("   - Use ModuleDict structure to match checkpoint naming")
    print("   - Implement proper separable convolutions")
    print("   - Don't create separate layers for depthwise/pointwise")
    print("   - Match exact layer structure from weight analysis above")
    
    print("\n3. WEIGHT LOADING:")
    print("   - Use exact key names from checkpoint")
    print("   - Handle BatchNorm parameters correctly")
    print("   - Test each block's weight loading individually")
    
    print("\n4. DEBUGGING STEPS:")
    print("   - Run the fixed model (in first artifact)")
    print("   - Check parameter count after each block creation")
    print("   - Verify weight shapes match before assignment")
    print("   - Test forward pass with dummy input")
    
    print("\n5. VERIFICATION:")
    print("   - Model parameters should be ~5.1M")
    print("   - Output shape should be [batch, time, vocab_size]")
    print("   - All weights should load without shape mismatches")
    
    return encoder_weights, decoder_weights

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskedConv1d(nn.Module):
    """1D Convolution with optional masking"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False):
        super(MaskedConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride, padding, dilation, groups, bias)
        
    def forward(self, x, mask=None):
        x = self.conv(x)
        if mask is not None:
            x = x * mask
        return x


class SeparableConv1d(nn.Module):
    """Separable 1D Convolution (Depthwise + Pointwise)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, bias=False):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                  stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class JasperBlock(nn.Module):
    """Jasper Block with residual connections"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 dilation=1, dropout=0.0, activation='relu', residual=True, 
                 separable=False, conv_mask=True):
        super(JasperBlock, self).__init__()
        
        self.residual = residual
        self.conv_mask = conv_mask
        
        # Calculate padding for 'same' padding
        padding = (kernel_size - 1) * dilation // 2
        
        # Choose convolution type
        if separable:
            self.conv = SeparableConv1d(in_channels, out_channels, kernel_size, 
                                       stride, padding, dilation)
        else:
            self.conv = MaskedConv1d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation)
        
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()  # default
            
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Residual connection
        if residual and in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        elif residual:
            self.residual_conv = None
        
    def forward(self, x, mask=None):
        residual = x
        
        # Main convolution path
        if self.conv_mask and mask is not None and hasattr(self.conv, 'forward'):
            if isinstance(self.conv, MaskedConv1d):
                x = self.conv(x, mask)
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)
            
        x = self.bn(x)
        x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Residual connection
        if self.residual:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            x = x + residual
            
        return x


class AudioToMelSpectrogramPreprocessor(nn.Module):
    """Audio preprocessing to mel spectrogram"""
    def __init__(self, sample_rate=16000, window_size=0.02, window_stride=0.01,
                 n_fft=512, features=64, normalize='per_feature', 
                 dither=0.00001, pad_to=16):
        super(AudioToMelSpectrogramPreprocessor, self).__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_fft = n_fft
        self.features = features
        self.normalize = normalize
        self.dither = dither
        self.pad_to = pad_to
        
        # Calculate window parameters
        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(window_stride * sample_rate)
        
        # Create mel filter bank
        self.register_buffer('mel_basis', self._create_mel_basis())
        
    def _create_mel_basis(self):
        """Create mel filter bank"""
        # Simplified mel basis creation
        return torch.randn(self.features, self.n_fft // 2 + 1)
    
    def forward(self, audio):
        # Simplified preprocessing - in practice would include STFT and mel conversion
        # For quantization testing, we'll create a placeholder
        batch_size = audio.shape[0]
        seq_len = audio.shape[1] // (self.hop_length)
        
        # Create dummy mel spectrogram features
        features = torch.randn(batch_size, self.features, seq_len, 
                              device=audio.device, dtype=audio.dtype)
        
        return features


class JasperEncoder(nn.Module):
    """Jasper/QuartzNet Encoder"""
    def __init__(self, jasper_config, activation='relu', conv_mask=True):
        super(JasperEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Build layers from config
        for i, layer_config in enumerate(jasper_config):
            filters = layer_config['filters']
            kernel_size = layer_config['kernel'][0]
            stride = layer_config['stride'][0]
            dilation = layer_config['dilation'][0]
            dropout = layer_config['dropout']
            residual = layer_config['residual']
            separable = layer_config.get('separable', False)
            repeat = layer_config.get('repeat', 1)
            
            # Determine input channels
            if i == 0:
                in_channels = 64  # from mel spectrogram features
            else:
                in_channels = jasper_config[i-1]['filters']
            
            # Create repeated blocks
            for j in range(repeat):
                block_in_channels = in_channels if j == 0 else filters
                
                block = JasperBlock(
                    in_channels=block_in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=stride if j == 0 else 1,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                    residual=residual,
                    separable=separable,
                    conv_mask=conv_mask
                )
                self.layers.append(block)
                
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class JasperDecoderForCTC(nn.Module):
    """CTC Decoder for Jasper"""
    def __init__(self, num_classes, hidden_size):
        super(JasperDecoderForCTC, self).__init__()
        self.decoder = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, features, time)
        x = x.transpose(1, 2)  # (batch, time, features)
        x = self.decoder(x)
        return x.log_softmax(dim=-1)


class QuartzNet(nn.Module):
    """Complete QuartzNet Model"""
    def __init__(self, config):
        super(QuartzNet, self).__init__()
        
        # Extract config parameters
        jasper_config = config['jasper']
        num_classes = len(config['labels'])
        activation = config.get('activation', 'relu')
        conv_mask = config.get('conv_mask', True)
        
        # Preprocessor
        preprocessor_config = config.get('preprocessor', {})
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preprocessor_config)
        
        # Encoder
        self.encoder = JasperEncoder(jasper_config, activation, conv_mask)
        
        # Decoder
        final_filters = jasper_config[-1]['filters']
        self.decoder = JasperDecoderForCTC(num_classes, final_filters)
        
    def forward(self, audio, audio_length=None):
        # Preprocess audio to mel spectrogram
        features = self.preprocessor(audio)
        
        # Create mask if needed
        mask = None
        if audio_length is not None:
            # Create mask based on audio length
            batch_size, _, max_time = features.shape
            mask = torch.ones(batch_size, 1, max_time, device=features.device)
            for i, length in enumerate(audio_length):
                # Simplified mask calculation
                actual_time = min(max_time, int(length * max_time / audio.shape[1]))
                if actual_time < max_time:
                    mask[i, :, actual_time:] = 0
        
        # Encode
        encoded = self.encoder(features, mask)
        
        # Decode
        logits = self.decoder(encoded)
        
        return logits


# Model configuration based on your config file
MODEL_CONFIG = {
    'jasper': [
        {'filters': 256, 'repeat': 1, 'kernel': [33], 'stride': [2], 'dilation': [1], 'dropout': 0.0, 'residual': False, 'separable': True},
        {'filters': 256, 'repeat': 1, 'kernel': [33], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 256, 'repeat': 1, 'kernel': [33], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 256, 'repeat': 1, 'kernel': [33], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 256, 'repeat': 1, 'kernel': [39], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 256, 'repeat': 1, 'kernel': [39], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 256, 'repeat': 1, 'kernel': [39], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 512, 'repeat': 1, 'kernel': [51], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 512, 'repeat': 1, 'kernel': [51], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 512, 'repeat': 1, 'kernel': [51], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 512, 'repeat': 1, 'kernel': [63], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 512, 'repeat': 1, 'kernel': [63], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 512, 'repeat': 1, 'kernel': [63], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 512, 'repeat': 1, 'kernel': [75], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': True, 'separable': True},
        {'filters': 1024, 'repeat': 1, 'kernel': [1], 'stride': [1], 'dilation': [1], 'dropout': 0.0, 'residual': False},
    ],
    'labels': [' ', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'đ', 'ĩ', 'ũ', 'ơ', 'ư', 'ạ', 'ả', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ'],
    'activation': 'relu',
    'conv_mask': True,
    'preprocessor': {
        'sample_rate': 16000,
        'window_size': 0.02,
        'window_stride': 0.01,
        'n_fft': 512,
        'features': 64,
        'normalize': 'per_feature',
        'dither': 0.00001,
        'pad_to': 16
    }
}


def create_model():
    """Create and return the QuartzNet model"""
    model = QuartzNet(MODEL_CONFIG)
    return model


def load_model(checkpoint_path):
    """Load model from checkpoint"""
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_model()
    
    # Test with dummy input
    batch_size = 2
    audio_length = 16000 * 2  # 2 seconds of audio at 16kHz
    dummy_audio = torch.randn(batch_size, audio_length)
    dummy_lengths = torch.tensor([audio_length, audio_length // 2])
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_audio, dummy_lengths)
        print(f"Model output shape: {output.shape}")
        print(f"Expected: (batch_size, time_steps, num_classes)")
        print(f"Actual: ({output.shape[0]}, {output.shape[1]}, {output.shape[2]})")
        print(f"Number of classes: {len(MODEL_CONFIG['labels'])}")
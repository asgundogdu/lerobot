# Setting up SmolVLA GPU Server on macOS with MLX

This guide explains how to configure and run the SmolVLA GPU server on macOS using Apple Silicon with MLX optimizations.

## Prerequisites

- **macOS 12.3 or later**
- **Apple Silicon Mac** (M1, M2, M3, or later)
- **Python 3.8 or later**
- **Xcode Command Line Tools**: `xcode-select --install`

## Installation Steps

### 1. Install MLX

MLX is Apple's machine learning framework optimized for Apple Silicon:

```bash
# Install MLX using pip
pip install mlx

# Or using conda
conda install -c conda-forge mlx
```

### 2. Install PyTorch with MPS Support

Install the latest PyTorch with Metal Performance Shaders (MPS) backend:

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Or using conda
conda install pytorch torchvision torchaudio -c pytorch
```

### 3. Verify MPS and MLX Installation

Test both frameworks:

```python
# Test PyTorch MPS
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    x = torch.ones(1, device="mps")
    print(f"MPS test tensor: {x}")

# Test MLX
import mlx.core as mx
print(f"MLX available: True")
x = mx.array([1.0, 2.0, 3.0])
mx.eval(x)
print(f"MLX test array: {x}")
```

### 4. Install Additional Dependencies

```bash
# Install gRPC and other dependencies
pip install grpcio grpcio-tools
pip install Pillow numpy
pip install transformers
pip install safetensors
```

### 5. Generate Protocol Buffer Files

If you don't have the protobuf files, generate them:

```bash
# Generate Python gRPC code from .proto files
python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. robotics.proto
```

## Performance Optimizations

### 1. Memory Management

For optimal performance on Apple Silicon:

```python
# Set memory limit for MPS
import torch
if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
```

### 2. MLX Optimizations

The server uses MLX for:
- Image preprocessing operations
- State tensor preprocessing
- Certain numerical computations

### 3. MPS Autocast

The server automatically uses `torch.autocast` with MPS for:
- Reduced memory usage
- Improved performance on Apple Silicon
- Better numerical stability

## Running the Server

```bash
# Start the SmolVLA server
python lerobot/scripts/gpu_server.py
```

Expected output:
```
✅ MLX is available
🔄 Loading SmolVLA model for macOS...
✅ Using Apple Silicon MPS backend
✅ SmolVLA model loaded on mps
🔄 Initializing MLX optimizations...
✅ MLX optimizations initialized successfully
🟢 SmolVLA gRPC server up on 0.0.0.0:50051
```

## Configuration Options

### Environment Variables

```bash
# Force CPU usage (for debugging)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Set MLX device (if needed)
export MLX_DEVICE=gpu
```

### Model Configuration

Edit the server configuration in `gpu_server.py`:

```python
# Adjust dataset stats if needed
dataset_stats = {
    # Your specific normalization parameters
}

# Configure model features
config.input_features = {
    'observation.images.top_view': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    'observation.images.side_view': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,)),
}
```

## Troubleshooting

### MPS Issues

If you encounter MPS-related errors:

```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# If MPS is not available, the server will fall back to CPU
```

### MLX Issues

If MLX is not working:

```bash
# Install MLX development version
pip install git+https://github.com/ml-explore/mlx.git

# Check MLX installation
python -c "import mlx.core as mx; print('MLX working')"
```

### Memory Issues

For large models or limited memory:

```python
# Reduce batch size or use gradient checkpointing
# Add to your configuration:
config.use_gradient_checkpointing = True
```

## Performance Benchmarks

Typical performance on Apple Silicon:

- **M1**: ~50-100ms inference latency
- **M2**: ~30-80ms inference latency  
- **M3**: ~20-60ms inference latency

Performance varies based on:
- Model size and complexity
- Input image resolution
- Available system memory
- Background processes

## Advanced Configuration

### Custom MLX Operations

To add custom MLX operations, modify the `_init_mlx_optimizations` method:

```python
def _init_mlx_optimizations(self):
    # Add custom MLX operations here
    pass
```

### Hybrid PyTorch-MLX Pipeline

The server supports a hybrid approach:
1. Use MLX for preprocessing (fast on Apple Silicon)
2. Use PyTorch MPS for model inference (compatibility)
3. Use MLX for postprocessing where beneficial 
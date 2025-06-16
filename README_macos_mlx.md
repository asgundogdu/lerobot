# SmolVLA on macOS with MLX Configuration

This directory contains the configuration and setup files for running SmolVLA on macOS with Apple Silicon using MLX optimizations.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_macos_mlx.txt
   ```

2. **Test your setup:**
   ```bash
   python test_macos_setup.py
   ```

3. **Start the server:**
   ```bash
   python lerobot/scripts/gpu_server.py
   ```

## 📁 Files Overview

- **`gpu_server.py`** - Modified GPU server with macOS MLX optimizations
- **`setup_macos_mlx.md`** - Detailed setup instructions and troubleshooting
- **`requirements_macos_mlx.txt`** - Python dependencies for macOS MLX setup
- **`test_macos_setup.py`** - Test script to verify installation

## 🔧 Key Modifications for macOS MLX

### Device Management
- **Before:** `device = "cuda"`
- **After:** `device = "mps"` (Apple's Metal Performance Shaders) with CPU fallback

### Hybrid Framework Approach
- **PyTorch MPS:** Main model inference (compatibility)
- **MLX:** Preprocessing and optimization (Apple Silicon performance)
- **Automatic fallback:** CPU if MPS/MLX unavailable

### Performance Optimizations
- **MPS Autocast:** Reduced memory usage with `torch.autocast`
- **MLX Preprocessing:** Fast image and tensor operations
- **Unified Memory:** Leverages Apple Silicon's unified memory architecture

## 🍎 Apple Silicon Benefits

### Performance Improvements
- **Memory Efficiency:** Unified memory architecture eliminates data copying
- **Neural Engine:** Potential acceleration for certain operations
- **Power Efficiency:** Better performance per watt compared to discrete GPUs

### Framework Synergy
- **PyTorch MPS:** Broad compatibility with existing models
- **MLX:** Apple-optimized operations for maximum performance
- **Metal:** Direct GPU access through Apple's graphics framework

## 🔍 Architecture Overview

```
Input (Images + State) 
    ↓
MLX Preprocessing (Fast)
    ↓
PyTorch MPS Model (Compatible)
    ↓
MLX Postprocessing (Optional)
    ↓
Output (Actions)
```

## ⚙️ Configuration Options

### Environment Variables
```bash
# Force CPU usage
export PYTORCH_ENABLE_MPS_FALLBACK=1

# MLX device selection
export MLX_DEVICE=gpu
```

### Server Configuration
- **Auto-device selection:** MPS → CPU fallback
- **Memory management:** Configurable memory limits
- **Batch processing:** Optimized for Apple Silicon

## 📊 Expected Performance

| Device | Typical Latency | Memory Usage |
|--------|----------------|--------------|
| M1     | 50-100ms       | ~2-4GB       |
| M2     | 30-80ms        | ~2-4GB       |
| M3     | 20-60ms        | ~2-4GB       |

*Performance varies based on model size, input resolution, and system load*

## 🛠️ Troubleshooting

### Common Issues

1. **MPS not available:**
   - Check macOS version (12.3+)
   - Verify Apple Silicon hardware
   - Update PyTorch to latest version

2. **MLX import errors:**
   - Install with: `pip install mlx`
   - Check Apple Silicon compatibility

3. **Memory issues:**
   - Reduce batch size
   - Enable gradient checkpointing
   - Close other applications

### Debug Commands
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Test MLX
python -c "import mlx.core as mx; print('MLX OK')"

# Run full test suite
python test_macos_setup.py
```

## 🤝 Contributing

To improve macOS MLX support:

1. **Add MLX operations:** Extend `_init_mlx_optimizations()`
2. **Profile performance:** Add benchmarking code
3. **Test compatibility:** Verify with different model configurations

## 📚 References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Silicon ML Guide](https://developer.apple.com/metal/pytorch/)

## 🎯 Next Steps

1. **Model Optimization:** Convert more operations to MLX
2. **Quantization:** Implement MLX-based model quantization
3. **Distributed Inference:** Multi-device support across Apple Silicon devices

---

For detailed setup instructions, see `setup_macos_mlx.md` 
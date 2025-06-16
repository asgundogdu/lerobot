#!/usr/bin/env python3
"""
Test script to verify macOS MLX setup for SmolVLA
Run this script to check if all dependencies are properly installed
"""

import sys
import traceback

def test_pytorch_mps():
    """Test PyTorch MPS availability and functionality"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        if torch.backends.mps.is_available():
            print("✅ MPS backend is available")
            
            # Test basic tensor operations on MPS
            x = torch.randn(3, 3, device="mps")
            y = torch.randn(3, 3, device="mps")
            z = torch.matmul(x, y)
            print(f"✅ MPS tensor operations working: shape={z.shape}")
            
            # Test autocast
            with torch.autocast(device_type="mps", dtype=torch.float16):
                result = torch.matmul(x, y).sum()
                print(f"✅ MPS autocast working: result={result:.4f}")
                
            return True
        else:
            print("⚠️  MPS backend not available - will use CPU")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch MPS test failed: {e}")
        traceback.print_exc()
        return False

def test_mlx():
    """Test MLX availability and functionality"""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        print("✅ MLX imported successfully")
        
        # Test basic array operations
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([4.0, 5.0, 6.0])
        z = mx.add(x, y)
        mx.eval(z)  # Force evaluation
        print(f"✅ MLX array operations working: {z}")
        
        # Test matrix operations
        a = mx.random.normal(shape=(3, 3))
        b = mx.random.normal(shape=(3, 3))
        c = mx.matmul(a, b)
        mx.eval(c)
        print(f"✅ MLX matrix operations working: shape={c.shape}")
        
        return True
        
    except ImportError:
        print("⚠️  MLX not available - install with: pip install mlx")
        return False
    except Exception as e:
        print(f"❌ MLX test failed: {e}")
        traceback.print_exc()
        return False

def test_grpc():
    """Test gRPC availability"""
    try:
        import grpc
        print("✅ gRPC imported successfully")
        return True
    except ImportError:
        print("❌ gRPC not available - install with: pip install grpcio grpcio-tools")
        return False

def test_image_processing():
    """Test image processing libraries"""
    try:
        from PIL import Image
        import numpy as np
        print("✅ PIL (Pillow) imported successfully")
        
        # Create a test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        print(f"✅ Image processing working: size={img.size}")
        
        return True
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def test_transformers():
    """Test transformers library"""
    try:
        from transformers import AutoProcessor
        print("✅ Transformers imported successfully")
        return True
    except ImportError:
        print("❌ Transformers not available - install with: pip install transformers")
        return False

def test_safetensors():
    """Test safetensors library"""
    try:
        import safetensors
        print("✅ Safetensors imported successfully")
        return True
    except ImportError:
        print("❌ Safetensors not available - install with: pip install safetensors")
        return False

def test_system_info():
    """Display system information"""
    import platform
    print(f"\n📋 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version}")
    
    # Check if running on Apple Silicon
    if platform.machine() == "arm64":
        print("✅ Running on Apple Silicon")
    else:
        print("⚠️  Not running on Apple Silicon - MLX optimizations may not be available")

def main():
    """Run all tests"""
    print("🧪 Testing macOS MLX setup for SmolVLA...")
    print("=" * 50)
    
    test_system_info()
    print("\n🔍 Testing dependencies:")
    print("-" * 30)
    
    results = {
        "PyTorch MPS": test_pytorch_mps(),
        "MLX": test_mlx(),
        "gRPC": test_grpc(),
        "Image Processing": test_image_processing(),
        "Transformers": test_transformers(),
        "Safetensors": test_safetensors(),
    }
    
    print(f"\n📊 Test Results:")
    print("-" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Your macOS MLX setup is ready for SmolVLA.")
        print("\n🚀 Next steps:")
        print("   1. Start the server: python lerobot/scripts/gpu_server.py")
        print("   2. Test with a client or check the health endpoint")
    else:
        print("⚠️  Some tests failed. Please install missing dependencies:")
        print("   pip install -r requirements_macos_mlx.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
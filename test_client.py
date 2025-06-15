#!/usr/bin/env python3
"""
Test client for SmolVLA gRPC server
"""
import grpc
import sys
import os

# Add scripts directory to path to import the generated protobuf files
sys.path.append(os.path.join(os.path.dirname(__file__), 'lerobot', 'scripts'))

try:
    import robotics_pb2
    import robotics_pb2_grpc
except ImportError as e:
    print(f"❌ Failed to import protobuf files: {e}")
    print("Make sure you've run: ./lerobot/scripts/compile_proto.sh")
    sys.exit(1)

def test_health_check():
    """Test the health check endpoint"""
    try:
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = robotics_pb2_grpc.SmolVLAServiceStub(channel)
            
            # Test health check
            health_request = robotics_pb2.HealthRequest()
            print("🔍 Testing health check...")
            response = stub.HealthCheck(health_request, timeout=10)
            
            print(f"✅ Health Check Response:")
            print(f"   Healthy: {response.healthy}")
            print(f"   Status: {response.status}")
            print(f"   Uptime: {response.uptime_seconds}s")
            
            return True
            
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_prediction():
    """Test a simple prediction with proper data format"""
    try:
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = robotics_pb2_grpc.SmolVLAServiceStub(channel)
            
            # Create proper test images based on training stats
            from PIL import Image
            import io
            import numpy as np
            
            print("🔧 Generating dummy data matching SmolVLA training format...")
            
            # Standard image size for SmolVLA
            height, width = 224, 224
            
            # Generate top_view image with its specific stats
            # mean=[0.49967906, 0.4697105, 0.4888590], std=[0.23773749, 0.234392, 0.22999627]
            top_view_data = np.random.normal(
                loc=[0.49967906, 0.4697105, 0.4888590],
                scale=[0.23773749, 0.234392, 0.22999627], 
                size=(height, width, 3)
            )
            top_view_data = np.clip(top_view_data, 0.0, 1.0)
            top_view_image = Image.fromarray((top_view_data * 255).astype(np.uint8))
            
            # Generate side_view image with its specific stats  
            # mean=[0.48981499, 0.49258736, 0.46523311], std=[0.16881491, 0.17684786, 0.19105635]
            side_view_data = np.random.normal(
                loc=[0.48981499, 0.49258736, 0.46523311],
                scale=[0.16881491, 0.17684786, 0.19105635],
                size=(height, width, 3)
            )
            side_view_data = np.clip(side_view_data, 0.0, 1.0)
            side_view_image = Image.fromarray((side_view_data * 255).astype(np.uint8))
            
            # Encode both images to bytes
            top_view_buffer = io.BytesIO()
            top_view_image.save(top_view_buffer, format='JPEG')
            top_view_bytes = top_view_buffer.getvalue()
            
            side_view_buffer = io.BytesIO()
            side_view_image.save(side_view_buffer, format='JPEG')
            side_view_bytes = side_view_buffer.getvalue()
            
            # Create realistic robot state (6 DOF) using training means
            robot_state = [
                20.925,   # Joint 1
                -34.089,  # Joint 2  
                31.721,   # Joint 3
                34.798,   # Joint 4
                6.348,    # Joint 5
                22.236    # Joint 6
            ]
            
            # Create observation request with separate camera views
            request = robotics_pb2.ObservationRequest(
                top_view_camera=top_view_bytes,    # Separate top view camera
                side_view_camera=side_view_bytes,  # Separate side view camera
                robot_state=robot_state,           # 6-DOF robot state
                task_instruction="move the robot forward",  # Text instruction!
                timestamp_ms=1234567890
            )
            
            print("🤖 Testing prediction with proper dual-camera setup...")
            print(f"   📷 Top view image: {len(top_view_bytes)} bytes")
            print(f"   📷 Side view image: {len(side_view_bytes)} bytes") 
            print(f"   🤖 Robot state: {robot_state}")
            print(f"   💬 Task: 'move the robot forward'")
            
            response = stub.PredictAction(request, timeout=30)
            
            print(f"✅ Prediction Response:")
            print(f"   Success: {response.success}")
            print(f"   Actions: {response.actions[:5]}..." if len(response.actions) > 5 else f"   Actions: {response.actions}")
            print(f"   Latency: {response.latency_ms}ms")
            
            return True
            
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing SmolVLA gRPC Server")
    print("=" * 50)
    
    # Test health check first
    if test_health_check():
        print("\n" + "=" * 50)
        # If health check passes, test prediction
        test_prediction()
    else:
        print("❌ Server not responding to health checks")
        sys.exit(1)
        
    print("\n🎉 All tests completed!") 
# Distributed SmolVLA Robot Control System

This guide explains how to set up a distributed robot control system where:
- **Remote GPU Server**: Runs SmolVLA inference with high-performance GPUs
- **Local Robot Machine**: Controls the physical robot and cameras

## Architecture Overview

```
┌─────────────────────────┐    gRPC     ┌─────────────────────────┐
│   Local Robot Machine   │◄──────────►│   Remote GPU Server     │
│                         │             │                         │
│ • Robot Hardware        │             │ • SmolVLA Model         │
│ • Camera System         │             │ • NVIDIA GPU            │
│ • robot_client.py       │             │ • gpu_server.py         │
│ • Safety Systems        │             │ • Model Cache           │
└─────────────────────────┘             └─────────────────────────┘
```

## Prerequisites

### Remote GPU Server Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3080+ recommended)
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Python**: 3.8+
- **Network**: Stable internet connection with low latency to robot

### Local Robot Machine Requirements
- **Robot**: 6-DOF robotic arm with control interface
- **Cameras**: 2 cameras (top_view and side_view) with 224x224 resolution
- **Python**: 3.8+
- **Network**: Stable connection to GPU server (< 100ms latency preferred)

## Installation

### 1. Remote GPU Server Setup

```bash
# Clone the repository
git clone <your-lerobot-repo>
cd lerobot

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pillow grpcio grpcio-tools numpy

# Install LeRobot
pip install -e .

# Authenticate with Hugging Face (for SmolVLA model access)
huggingface-cli login

# Compile protobuf files
chmod +x lerobot/scripts/compile_proto.sh
./lerobot/scripts/compile_proto.sh

# Test the server
python lerobot/scripts/gpu_server.py
```

### 2. Local Robot Machine Setup

```bash
# Clone the repository (same as GPU server)
git clone <your-lerobot-repo>
cd lerobot

# Install minimal dependencies (no CUDA needed)
pip install grpcio pillow numpy

# Compile protobuf files
chmod +x lerobot/scripts/compile_proto.sh
./lerobot/scripts/compile_proto.sh

# Install robot-specific libraries
# TODO: Add your robot's Python SDK here
# pip install your-robot-sdk
```

## Configuration

### 1. Network Configuration

**GPU Server** (`gpu_server.py`):
- Runs on `0.0.0.0:50051` (accessible from network)
- Configure firewall to allow port 50051
- Consider using SSL/TLS for production

**Robot Client** (`robot_client.py`):
- Connects to GPU server via hostname/IP
- Configurable timeout and retry settings

### 2. Robot Hardware Integration

The `robot_client.py` now uses the LeRobot Robot interface, so you just need to:

1. **Configure your robot type** in the command line arguments
2. **Set up cameras** using LeRobot camera configurations
3. **Specify robot connection** (USB port, etc.)

**Supported Robot Types:**
- `so101_follower` - SO-101 Follower Arm
- `so100_follower` - SO-100 Follower Arm  
- `koch_follower` - Koch Follower Arm

**Camera Configuration Examples:**
```bash
# Single USB camera
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 224, height: 224, fps: 30}}"

# Dual USB cameras
--robot.cameras="{ 
    top_view: {type: opencv, index_or_path: 0, width: 224, height: 224, fps: 30},
    side_view: {type: opencv, index_or_path: 1, width: 224, height: 224, fps: 30}
}"

# RealSense camera
--robot.cameras="{ front: {type: realsense, serial_number: 123456789, width: 224, height: 224, fps: 30}}"
```

## Usage

### 1. Start GPU Server (Remote Machine)

```bash
# Start the SmolVLA inference server
python lerobot/scripts/gpu_server.py

# Expected output:
# 🔄 Loading SmolVLA model with dataset stats...
# ✅ SmolVLA model loaded with custom dataset stats!
# 🟢 SmolVLA gRPC server up on 0.0.0.0:50051
```

### 2. Test Connection (Local Machine)

```bash
# Test the connection to GPU server
python test_client.py

# Expected output:
# ✅ Health Check Response: Healthy: True
# ✅ Prediction Response: Success: True
```

### 3. Run Robot Control (Local Machine)

```bash
# Basic usage with SO-101 robot
python robot_client.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 224, height: 224, fps: 30}}" \
    --robot.id=my_robot \
    --gpu_server_host=your-gpu-server.com \
    --task_instruction="pick up the red block"

# Advanced usage with dual cameras
python robot_client.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.cameras="{ 
        top_view: {type: opencv, index_or_path: 0, width: 224, height: 224, fps: 30},
        side_view: {type: opencv, index_or_path: 1, width: 224, height: 224, fps: 30}
    }" \
    --robot.id=my_robot \
    --gpu_server_host=192.168.1.100 \
    --gpu_server_port=50051 \
    --task_instruction="move the robot to the left" \
    --control_frequency=20.0 \
    --display_actions=true
```

### 4. Emergency Stop

Create emergency stop file:
```bash
touch /tmp/robot_emergency_stop
```

Or press `Ctrl+C` to stop the control loop.

## Performance Optimization

### 1. Network Optimization
- Use wired Ethernet connection (avoid WiFi for critical applications)
- Optimize gRPC settings for low latency
- Consider edge computing for ultra-low latency

### 2. GPU Server Optimization
- Use GPU with sufficient VRAM (8GB+ recommended)
- Enable CUDA optimizations
- Consider model quantization for faster inference

### 3. Control Loop Optimization
- Adjust control frequency based on task requirements
- Implement predictive control for smoother motion
- Add motion planning for complex tasks

## Safety Considerations

### 1. Emergency Stop Systems
- Hardware emergency stop button
- Software emergency stop monitoring
- Network timeout handling
- Robot workspace limits

### 2. Network Reliability
- Connection monitoring and auto-reconnect
- Graceful degradation when network fails
- Local fallback behaviors

### 3. Robot Safety
- Joint limit enforcement
- Collision detection
- Force/torque monitoring
- Safe home position

## Troubleshooting

### Common Issues

**1. "401 Client Error: Unauthorized"**
```bash
# Re-authenticate with Hugging Face
huggingface-cli login
```

**2. "gRPC Connection Error"**
- Check network connectivity
- Verify firewall settings
- Confirm GPU server is running

**3. "must be real number, not list"**
- This should be fixed in the current version
- Check action tensor conversion in `gpu_server.py`

**4. High Latency (>200ms)**
- Check network quality
- Reduce image resolution if needed
- Optimize gRPC settings

### Debug Mode

Enable debug output:
```bash
# GPU Server debug
GRPC_VERBOSITY=DEBUG python lerobot/scripts/gpu_server.py

# Robot Client debug
python robot_client.py --gpu-server localhost --task "debug test" --frequency 1.0
```

## Production Deployment

### 1. Security
- Use TLS/SSL encryption for gRPC
- Implement authentication tokens
- Network segmentation and VPN

### 2. Monitoring
- Add logging and metrics collection
- Health monitoring dashboards
- Performance alerting

### 3. Scalability
- Load balancing for multiple robots
- Model caching and optimization
- Horizontal scaling of GPU servers

## Example Integration

Here's a complete example for a UR5 robot with RealSense cameras:

```python
# robot_hardware.py - Hardware integration example
import pyrealsense2 as rs
import urx

class UR5RealSenseIntegration:
    def __init__(self):
        # Initialize UR5 robot
        self.robot = urx.Robot("192.168.1.10")
        
        # Initialize RealSense cameras
        self.pipeline1 = rs.pipeline()  # top_view
        self.pipeline2 = rs.pipeline()  # side_view
        
    def capture_camera_images(self):
        # Capture from RealSense cameras
        frames1 = self.pipeline1.wait_for_frames()
        frames2 = self.pipeline2.wait_for_frames()
        
        color1 = frames1.get_color_frame()
        color2 = frames2.get_color_frame()
        
        # Convert to JPEG bytes...
        
    def read_robot_state(self):
        return self.robot.getj()  # Get joint positions
        
    def execute_robot_action(self, actions):
        self.robot.movej(actions, acc=0.5, vel=0.5)
        return True
```

This distributed system provides:
- ✅ **Scalability**: Multiple robots can use the same GPU server
- ✅ **Performance**: Dedicated GPU for inference, real-time robot control
- ✅ **Flexibility**: Easy to swap robot hardware or models
- ✅ **Safety**: Multiple emergency stop mechanisms
- ✅ **Monitoring**: Built-in performance tracking and debugging 
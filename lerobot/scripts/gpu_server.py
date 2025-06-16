# gpu_server.py - Remote SmolVLA Service for macOS with MLX
import asyncio
import grpc
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import robotics_pb2_grpc
import robotics_pb2
from lerobot.configs.types import FeatureType, PolicyFeature

# Check for MLX availability
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
    print("✅ MLX is available")
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️  MLX not available, falling back to PyTorch MPS")

class SmolVLAInferenceService(robotics_pb2_grpc.SmolVLAServiceServicer):
    def __init__(self):
        # Load SmolVLA model with proper dataset stats
        print("🔄 Loading SmolVLA model for macOS...")
        
        # Check for Apple Silicon and MPS availability
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ Using Apple Silicon MPS backend")
        else:
            self.device = "cpu"
            print("⚠️  MPS not available, using CPU")
        
        # Convert your training stats to the proper format
        dataset_stats = {
            "action": {
                "mean": torch.tensor([20.9647274017334, -36.77244186401367, 30.183874130249023, 34.86293029785156, 6.29755163192749, 20.77574348449707], dtype=torch.float32),
                "std": torch.tensor([19.393949508666992, 52.43775939941406, 53.70619583129883, 80.17243194580078, 16.645732879638672, 14.039448738098145], dtype=torch.float32)
            },
            "observation.state": {
                "mean": torch.tensor([20.925312042236328, -34.08867263793945, 31.721282958984375, 34.7981071472168, 6.34783411026001, 22.236248016357422], dtype=torch.float32),
                "std": torch.tensor([19.295568466186523, 53.824893951416016, 52.096710205078125, 79.18147277832031, 16.619688034057617, 12.402323722839355], dtype=torch.float32)
            },
            "observation.images.top_view": {
                "mean": torch.tensor([[[0.49967906224299874]], [[0.4697105787127213]], [[0.4888590532975515]]], dtype=torch.float32),
                "std": torch.tensor([[[0.23773749695969776]], [[0.234392739224721]], [[0.22999627993389068]]], dtype=torch.float32)
            },
            "observation.images.side_view": {
                "mean": torch.tensor([[[0.4898149933630553]], [[0.49258736239160855]], [[0.46523311186137445]]], dtype=torch.float32),
                "std": torch.tensor([[[0.16881491582389432]], [[0.17684786109552253]], [[0.19105635131405319]]], dtype=torch.float32)
            }
        }
        
        # Load config and create model with our stats
        from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
        config = SmolVLAConfig()
        
        # Register expected input / output features so that config.image_features is populated
        config.input_features = {
            'observation.images.top_view': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            'observation.images.side_view': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        }
        config.output_features = {
            'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        }
        config.validate_features()
        
        # Create and load model
        #self.policy = SmolVLAPolicy(config, dataset_stats=dataset_stats).to(self.device).eval()
        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=config, dataset_stats=dataset_stats)
        self.policy.to(self.device).eval()
        print(f"✅ SmolVLA model loaded on {self.device}")
        
        # Initialize MLX arrays for optimization if available
        if MLX_AVAILABLE:
            self._init_mlx_optimizations()
            
    def _init_mlx_optimizations(self):
        """Initialize MLX optimizations for tensor operations"""
        print("🔄 Initializing MLX optimizations...")
        # We can use MLX for certain preprocessing operations
        # while keeping the main model in PyTorch for compatibility
        try:
            # Test MLX functionality
            test_array = mx.array([1.0, 2.0, 3.0])
            mx.eval(test_array)
            print("✅ MLX optimizations initialized successfully")
        except Exception as e:
            print(f"⚠️  MLX optimization initialization failed: {e}")
            
    def _pytorch_to_mlx(self, tensor):
        """Convert PyTorch tensor to MLX array"""
        if not MLX_AVAILABLE:
            return tensor
        try:
            # Convert to numpy first, then to MLX
            return mx.array(tensor.detach().cpu().numpy())
        except:
            return tensor
            
    def _mlx_to_pytorch(self, array):
        """Convert MLX array to PyTorch tensor"""
        if not MLX_AVAILABLE:
            return array
        try:
            # Convert MLX array to numpy, then to PyTorch
            numpy_array = np.array(array)
            return torch.from_numpy(numpy_array)
        except:
            return array
    
    async def PredictAction(self, request, context):
        """
        Process observation and return action prediction
        """
        # Deserialize observation data from the new protobuf format
        top_view_image = self._decode_image(request.top_view_camera)
        side_view_image = self._decode_image(request.side_view_camera)
        
        observation = {
            'observation.images.top_view': top_view_image,
            'observation.images.side_view': side_view_image,
            'observation.state': self._decode_state(request.robot_state),
            'task': request.task_instruction  # Text input!
        }
        
        # SmolVLA inference
        with torch.no_grad():
            # Use autocast for MPS optimization
            if self.device == "mps":
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    observation_frame = self._prepare_observation(observation)
                    action_tensor = self.policy.select_action(observation_frame)
            else:
                observation_frame = self._prepare_observation(observation)
                action_tensor = self.policy.select_action(observation_frame)
            
        # Convert action tensor to flat list of Python floats
        action_tensor = action_tensor.flatten()  # Ensure it's 1D
        action_list = [float(x) for x in action_tensor.cpu().numpy()]
        
        # Return structured action
        return robotics_pb2.ActionResponse(
            actions=action_list,
            success=True,
            latency_ms=self._calculate_latency()
        )
    
    def _decode_image(self, image_bytes):
        """Decode image bytes to tensor format with MLX optimization"""
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
        
        # Use MLX for preprocessing if available
        if MLX_AVAILABLE:
            try:
                # Use MLX for image preprocessing
                mlx_array = mx.array(image_array)
                # Perform any MLX-optimized operations here
                mx.eval(mlx_array)  # Force evaluation
                image_array = np.array(mlx_array)
            except Exception as e:
                print(f"⚠️  MLX preprocessing failed, using NumPy: {e}")
        
        image_tensor = torch.from_numpy(image_array)
        
        print(f"🔍 Decoded image: shape={image_tensor.shape}, dtype={image_tensor.dtype}, range=[{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
        return image_tensor
        
    def _decode_state(self, state_list):
        """Convert state list to tensor with MLX optimization"""
        if MLX_AVAILABLE:
            try:
                # Use MLX for state preprocessing
                mlx_array = mx.array(state_list)
                mx.eval(mlx_array)
                state_array = np.array(mlx_array)
                return torch.tensor(state_array, dtype=torch.float32)
            except:
                pass
        return torch.tensor(state_list, dtype=torch.float32)
        
    def _prepare_observation(self, observation):
        """Prepare observation for SmolVLA input"""
        # Convert images to proper tensor format for SmolVLA
        top_view = observation['observation.images.top_view']
        side_view = observation['observation.images.side_view']
        
        # Ensure images are float tensors in [0,1] range
        if top_view.dtype != torch.float32:
            top_view = top_view.float()
        if side_view.dtype != torch.float32:
            side_view = side_view.float()
        
        # Convert from (H, W, C) to (C, H, W) if needed
        if len(top_view.shape) == 3 and top_view.shape[2] == 3:
            top_view = top_view.permute(2, 0, 1)
            side_view = side_view.permute(2, 0, 1)
        
        # Add batch dimension and move to device
        observation_frame = {
            'observation.images.top_view': top_view.unsqueeze(0).to(self.device),
            'observation.images.side_view': side_view.unsqueeze(0).to(self.device), 
            'observation.state': observation['observation.state'].unsqueeze(0).to(self.device),
            'task': [observation['task']]  # List of task instructions
        }
        
        # Debug: Print tensor info
        print(f"🔍 Prepared observation:")
        print(f"   top_view shape: {observation_frame['observation.images.top_view'].shape}")
        print(f"   top_view dtype: {observation_frame['observation.images.top_view'].dtype}")
        print(f"   top_view range: [{observation_frame['observation.images.top_view'].min():.3f}, {observation_frame['observation.images.top_view'].max():.3f}]")
        print(f"   side_view shape: {observation_frame['observation.images.side_view'].shape}")
        print(f"   state shape: {observation_frame['observation.state'].shape}")
        print(f"   task: {observation_frame['task']}")
        print(f"   device: {self.device}")
        
        return observation_frame
        
    def _calculate_latency(self):
        """Calculate processing latency (placeholder)"""
        return 50  # milliseconds
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        return robotics_pb2.HealthResponse(
            healthy=True,
            status=f"SmolVLA service running on {self.device}",
            uptime_seconds=0  # Could track actual uptime
        )

async def serve(host="0.0.0.0", port=50051):
    # Configure for low latency
    options = [
        ('grpc.keepalive_time_ms', 10_000),
        ('grpc.keepalive_timeout_ms', 5_000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', 10_000),
        ('grpc.http2.min_ping_interval_without_data_ms', 300_000)
    ]
    
    # executor only matters for blocking business logic; aio itself is event-loop driven
    server = grpc.aio.server(ThreadPoolExecutor(max_workers=10), options=options)
    robotics_pb2_grpc.add_SmolVLAServiceServicer_to_server(
        SmolVLAInferenceService(), server)
    
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    print(f"🟢 SmolVLA gRPC server up on {host}:{port}")
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())

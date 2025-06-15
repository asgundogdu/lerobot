# gpu_server.py - Remote SmolVLA Service
import asyncio
import grpc
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import robotics_pb2_grpc
import robotics_pb2
from lerobot.configs.types import FeatureType, PolicyFeature

class SmolVLAInferenceService(robotics_pb2_grpc.SmolVLAServiceServicer):
    def __init__(self):
        # Load SmolVLA model with proper dataset stats
        print("🔄 Loading SmolVLA model with dataset stats...")
        
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
        
        # Method 1: Try to load pretrained model first (it should have stats)
        print("🔄 Loading model with custom dataset stats...")
        
        # Method 2: Load config and create model with our stats
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
        
        self.policy = SmolVLAPolicy(config, dataset_stats=dataset_stats).to("cuda").eval()
        print("✅ SmolVLA model loaded with custom dataset stats!")
            
        self.device = "cuda"
        
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
            # Prepare observation frame
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
        """Decode image bytes to tensor format"""
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and then to torch tensor
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
        image_tensor = torch.from_numpy(image_array)
        
        print(f"🔍 Decoded image: shape={image_tensor.shape}, dtype={image_tensor.dtype}, range=[{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
        return image_tensor
        
    def _decode_state(self, state_list):
        """Convert state list to tensor"""
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
        
        return observation_frame
        
    def _calculate_latency(self):
        """Calculate processing latency (placeholder)"""
        return 50  # milliseconds
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        return robotics_pb2.HealthResponse(
            healthy=True,
            status="SmolVLA service running",
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

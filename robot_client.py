#!/usr/bin/env python3
"""
Robot Control Client - Runs on local robot machine
Communicates with remote SmolVLA GPU server for inference
"""
import grpc
import sys
import os
import time
import threading
import logging
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
from pprint import pformat

import draccus
import numpy as np
from PIL import Image
import io

from lerobot.common.robots import (  # noqa: F401
    Robot, 
    RobotConfig, 
    make_robot_from_config,
    so101_follower,
    so100_follower,
    koch_follower,
)
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up

# Add scripts directory to path for protobuf imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'lerobot', 'scripts'))

try:
    import robotics_pb2
    import robotics_pb2_grpc
except ImportError as e:
    print(f"❌ Failed to import protobuf files: {e}")
    print("Make sure you've run: ./lerobot/scripts/compile_proto.sh")
    sys.exit(1)

@dataclass
class SmolVLAControlConfig:
    robot: RobotConfig
    gpu_server_host: str = "localhost"
    gpu_server_port: int = 50051
    task_instruction: str = "move the robot forward"
    control_frequency: float = 10.0
    max_duration_s: float | None = None
    display_actions: bool = True


class SmolVLARobotController:
    """
    SmolVLA Robot Controller that integrates with LeRobot Robot interface
    and communicates with remote GPU server for inference
    """
    
    def __init__(self, config: SmolVLAControlConfig):
        self.config = config
        self.gpu_server_address = f"{config.gpu_server_host}:{config.gpu_server_port}"
        self.running = False
        self.emergency_stop = False
        
        # Performance monitoring
        self.total_requests = 0
        self.total_latency = 0
        
        # Initialize robot from config
        self.robot = make_robot_from_config(config.robot)
        
        logging.info(f"🤖 SmolVLA Robot Controller initialized")
        logging.info(f"   Robot: {self.robot}")
        logging.info(f"   GPU Server: {self.gpu_server_address}")
        logging.info(f"   Task: '{config.task_instruction}'")
        
    def connect_to_gpu_server(self) -> bool:
        """Test connection to remote GPU server"""
        try:
            with grpc.insecure_channel(self.gpu_server_address) as channel:
                stub = robotics_pb2_grpc.SmolVLAServiceStub(channel)
                
                # Test health check
                health_request = robotics_pb2.HealthRequest()
                response = stub.HealthCheck(health_request, timeout=5)
                
                if response.healthy:
                    logging.info(f"✅ Connected to GPU server: {response.status}")
                    return True
                else:
                    logging.error(f"❌ GPU server unhealthy: {response.status}")
                    return False
                    
        except grpc.RpcError as e:
            logging.error(f"❌ gRPC Connection Error: {e.code()} - {e.details()}")
            return False
        except Exception as e:
            logging.error(f"❌ Connection Error: {e}")
            return False
    
    def capture_camera_images(self) -> Tuple[bytes, bytes]:
        """
        Capture images from robot's camera system using LeRobot Robot interface
        Returns: (top_view_jpeg_bytes, side_view_jpeg_bytes)
        """
        # Get observation from robot (includes camera images)
        observation = self.robot.get_observation()
        
        # Extract camera images - adapt these keys based on your robot's camera configuration
        # Common camera keys: 'top_view', 'side_view', 'front', 'wrist', etc.
        camera_keys = [key for key in observation.keys() if not key.endswith('.pos')]
        
        if len(camera_keys) < 2:
            # Fallback: generate dummy images if not enough cameras
            logging.warning("Not enough cameras found, generating dummy images")
            return self._generate_dummy_images()
        
        # Take first two cameras as top_view and side_view
        top_view_key = camera_keys[0]
        side_view_key = camera_keys[1] if len(camera_keys) > 1 else camera_keys[0]
        
        top_view_image = observation[top_view_key]
        side_view_image = observation[side_view_key]
        
        # Convert numpy arrays to JPEG bytes
        top_view_bytes = self._image_to_jpeg_bytes(top_view_image)
        side_view_bytes = self._image_to_jpeg_bytes(side_view_image)
        
        return top_view_bytes, side_view_bytes
    
    def _generate_dummy_images(self) -> Tuple[bytes, bytes]:
        """Generate dummy images when cameras are not available"""
        height, width = 224, 224
        
        top_view_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        side_view_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        top_view_bytes = self._image_to_jpeg_bytes(top_view_data)
        side_view_bytes = self._image_to_jpeg_bytes(side_view_data)
        
        return top_view_bytes, side_view_bytes
    
    def _image_to_jpeg_bytes(self, image_array: np.ndarray) -> bytes:
        """Convert numpy image array to JPEG bytes"""
        # Ensure image is in correct format (H, W, C) and uint8
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Resize to 224x224 if needed (SmolVLA requirement)
        if image_array.shape[:2] != (224, 224):
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image_array)
            pil_image = pil_image.resize((224, 224))
            image_array = np.array(pil_image)
        
        # Convert to JPEG bytes
        pil_image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
    
    def read_robot_state(self) -> list:
        """
        Read current robot joint positions using LeRobot Robot interface
        Returns: List of joint positions
        """
        observation = self.robot.get_observation()
        
        # Extract joint positions (keys ending with '.pos')
        joint_positions = []
        for key, value in observation.items():
            if key.endswith('.pos'):
                joint_positions.append(float(value))
        
        return joint_positions
    
    def execute_robot_action(self, actions: list) -> bool:
        """
        Execute action commands using LeRobot Robot interface
        Args:
            actions: List of joint commands
        Returns: True if successful
        """
        if self.emergency_stop:
            logging.warning("🛑 Emergency stop active - ignoring action")
            return False
        
        try:
            # Convert action list to robot action format
            action_dict = self._convert_actions_to_robot_format(actions)
            
            # Send action to robot
            actual_action = self.robot.send_action(action_dict)
            
            if self.config.display_actions:
                logging.info(f"🤖 Action sent: {[f'{v:.2f}' for v in actions]}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to execute robot action: {e}")
            return False
    
    def _convert_actions_to_robot_format(self, actions: list) -> dict:
        """Convert action list to robot's expected action dictionary format"""
        action_dict = {}
        action_features = self.robot.action_features
        
        # Map actions to robot's action features
        action_keys = list(action_features.keys())
        for i, action_value in enumerate(actions):
            if i < len(action_keys):
                action_dict[action_keys[i]] = float(action_value)
        
        return action_dict
    
    def get_inference_from_gpu_server(self, task_instruction: str) -> Optional[list]:
        """
        Send observation to GPU server and get action prediction
        """
        try:
            # Capture current observation
            top_view_bytes, side_view_bytes = self.capture_camera_images()
            robot_state = self.read_robot_state()
            
            # Create gRPC request
            request = robotics_pb2.ObservationRequest(
                top_view_camera=top_view_bytes,
                side_view_camera=side_view_bytes,
                robot_state=robot_state,
                task_instruction=task_instruction,
                timestamp_ms=int(time.time() * 1000)
            )
            
            # Send to GPU server
            start_time = time.time()
            
            with grpc.insecure_channel(self.gpu_server_address) as channel:
                stub = robotics_pb2_grpc.SmolVLAServiceStub(channel)
                response = stub.PredictAction(request, timeout=10)
                
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Update performance stats
            self.total_requests += 1
            self.total_latency += latency_ms
            
            if response.success:
                logging.debug(f"✅ Inference successful (latency: {latency_ms:.1f}ms)")
                return list(response.actions)
            else:
                logging.error(f"❌ Inference failed: {response.error_message}")
                return None
                
        except grpc.RpcError as e:
            logging.error(f"❌ gRPC Error: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logging.error(f"❌ Inference Error: {e}")
            return None
    
    def emergency_stop_handler(self):
        """Emergency stop monitoring (run in separate thread)"""
        while self.running:
            # TODO: Monitor emergency stop button/signal
            # For now, check for keyboard interrupt or file-based stop signal
            try:
                if os.path.exists("/tmp/robot_emergency_stop"):
                    self.emergency_stop = True
                    logging.critical("🛑 EMERGENCY STOP ACTIVATED")
                    # TODO: Send immediate stop commands to robot
                    break
            except:
                pass
            time.sleep(0.1)
    
    def run_control_loop(self):
        """
        Main robot control loop following teleoperate.py pattern
        """
        logging.info(f"🚀 Starting SmolVLA robot control loop")
        logging.info(f"   Task: '{self.config.task_instruction}'")
        logging.info(f"   Frequency: {self.config.control_frequency} Hz")
        logging.info(f"   Emergency stop: Create file '/tmp/robot_emergency_stop' to stop")
        
        # Connect to robot
        self.robot.connect()
        
        # Start emergency stop monitor
        self.running = True
        emergency_thread = threading.Thread(target=self.emergency_stop_handler)
        emergency_thread.daemon = True
        emergency_thread.start()
        
        # Control loop timing
        loop_period = 1.0 / self.config.control_frequency
        display_len = max(len(key) for key in self.robot.action_features) if self.config.display_actions else 0
        
        start_time = time.perf_counter()
        
        try:
            while self.running and not self.emergency_stop:
                loop_start = time.perf_counter()
                
                # Get action from GPU server
                actions = self.get_inference_from_gpu_server(self.config.task_instruction)
                
                if actions is not None:
                    # Execute action on robot
                    success = self.execute_robot_action(actions)
                    if not success:
                        logging.error("❌ Failed to execute robot action")
                        break
                        
                    # Display action info (similar to teleoperate.py)
                    if self.config.display_actions:
                        action_dict = self._convert_actions_to_robot_format(actions)
                        loop_time = time.perf_counter() - loop_start
                        
                        print("\n" + "-" * (display_len + 10))
                        print(f"{'JOINT':<{display_len}} | {'VALUE':>7}")
                        for joint, value in action_dict.items():
                            print(f"{joint:<{display_len}} | {value:>7.2f}")
                        print(f"\ntime: {loop_time * 1e3:.2f}ms ({1 / loop_time:.0f} Hz)")
                        move_cursor_up(len(action_dict) + 5)
                        
                else:
                    logging.warning("⚠️ No action received from GPU server")
                
                # Maintain control frequency using busy_wait (like teleoperate.py)
                loop_time = time.perf_counter() - loop_start
                busy_wait(loop_period - loop_time)
                
                # Check duration limit
                if (self.config.max_duration_s is not None and 
                    time.perf_counter() - start_time >= self.config.max_duration_s):
                    logging.info("⏰ Maximum duration reached")
                    break
                    
        except KeyboardInterrupt:
            logging.info("\n🛑 Control loop interrupted by user")
        except Exception as e:
            logging.error(f"❌ Control loop error: {e}")
        finally:
            self.running = False
            self.robot.disconnect()
            logging.info("🏁 Control loop stopped")
            
            # Print performance stats
            if self.total_requests > 0:
                avg_latency = self.total_latency / self.total_requests
                logging.info(f"📊 Performance Stats:")
                logging.info(f"   Total requests: {self.total_requests}")
                logging.info(f"   Average latency: {avg_latency:.1f}ms")

@draccus.wrap()
def main(cfg: SmolVLAControlConfig):
    """Main entry point following teleoperate.py pattern"""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Create SmolVLA robot controller
    controller = SmolVLARobotController(cfg)
    
    # Test connection to GPU server first
    if not controller.connect_to_gpu_server():
        logging.error("❌ Cannot connect to GPU server. Exiting.")
        sys.exit(1)
    
    try:
        # Run control loop
        controller.run_control_loop()
    except KeyboardInterrupt:
        logging.info("🛑 Interrupted by user")
    finally:
        logging.info("🏁 SmolVLA robot control finished")

if __name__ == "__main__":
    main() 
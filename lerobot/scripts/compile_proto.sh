#!/bin/bash
# compile_proto.sh - Compile protobuf files for gRPC

cd "$(dirname "$0")/../.."

echo "🔨 Compiling protobuf files..."

# Install grpcio-tools if not present
pip install grpcio-tools

# Compile proto files
python -m grpc_tools.protoc \
  -I ./protos \
  --python_out=./lerobot/scripts \
  --grpc_python_out=./lerobot/scripts \
  protos/robotics.proto

echo "✅ Protobuf compilation complete!"
echo "Generated files:"
echo "  - lerobot/scripts/robotics_pb2.py"
echo "  - lerobot/scripts/robotics_pb2_grpc.py" 
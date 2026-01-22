# CosyVoice Triton Server

Triton Inference Server deployment for FastCosyVoice TTS with voice caching.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Triton Server                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              CosyVoice Python Backend               │   │
│  │                                                     │   │
│  │  • Voice Cache (in-memory)                          │   │
│  │  • Streaming TTS Inference                          │   │
│  │  • MongoDB Integration (startup load)               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Ports: 8000 (HTTP), 8001 (gRPC), 8002 (Metrics)           │
└─────────────────────────────────────────────────────────────┘
```

## Operations

| Operation | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| `tts_cached` | TTS with cached voice | voice_id, text, instruction | audio stream |
| `tts_custom` | TTS with uploaded audio | audio_base64, transcription, text, instruction | audio stream |
| `cache_voice` | Cache new voice | voice_id, audio_base64, transcription | status |
| `evict_voice` | Remove voice from cache | voice_id | status |
| `list_voices` | List cached voices | - | voice_ids |

## Quick Start

### Build and Run

```bash
cd FastCosyVoice

# Build the image (from project root)
docker build -t fastcosyvoice-triton:24.07 .

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your MongoDB credentials

# Run Triton
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --env-file .env \
  fastcosyvoice-triton:24.07

# Check health
curl http://localhost:8000/v2/health/ready
```

### Test TTS

```python
import tritonclient.grpc as grpcclient
import numpy as np

client = grpcclient.InferenceServerClient("localhost:8001")

# Create inputs
operation = np.array(["tts_cached"], dtype=object)
voice_id = np.array(["your-voice-id"], dtype=object)
text = np.array(["Hello, this is a test."], dtype=object)
instruction = np.array(["You are a helpful assistant."], dtype=object)

inputs = [
    grpcclient.InferInput("operation", operation.shape, "BYTES"),
    grpcclient.InferInput("voice_id", voice_id.shape, "BYTES"),
    grpcclient.InferInput("text", text.shape, "BYTES"),
    grpcclient.InferInput("instruction", instruction.shape, "BYTES"),
]

inputs[0].set_data_from_numpy(operation)
inputs[1].set_data_from_numpy(voice_id)
inputs[2].set_data_from_numpy(text)
inputs[3].set_data_from_numpy(instruction)

outputs = [
    grpcclient.InferRequestedOutput("audio"),
    grpcclient.InferRequestedOutput("sample_rate"),
    grpcclient.InferRequestedOutput("status"),
]

# For streaming, use stream=True
result = client.infer("cosyvoice", inputs, outputs=outputs)

audio = result.as_numpy("audio")
sample_rate = result.as_numpy("sample_rate")[0]
status = result.as_numpy("status")[0].decode()

print(f"Status: {status}")
print(f"Audio shape: {audio.shape}, Sample rate: {sample_rate}")
```

## Environment Variables

Copy `.env.example` to `.env` and configure your values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `MONGO_URI` | MongoDB connection string |
| `MONGO_DB` | Database name |
| `MONGO_COLLECTION` | Collection for voices |
| `COSYVOICE_MODEL_DIR` | Path to CosyVoice model (default: `/models/Fun-CosyVoice3-0.5B`) |

## Model Repository Structure

```
models/
└── cosyvoice/
    ├── config.pbtxt      # Triton model configuration
    └── 1/
        ├── __init__.py
        └── model.py      # Python backend implementation
```

## Voice Cache Workflow

### Startup
1. Triton starts and loads CosyVoice model
2. Connects to MongoDB
3. Loads all voices with "Narrative" emotion
4. Extracts embeddings and caches in memory

### Runtime (New Voice)
1. Pipecat saves voice to MongoDB
2. Pipecat calls `cache_voice` operation with audio + transcription
3. Triton extracts embedding and adds to cache
4. Voice ready for TTS

### Runtime (TTS)
1. Client sends `tts_cached` with voice_id and text
2. Triton looks up embedding in cache (fast!)
3. Runs inference and streams audio back

## Monitoring

- Health: `http://localhost:8000/v2/health/ready`
- Metrics: `http://localhost:8002/metrics`
- Model status: `http://localhost:8000/v2/models/cosyvoice`

## Troubleshooting

### Model not loading
```bash
# Check Triton logs
docker compose logs triton | grep -i error

# Verify model files exist
docker compose exec triton ls -la /models/Fun-CosyVoice3-0.5B/
```

### MongoDB connection issues
```bash
# Test MongoDB connectivity from container
docker compose exec triton python -c "
import os
from pymongo import MongoClient
client = MongoClient(os.environ['MONGO_URI'])
print(client.list_database_names())
"
```

### Voice not found
```bash
# List cached voices
curl -X POST http://localhost:8000/v2/models/cosyvoice/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"operation","shape":[1],"datatype":"BYTES","data":["list_voices"]}]}'
```

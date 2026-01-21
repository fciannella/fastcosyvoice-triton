#!/usr/bin/env python3
"""
Test client for CosyVoice Triton server.

Usage:
    python test_triton_client.py --server 10.49.166.237:8001 --operation list_voices
    python test_triton_client.py --server 10.49.166.237:8001 --operation tts_cached --voice-id <id> --text "Hello world"
"""

import argparse
import numpy as np
import wave
import io
import sys

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
except ImportError:
    print("Error: tritonclient not installed. Install with:")
    print("  pip install tritonclient[grpc]")
    sys.exit(1)


def list_voices(client: grpcclient.InferenceServerClient, model_name: str = "cosyvoice"):
    """List all cached voices using streaming (required for decoupled model)."""
    # Create inputs
    inputs = [
        grpcclient.InferInput("operation", [1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(np.array(["list_voices"], dtype=object))
    
    # Create outputs
    outputs = [
        grpcclient.InferRequestedOutput("status"),
        grpcclient.InferRequestedOutput("voice_ids"),
    ]
    
    print("Requesting list of cached voices...")
    
    # For decoupled model, use streaming API
    voice_ids = []
    status = None
    
    def callback(result, error):
        nonlocal voice_ids, status
        if error:
            print(f"Error: {error}")
            return
        
        status = result.as_numpy("status")[0].decode("utf-8")
        voice_ids_arr = result.as_numpy("voice_ids")
        
        for vid in voice_ids_arr:
            if isinstance(vid, bytes):
                vid = vid.decode("utf-8")
            voice_ids.append(vid)
    
    # Start streaming request
    client.start_stream(callback=callback)
    client.async_stream_infer(model_name=model_name, inputs=inputs, outputs=outputs)
    client.stop_stream()
    
    print(f"Status: {status}")
    print(f"Cached voices ({len(voice_ids)}):")
    for vid in voice_ids:
        print(f"  - {vid}")
    
    return voice_ids


def tts_cached(
    client: grpcclient.InferenceServerClient,
    voice_id: str,
    text: str,
    instruction: str = "You are a helpful assistant.",
    model_name: str = "cosyvoice",
    output_file: str = "output.wav",
):
    """Synthesize speech using a cached voice."""
    # Create inputs
    inputs = [
        grpcclient.InferInput("operation", [1], "BYTES"),
        grpcclient.InferInput("voice_id", [1], "BYTES"),
        grpcclient.InferInput("text", [1], "BYTES"),
        grpcclient.InferInput("instruction", [1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(np.array(["tts_cached"], dtype=object))
    inputs[1].set_data_from_numpy(np.array([voice_id], dtype=object))
    inputs[2].set_data_from_numpy(np.array([text], dtype=object))
    inputs[3].set_data_from_numpy(np.array([instruction], dtype=object))
    
    # Create outputs
    outputs = [
        grpcclient.InferRequestedOutput("audio"),
        grpcclient.InferRequestedOutput("sample_rate"),
        grpcclient.InferRequestedOutput("status"),
    ]
    
    print(f"Synthesizing: '{text[:50]}...' with voice {voice_id}")
    
    # For decoupled mode, we need to use streaming
    all_audio = []
    sample_rate = None
    
    # Use streaming for decoupled model
    def callback(result, error):
        if error:
            print(f"Error: {error}")
            return
        
        nonlocal sample_rate, all_audio
        
        status = result.as_numpy("status")[0].decode("utf-8")
        audio_chunk = result.as_numpy("audio")
        sr = result.as_numpy("sample_rate")[0]
        
        if sample_rate is None:
            sample_rate = sr
        
        if len(audio_chunk) > 0:
            all_audio.append(audio_chunk)
        
        print(f"  Received chunk: {len(audio_chunk)} samples, status={status}")
    
    # Start streaming request
    client.start_stream(callback=callback)
    client.async_stream_infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    # Wait for completion
    client.stop_stream()
    
    if all_audio:
        # Concatenate all audio
        full_audio = np.concatenate(all_audio)
        
        # Save to WAV file
        save_wav(full_audio, sample_rate, output_file)
        print(f"\n✅ Saved {len(full_audio)} samples ({len(full_audio)/sample_rate:.2f}s) to {output_file}")
    else:
        print("❌ No audio received")


def save_wav(audio: np.ndarray, sample_rate: int, filename: str):
    """Save audio array to WAV file."""
    # Convert float32 to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Test CosyVoice Triton server")
    parser.add_argument("--server", default="10.49.166.237:8001", help="Triton gRPC server address")
    parser.add_argument("--operation", choices=["list_voices", "tts_cached"], required=True)
    parser.add_argument("--voice-id", help="Voice ID for TTS")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    
    args = parser.parse_args()
    
    # Create client
    try:
        client = grpcclient.InferenceServerClient(url=args.server)
        
        # Check server is live
        if not client.is_server_live():
            print(f"❌ Server {args.server} is not live")
            sys.exit(1)
        
        print(f"✅ Connected to Triton server at {args.server}")
        
        # Check model is ready
        if not client.is_model_ready("cosyvoice"):
            print("❌ Model 'cosyvoice' is not ready")
            sys.exit(1)
        
        print("✅ Model 'cosyvoice' is ready\n")
        
    except InferenceServerException as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)
    
    # Execute operation
    if args.operation == "list_voices":
        list_voices(client)
    
    elif args.operation == "tts_cached":
        if not args.voice_id:
            print("❌ --voice-id is required for tts_cached")
            sys.exit(1)
        if not args.text:
            print("❌ --text is required for tts_cached")
            sys.exit(1)
        
        tts_cached(client, args.voice_id, args.text, output_file=args.output)


if __name__ == "__main__":
    main()

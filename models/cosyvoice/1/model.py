"""
CosyVoice TTS Model for Triton Inference Server

Python backend that provides:
- Voice caching with MongoDB persistence
- Streaming TTS inference
- Voice management operations

Operations:
- tts_cached: TTS using a cached voice (voice_id + text)
- tts_custom: TTS with uploaded audio (one-shot, no caching)
- cache_voice: Cache a new voice embedding
- evict_voice: Remove a voice from cache
- list_voices: List all cached voice IDs
"""

import os
import sys
import json
import base64
import tempfile
import logging
from typing import Generator, Optional

import numpy as np
import torch

# Add CosyVoice to path
# Note: matcha-tts is installed via pip, no need for third_party path
sys.path.append('/workspace/FastCosyVoice')

import triton_python_backend_utils as pb_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("cosyvoice_model")


class TritonPythonModel:
    """
    CosyVoice TTS model for Triton.
    
    Handles voice caching and streaming TTS inference.
    """
    
    def initialize(self, args):
        """
        Initialize model and load voices from MongoDB.
        
        Called once when Triton loads the model.
        """
        logger.info("=" * 60)
        logger.info("CosyVoice TTS Model Initializing...")
        logger.info("=" * 60)
        
        # Parse model config
        self.model_config = json.loads(args['model_config'])
        
        # Get parameters from config (Triton passes params as a dict)
        params = self.model_config.get('parameters', {})
        
        # Helper to get param value (handles both string and dict formats)
        def get_param(key, default=''):
            if key in params:
                val = params[key]
                if isinstance(val, dict):
                    return val.get('string_value', default)
                return val
            return default
        
        # Environment variables take precedence over config
        self.mongo_uri = os.environ.get('MONGO_URI', get_param('MONGO_URI', ''))
        self.mongo_db = os.environ.get('MONGO_DB', get_param('MONGO_DB', 'audio_sources_db'))
        self.mongo_collection = os.environ.get('MONGO_COLLECTION', get_param('MONGO_COLLECTION', 'audio_prompts_a2flow'))
        self.model_dir = os.environ.get('COSYVOICE_MODEL_DIR', get_param('COSYVOICE_MODEL_DIR', '/models/Fun-CosyVoice3-0.5B'))
        
        logger.info(f"MongoDB URI: {self.mongo_uri[:50]}...")
        logger.info(f"MongoDB DB: {self.mongo_db}")
        logger.info(f"MongoDB Collection: {self.mongo_collection}")
        logger.info(f"Model Directory: {self.model_dir}")
        
        # Voice cache: voice_id -> (spk_id, transcription)
        # No need to store audio_base64 - cached voices use prompt_wav=None for inference
        self.voice_cache: dict[str, tuple[str, str]] = {}
        
        # Load CosyVoice model
        self._load_model()
        
        # Load voices from MongoDB
        self._load_voices_from_mongodb()
        
        logger.info("=" * 60)
        logger.info(f"CosyVoice ready! Cached voices: {len(self.voice_cache)}")
        logger.info("=" * 60)
    
    def _load_model(self):
        """Load the CosyVoice model."""
        from fastcosyvoice import FastCosyVoice3
        
        logger.info(f"Loading CosyVoice from {self.model_dir}...")
        
        # Check if model exists
        if not os.path.exists(self.model_dir):
            raise RuntimeError(f"Model not found: {self.model_dir}")
        
        # Load model with TensorRT optimizations (same as main service)
        use_trt = os.environ.get('USE_TRT_FLOW', 'true').lower() == 'true'
        use_trt_llm = os.environ.get('USE_TRT_LLM', 'true').lower() == 'true'
        trt_llm_dtype = os.environ.get('TRT_LLM_DTYPE', 'bfloat16')
        
        logger.info(f"TensorRT Flow: {'Enabled' if use_trt else 'Disabled'}")
        logger.info(f"TensorRT-LLM: {'Enabled' if use_trt_llm else 'Disabled'} (dtype={trt_llm_dtype})")
        
        self.cosyvoice = FastCosyVoice3(
            model_dir=self.model_dir,
            fp16=True,
            load_trt=use_trt,
            load_trt_llm=use_trt_llm,
            trt_llm_dtype=trt_llm_dtype,
            trt_llm_kv_cache_tokens=8192,
        )
        
        self.sample_rate = self.cosyvoice.sample_rate
        logger.info(f"✅ Model loaded! Sample rate: {self.sample_rate} Hz")
        
        # Warmup
        self._warmup()
    
    def _warmup(self):
        """Warmup the model with a short generation."""
        logger.info("🔥 Warming up model...")
        
        import soundfile as sf
        
        # Create dummy audio
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), dtype=np.float32)
        dummy_audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, dummy_audio, self.sample_rate)
            temp_path = f.name
        
        try:
            prompt_text = "You are a helpful assistant.<|endofprompt|>Hello"
            spk_id = "_warmup_"
            
            self.cosyvoice.add_zero_shot_spk(prompt_text, temp_path, spk_id)
            
            for _ in self.cosyvoice.inference_zero_shot_stream(
                tts_text="Warmup",
                prompt_text=prompt_text,
                prompt_wav=temp_path,
                zero_shot_spk_id=spk_id,
            ):
                pass
            
            # Clean up warmup speaker (delete from spk2info directly)
            if spk_id in self.cosyvoice.frontend.spk2info:
                del self.cosyvoice.frontend.spk2info[spk_id]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            logger.info("✅ Warmup complete!")
        finally:
            os.unlink(temp_path)
    
    def _load_voices_from_mongodb(self):
        """Load all voices from MongoDB at startup."""
        if not self.mongo_uri:
            logger.warning("No MongoDB URI configured, skipping voice preload")
            return
        
        try:
            from pymongo import MongoClient
            import soundfile as sf
            
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db]
            collection = db[self.mongo_collection]
            
            # Fetch voices with Narrative emotion (excluding libritts)
            query = {
                "emotion": "Narrative",
                "actor_name": {"$not": {"$regex": "^libritts", "$options": "i"}}
            }
            voices = list(collection.find(query))
            
            logger.info(f"Found {len(voices)} voices in MongoDB")
            
            for voice in voices:
                voice_id = str(voice.get("_id"))
                actor_name = voice.get("actor_name", "Unknown")
                transcription = voice.get("transcription", "")
                audio_base64 = voice.get("audio_base64", "")
                
                if not audio_base64 or not transcription:
                    continue
                
                try:
                    self._cache_voice_internal(voice_id, audio_base64, transcription)
                    logger.info(f"  ✓ Loaded: {actor_name} ({voice_id})")
                except Exception as e:
                    logger.warning(f"  ✗ Failed: {actor_name}: {e}")
            
            client.close()
            
        except Exception as e:
            logger.error(f"Failed to load voices from MongoDB: {e}")
    
    def _cache_voice_internal(self, voice_id: str, audio_base64: str, transcription: str):
        """
        Cache a voice embedding internally.
        
        Args:
            voice_id: Unique voice identifier
            audio_base64: Base64-encoded WAV audio
            transcription: Transcription of the audio
        """
        import soundfile as sf
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            # Create internal speaker ID
            spk_id = f"cached_{voice_id}"
            
            # Use instruction prefix + transcription (MUST match service.py!)
            # This is how service.py preloads voices
            prompt_text = f"You are a helpful assistant.<|endofprompt|>{transcription}"
            
            # Register speaker (extracts and caches embedding in CosyVoice)
            self.cosyvoice.add_zero_shot_spk(prompt_text, temp_path, spk_id)
            
            # Store mapping with transcription only (no audio_base64 needed for inference)
            self.voice_cache[voice_id] = (spk_id, transcription)
            
            logger.info(f"Cached voice {voice_id}: spk_id={spk_id}, transcription='{transcription[:50]}...'")
            
        finally:
            os.unlink(temp_path)
    
    def execute(self, requests):
        """
        Execute inference requests.
        
        This is called for each batch of requests.
        For decoupled mode, we must return None (responses sent via response sender).
        """
        for request in requests:
            try:
                self._process_request(request)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                # Send error response via sender
                response_sender = request.get_response_sender()
                error_response = self._create_error_response(str(e))
                response_sender.send(error_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        
        # Decoupled mode requires returning None
        return None
    
    def _process_request(self, request):
        """Process a single request based on operation type."""
        
        # Get operation type
        operation = pb_utils.get_input_tensor_by_name(request, "operation")
        operation = operation.as_numpy()[0].decode('utf-8')
        
        logger.info(f"Processing operation: {operation}")
        
        if operation == "tts_cached":
            return self._handle_tts_cached(request)
        elif operation == "tts_custom":
            return self._handle_tts_custom(request)
        elif operation == "cache_voice":
            return self._handle_cache_voice(request)
        elif operation == "evict_voice":
            return self._handle_evict_voice(request)
        elif operation == "list_voices":
            return self._handle_list_voices(request)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _handle_tts_cached(self, request):
        """
        Handle TTS with a cached voice.
        
        Streams audio back using decoupled response mode.
        Uses the SAME approach as service.py for consistency.
        """
        # Get inputs
        voice_id = pb_utils.get_input_tensor_by_name(request, "voice_id")
        voice_id = voice_id.as_numpy()[0].decode('utf-8')
        
        text = pb_utils.get_input_tensor_by_name(request, "text")
        text = text.as_numpy()[0].decode('utf-8')
        
        # Get instruction (optional, defaults to "You are a helpful assistant.")
        instruction_tensor = pb_utils.get_input_tensor_by_name(request, "instruction")
        if instruction_tensor is not None:
            instruction = instruction_tensor.as_numpy()[0].decode('utf-8')
        else:
            instruction = "You are a helpful assistant."
        
        logger.info(f"TTS cached: voice_id={voice_id}, text='{text[:50]}...'")
        
        # Check if voice is cached
        if voice_id not in self.voice_cache:
            raise ValueError(f"Voice not found in cache: {voice_id}")
        
        # Get spk_id and transcription from cache
        spk_id, transcription = self.voice_cache[voice_id]
        
        # Format prompt_text with instruction (SAME as service.py!)
        prompt_text = f"{instruction}<|endofprompt|>{transcription}"
        
        logger.info(f"  spk_id: {spk_id}")
        logger.info(f"  prompt_text: '{prompt_text[:80]}...'")
        
        # Get response sender for streaming
        response_sender = request.get_response_sender()
        
        # Stream audio chunks - use prompt_wav=None (SAME as service.py!)
        # The voice embedding is already cached, no need for prompt audio
        
        # IMPORTANT: Use inference_mode() just like service.py does!
        # This affects model behavior and generation quality
        with torch.inference_mode():
        for pcm_bytes in self.cosyvoice.inference_zero_shot_stream(
            tts_text=text,
            prompt_text=prompt_text,
                prompt_wav=None,  # IMPORTANT: None for cached voices (matches service.py)
            zero_shot_spk_id=spk_id,
        ):
            # Convert PCM bytes to float32 numpy array
            audio_chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Send intermediate response
            audio_tensor = pb_utils.Tensor("audio", audio_chunk)
            sample_rate_tensor = pb_utils.Tensor("sample_rate", np.array([self.sample_rate], dtype=np.int32))
            status_tensor = pb_utils.Tensor("status", np.array(["streaming"], dtype=object))
            voice_ids_tensor = pb_utils.Tensor("voice_ids", np.array([""], dtype=object))
            
            response = pb_utils.InferenceResponse(output_tensors=[
                audio_tensor,
                sample_rate_tensor,
                status_tensor,
                voice_ids_tensor,
            ])
            response_sender.send(response)
        
        # Send final "complete" response with EMPTY audio
        # (Don't send duplicate audio - streaming chunks already sent everything)
        empty_audio = np.array([], dtype=np.float32)
        
        audio_tensor = pb_utils.Tensor("audio", empty_audio)
        sample_rate_tensor = pb_utils.Tensor("sample_rate", np.array([self.sample_rate], dtype=np.int32))
        status_tensor = pb_utils.Tensor("status", np.array(["complete"], dtype=object))
        voice_ids_tensor = pb_utils.Tensor("voice_ids", np.array([""], dtype=object))
        
        response = pb_utils.InferenceResponse(output_tensors=[
            audio_tensor,
            sample_rate_tensor,
            status_tensor,
            voice_ids_tensor,
        ])
        response_sender.send(response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        
        return None  # Response already sent via sender
    
    def _handle_tts_custom(self, request):
        """
        Handle TTS with custom (uploaded) audio.
        
        One-shot TTS without caching the voice.
        """
        import soundfile as sf
        
        # Get inputs
        audio_base64 = pb_utils.get_input_tensor_by_name(request, "audio_base64")
        audio_base64 = audio_base64.as_numpy()[0].decode('utf-8')
        
        transcription = pb_utils.get_input_tensor_by_name(request, "transcription")
        transcription = transcription.as_numpy()[0].decode('utf-8')
        
        text = pb_utils.get_input_tensor_by_name(request, "text")
        text = text.as_numpy()[0].decode('utf-8')
        
        instruction_tensor = pb_utils.get_input_tensor_by_name(request, "instruction")
        instruction = "You are a helpful assistant."
        if instruction_tensor is not None:
            instruction = instruction_tensor.as_numpy()[0].decode('utf-8')
        
        logger.info(f"TTS custom: text={text[:50]}...")
        
        # Decode and save audio
        audio_bytes = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            # Create temporary speaker
            spk_id = f"_temp_{id(request)}"
            prompt_text = f"{instruction}<|endofprompt|>{transcription}"
            
            self.cosyvoice.add_zero_shot_spk(prompt_text, temp_path, spk_id)
            
            # Get response sender for streaming
            response_sender = request.get_response_sender()
            
            # Stream audio with inference_mode() (SAME as service.py!)
            with torch.inference_mode():
            for pcm_bytes in self.cosyvoice.inference_zero_shot_stream(
                tts_text=text,
                prompt_text=prompt_text,
                prompt_wav=temp_path,
                zero_shot_spk_id=spk_id,
            ):
                audio_chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                audio_tensor = pb_utils.Tensor("audio", audio_chunk)
                sample_rate_tensor = pb_utils.Tensor("sample_rate", np.array([self.sample_rate], dtype=np.int32))
                status_tensor = pb_utils.Tensor("status", np.array(["streaming"], dtype=object))
                voice_ids_tensor = pb_utils.Tensor("voice_ids", np.array([""], dtype=object))
                
                response = pb_utils.InferenceResponse(output_tensors=[
                    audio_tensor,
                    sample_rate_tensor,
                    status_tensor,
                    voice_ids_tensor,
                ])
                response_sender.send(response)
            
            # Clean up temporary speaker (delete from spk2info directly)
            if spk_id in self.cosyvoice.frontend.spk2info:
                del self.cosyvoice.frontend.spk2info[spk_id]
            
            # Send final response (empty audio - all data already streamed)
            audio_tensor = pb_utils.Tensor("audio", np.array([], dtype=np.float32))
            sample_rate_tensor = pb_utils.Tensor("sample_rate", np.array([self.sample_rate], dtype=np.int32))
            status_tensor = pb_utils.Tensor("status", np.array(["complete"], dtype=object))
            voice_ids_tensor = pb_utils.Tensor("voice_ids", np.array([""], dtype=object))
            
            response = pb_utils.InferenceResponse(output_tensors=[
                audio_tensor,
                sample_rate_tensor,
                status_tensor,
                voice_ids_tensor,
            ])
            response_sender.send(response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            
            return None
            
        finally:
            os.unlink(temp_path)
    
    def _handle_cache_voice(self, request):
        """
        Cache a new voice embedding.
        
        Called by Pipecat when a new voice is registered.
        """
        # Get inputs
        voice_id = pb_utils.get_input_tensor_by_name(request, "voice_id")
        voice_id = voice_id.as_numpy()[0].decode('utf-8')
        
        audio_base64 = pb_utils.get_input_tensor_by_name(request, "audio_base64")
        audio_base64 = audio_base64.as_numpy()[0].decode('utf-8')
        
        transcription = pb_utils.get_input_tensor_by_name(request, "transcription")
        transcription = transcription.as_numpy()[0].decode('utf-8')
        
        logger.info(f"Caching voice: {voice_id}")
        
        # Cache the voice
        self._cache_voice_internal(voice_id, audio_base64, transcription)
        
        # Get response sender for decoupled mode
        response_sender = request.get_response_sender()
        
        # Send success response
        audio_tensor = pb_utils.Tensor("audio", np.array([], dtype=np.float32))
        sample_rate_tensor = pb_utils.Tensor("sample_rate", np.array([self.sample_rate], dtype=np.int32))
        status_tensor = pb_utils.Tensor("status", np.array([f"cached:{voice_id}"], dtype=object))
        voice_ids_tensor = pb_utils.Tensor("voice_ids", np.array([], dtype=object))
        
        response = pb_utils.InferenceResponse(output_tensors=[
            audio_tensor,
            sample_rate_tensor,
            status_tensor,
            voice_ids_tensor,
        ])
        response_sender.send(response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        
        return None
    
    def _handle_evict_voice(self, request):
        """
        Remove a voice from the cache.
        
        Called by Pipecat when a voice is deleted.
        """
        voice_id = pb_utils.get_input_tensor_by_name(request, "voice_id")
        voice_id = voice_id.as_numpy()[0].decode('utf-8')
        
        logger.info(f"Evicting voice: {voice_id}")
        
        if voice_id in self.voice_cache:
            spk_id, _ = self.voice_cache[voice_id]  # Unpack tuple (spk_id, transcription)
            try:
                # Delete from spk2info directly (FastCosyVoice3 doesn't have del_zero_shot_spk)
                if spk_id in self.cosyvoice.frontend.spk2info:
                    del self.cosyvoice.frontend.spk2info[spk_id]
            except Exception as e:
                logger.warning(f"Failed to delete speaker {spk_id}: {e}")
            del self.voice_cache[voice_id]
            status = f"evicted:{voice_id}"
        else:
            status = f"not_found:{voice_id}"
        
        # Get response sender for decoupled mode
        response_sender = request.get_response_sender()
        
        audio_tensor = pb_utils.Tensor("audio", np.array([], dtype=np.float32))
        sample_rate_tensor = pb_utils.Tensor("sample_rate", np.array([self.sample_rate], dtype=np.int32))
        status_tensor = pb_utils.Tensor("status", np.array([status], dtype=object))
        voice_ids_tensor = pb_utils.Tensor("voice_ids", np.array([], dtype=object))
        
        response = pb_utils.InferenceResponse(output_tensors=[
            audio_tensor,
            sample_rate_tensor,
            status_tensor,
            voice_ids_tensor,
        ])
        response_sender.send(response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        
        return None
    
    def _handle_list_voices(self, request):
        """List all cached voice IDs."""
        voice_ids = list(self.voice_cache.keys())
        
        logger.info(f"Listing voices: {len(voice_ids)} cached")
        
        # Get response sender for decoupled mode
        response_sender = request.get_response_sender()
        
        audio_tensor = pb_utils.Tensor("audio", np.array([], dtype=np.float32))
        sample_rate_tensor = pb_utils.Tensor("sample_rate", np.array([self.sample_rate], dtype=np.int32))
        status_tensor = pb_utils.Tensor("status", np.array(["success"], dtype=object))
        voice_ids_tensor = pb_utils.Tensor("voice_ids", np.array(voice_ids if voice_ids else [""], dtype=object))
        
        response = pb_utils.InferenceResponse(output_tensors=[
            audio_tensor,
            sample_rate_tensor,
            status_tensor,
            voice_ids_tensor,
        ])
        response_sender.send(response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        
        return None
    
    def _create_error_response(self, error_message: str):
        """Create an error response."""
        audio_tensor = pb_utils.Tensor("audio", np.array([], dtype=np.float32))
        sample_rate_tensor = pb_utils.Tensor("sample_rate", np.array([0], dtype=np.int32))
        status_tensor = pb_utils.Tensor("status", np.array([f"error:{error_message}"], dtype=object))
        voice_ids_tensor = pb_utils.Tensor("voice_ids", np.array([""], dtype=object))
        
        return pb_utils.InferenceResponse(output_tensors=[
            audio_tensor,
            sample_rate_tensor,
            status_tensor,
            voice_ids_tensor,
        ])
    
    def finalize(self):
        """Cleanup when model is unloaded."""
        logger.info("CosyVoice model finalizing...")
        
        # Clear voice cache
        self.voice_cache.clear()
        
        # Clean up model
        if hasattr(self, 'cosyvoice'):
            del self.cosyvoice
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("CosyVoice model finalized")

import os
import tempfile
from typing import Dict, Any
import ray
from ray import serve
from starlette.requests import Request
import whisperx
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@serve.deployment(
    ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0},
    max_concurrent_queries=1  # Limit concurrent requests to avoid OOM
)
class WhisperXService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        logger.info(f"Initializing WhisperX on device: {self.device}")
        
        # Load the model
        self.model = whisperx.load_model(
            "large-v2", 
            self.device, 
            compute_type=self.compute_type
        )
        
        # Load alignment model (for better timestamps)
        self.align_model = None
        self.align_metadata = None
        
        logger.info("WhisperX model loaded successfully")

    def load_alignment_model(self, language_code: str):
        """Load alignment model for specific language if not already loaded"""
        if self.align_model is None or getattr(self, 'current_lang', None) != language_code:
            try:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language_code, 
                    device=self.device
                )
                self.current_lang = language_code
                logger.info(f"Alignment model loaded for language: {language_code}")
            except Exception as e:
                logger.warning(f"Could not load alignment model for {language_code}: {e}")
                self.align_model = None

    def health_check(self):
        """Health check endpoint"""
        return {"status": "healthy", "device": self.device}

    def transcribe_audio_data(self, audio_data: bytes, file_extension: str = ".wav", language: str = None) -> Dict[str, Any]:
        """Transcribe audio from raw bytes"""
        try:
            # Save to temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(audio_data)
                temp_audio_path = tmp_file.name
            
            # Load audio
            logger.info(f"Loading audio from: {temp_audio_path}")
            audio = whisperx.load_audio(temp_audio_path)
            
            # Transcribe
            logger.info("Starting transcription...")
            result = self.model.transcribe(
                audio, 
                batch_size=self.batch_size,
                language=language
            )
            
            # Detect language if not provided
            detected_language = result.get("language", "en")
            
            # Align if possible
            if result["segments"]:
                self.load_alignment_model(detected_language)
                if self.align_model:
                    logger.info("Performing alignment...")
                    result = whisperx.align(
                        result["segments"], 
                        self.align_model, 
                        self.align_metadata, 
                        audio, 
                        self.device, 
                        return_char_alignments=False
                    )
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            return {
                "transcription": result,
                "language": detected_language,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def __call__(self, request) -> Dict[str, Any]:
        # Handle different request types based on HTTP method or content
        if hasattr(request, 'method'):
            # HTTP request from FastAPI integration
            if request.method == "GET" and request.url.path.endswith("/health"):
                return self.health_check()
            elif request.method == "POST" and request.url.path.endswith("/transcribe"):
                # Handle multipart form data
                form = await request.form()
                audio_file = form.get("audio_file")
                language = form.get("language")
                
                if not audio_file:
                    return {"error": "No audio file provided", "status": "failed"}
                
                # Validate file type
                filename = getattr(audio_file, 'filename', '') or ''
                file_extension = os.path.splitext(filename)[1].lower()
                
                if file_extension not in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                    return {
                        "error": "Unsupported file format. Supported formats: .mp3, .wav, .m4a, .flac, .ogg", 
                        "status": "failed"
                    }
                
                # Read audio data
                audio_data = await audio_file.read()
                
                return self.transcribe_audio_data(audio_data, file_extension, language)
        else:
            # Direct Ray Serve call (legacy support)
            audio_data = request.get("audio_data")
            file_extension = request.get("file_extension", ".wav")
            language = request.get("language", None)
            
            if not audio_data:
                return {"error": "No audio data provided"}
            
            return self.transcribe_audio_data(audio_data, file_extension, language)

def main():
    # Initialize Ray
    ray.init()
    
    # Start Ray Serve with HTTP options
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    
    # Deploy the WhisperX service with route prefix
    serve.run(
        WhisperXService.bind(), 
        name="WhisperXService",
        route_prefix="/"
    )
    
    logger.info("WhisperX Ray Serve deployment started on port 8000")
    logger.info("Available endpoints:")
    logger.info("- Health check: GET http://localhost:8000/health")
    logger.info("- Transcribe: POST http://localhost:8000/transcribe")
    
    # Keep the main process running
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        serve.shutdown()
        ray.shutdown()

if __name__ == "__main__":
    main()
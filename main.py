import os
import tempfile
from typing import Dict, Any
import ray
from ray import serve
import whisperx
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@serve.deployment(
    ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0}
)
class WhisperXService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        logger.info(f"Initializing WhisperX on device: {self.device}")
        
        # Load the model
        self.model = whisperx.load_model(
            "tiny", 
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

    async def __call__(self, request) -> Dict[str, Any]:
        try:
            # Get audio file from request
            audio_data = request.get("audio_data")
            file_extension = request.get("file_extension", ".wav")
            language = request.get("language", None)  # Optional language hint
            
            if not audio_data:
                return {"error": "No audio data provided"}
            
            # Handle audio data (bytes from uploaded file)
            audio_bytes = audio_data
            
            # Save to temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(audio_bytes)
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

# FastAPI app for health checks and simple interface
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="WhisperX Ray Serve API")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: str = Form(None)
):
    try:
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith(('audio/', 'application/')):
            return JSONResponse(
                content={"error": "Invalid file type. Please upload .mp3 or .wav files", "status": "failed"}, 
                status_code=400
            )
        
        # Get file extension
        filename = audio_file.filename or ""
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension not in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            return JSONResponse(
                content={"error": "Unsupported file format. Supported formats: .mp3, .wav, .m4a, .flac, .ogg", "status": "failed"}, 
                status_code=400
            )
        
        # Read audio file
        audio_data = await audio_file.read()
        
        # Get Ray Serve handle
        handle = serve.get_deployment("WhisperXService").get_handle()
        
        # Make request
        result = await handle.remote({
            "audio_data": audio_data,
            "file_extension": file_extension,
            "language": language
        })
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "status": "failed"}, 
            status_code=500
        )

def main():
    # Initialize Ray
    ray.init()
    
    # Deploy the WhisperX service 
    serve.start(detached=True)
    serve.run(WhisperXService.bind(), name="WhisperXService")
    
    logger.info("WhisperX Ray Serve deployment started")
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
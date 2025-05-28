# main.py
import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import os
import tempfile
import uvicorn
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray
ray.init(dashboard_host="0.0.0.0")

# Ray Serve deployment
@serve.deployment(ray_actor_options={"num_gpus": 0.5}, num_replicas=1)
class WhisperXModel:
    def __init__(self):
        import whisperx
        self.device = "cuda"
        logger.info("Loading WhisperX model...")
        self.model = whisperx.load_model("small", device=self.device, compute_type="float16")
        self.align_model = None
        logger.info("WhisperX model loaded successfully")

    def transcribe_and_align(self, audio_path: str) -> Dict[str, Any]:
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            
            # Transcribe
            result = self.model.transcribe(audio_path)
            
            if not result.get("segments"):
                return {"error": "No speech detected in audio"}
            
            # Load alignment model if not already loaded
            if self.align_model is None:
                import whisperx
                language_code = result.get("language", "en")
                logger.info(f"Loading alignment model for language: {language_code}")
                self.align_model = whisperx.load_align_model(
                    language_code=language_code, 
                    device=self.device
                )
            
            # Align the transcription
            result_aligned = whisperx.align(
                result["segments"], 
                self.model, 
                self.align_model, 
                audio_path,
                self.device
            )
            
            # Extract text and word timestamps from all segments
            full_text = " ".join([segment["text"] for segment in result_aligned["segments"]])
            all_words = []
            for segment in result_aligned["segments"]:
                if "words" in segment:
                    all_words.extend(segment["words"])
            
            return {
                "text": full_text,
                "language": result.get("language"),
                "segments": result_aligned["segments"],
                "word_timestamps": all_words
            }
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return {"error": f"Transcription failed: {str(e)}"}

    def __call__(self, audio_path: str) -> Dict[str, Any]:
        return self.transcribe_and_align(audio_path)

# Create and deploy the Ray Serve application
whisperx_handle = WhisperXModel.bind()

# Start Ray Serve
serve.start(http_options={"host": "0.0.0.0", "port": 8001})  # Ray Serve on port 8001
serve.run(whisperx_handle, name="whisperx-service", route_prefix="/")

# Create FastAPI app
app = FastAPI(
    title="WhisperX Transcription Service",
    description="Audio transcription service using WhisperX with word-level timestamps",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "WhisperX Transcription Service is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "whisperx-transcription"}

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Transcribe an audio file and return text with word-level timestamps
    
    Args:
        audio_file: Audio file (WAV, MP3, M4A, etc.)
        
    Returns:
        JSON with transcribed text, language, and word timestamps
    """
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Check file size (limit to 100MB)
    if audio_file.size and audio_file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 100MB")
    
    # Create temporary file
    temp_file = None
    try:
        # Create temporary file with proper extension
        file_extension = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        # Write uploaded content to temporary file
        content = await audio_file.read()
        temp_file.write(content)
        temp_file.close()
        
        logger.info(f"Processing file: {audio_file.filename} ({len(content)} bytes)")
        
        # Call Ray Serve deployment
        result = await whisperx_handle.remote(temp_file.name)
        
        # Check for errors in result
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content={
            "filename": audio_file.filename,
            "transcription": result
        })
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")

@app.post("/transcribe-url")
async def transcribe_from_url(audio_url: str):
    """
    Transcribe an audio file from a URL
    
    Args:
        audio_url: URL to audio file
        
    Returns:
        JSON with transcribed text and word timestamps
    """
    import requests
    
    try:
        # Download file from URL
        response = requests.get(audio_url, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(response.content)
        temp_file.close()
        
        logger.info(f"Processing URL: {audio_url}")
        
        # Call Ray Serve deployment
        result = await whisperx_handle.remote(temp_file.name)
        
        # Check for errors in result
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content={
            "url": audio_url,
            "transcription": result
        })
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing audio from URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    finally:
        # Clean up temporary file
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")

# Graceful shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down WhisperX service...")
    serve.shutdown()
    ray.shutdown()

if __name__ == "__main__":
    logger.info("Starting WhisperX Transcription Service...")
    logger.info("Ray Dashboard available at: http://0.0.0.0:8265")
    logger.info("API Documentation available at: http://0.0.0.0:8000/docs")
    
    # Run FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
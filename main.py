# File name: serve_quickstart.py
from starlette.requests import Request

import ray
from ray import serve
import torch 
import whisperx
import io
import tempfile
import os
import time
from whisperx.utils import get_writer, WriteSRT
from starlette.responses import FileResponse, JSONResponse
import base64

print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")


@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 0.5})
class Transcriber:
    def __init__(self):
        # Load WhisperX model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisperx.load_model("large-v2", self.device, compute_type="float16" if self.device == "cuda" else "int8")

    def transcribe(self, audio_tensor) -> str:
        # Transcribe loaded audio
        result = self.model.transcribe(audio_tensor, batch_size=16)
        return result # before alignment

    async def __call__(self, http_request: Request) -> str:
        form = await http_request.form()

        if "file" not in form:
            return {"error": "No file part in the form"}

        audio_file = form["file"]
        print(f"File '{audio_file.filename}' uploaded.")
        audio_bytes = await audio_file.read()
        print(f"Read {len(audio_bytes)} bytes from the uploaded file.")

        # Extract file extension, default to .wav if none found
        ext = os.path.splitext(audio_file.filename)[1].lower() or ".wav"

        # Use a temporary file with the same extension as the input file
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            audio_path = tmp.name

            audio_tensor = whisperx.load_audio(tmp.name)
            start = time.time()
            print(f"Starting transcription for '{audio_file.filename}'...")
            result = self.transcribe(audio_tensor)
            print(f"Transcription completed for '{audio_file.filename}'.")
            end = time.time()
            print(f"Transcription time: {end - start:.2f} seconds")

            language = result.get("language", "en")


            # 2. Align whisper output
            start = time.time()
            print(f"Aligning transcription for '{audio_file.filename}'...")
            model_a, metadata = whisperx.load_align_model(language_code=language, device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio_tensor, self.device, return_char_alignments=False)
            print(f"Alignment completed for '{audio_file.filename}'.")
            end = time.time()
            print(f"Alignment time: {end - start:.2f} seconds")
            result['language'] = language

            return result

            
transcriber_app = Transcriber.bind()


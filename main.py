# File name: serve_quickstart.py
from starlette.requests import Request

import ray
from ray import serve
import torch 
import whisperx
import io
import tempfile
import os

ray.init()

serve.start(detached=False, http_options={"host": "0.0.0.0", "port": 8000})

@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class Transcriber:
    def __init__(self):
        # Load WhisperX model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisperx.load_model("tiny", self.device, compute_type="float16" if self.device == "cuda" else "int8")

    def transcribe(self, audio_tensor) -> str:
        # Transcribe loaded audio
        result = self.model.transcribe(audio_tensor)
        return result["segments"] # before alignment

    async def __call__(self, http_request: Request) -> str:
        form = await http_request.form()

        if "file" not in form:
            return {"error": "No file part in the form"}

        audio_file = form["file"]
        audio_bytes = await audio_file.read()

        # Extract file extension, default to .wav if none found
        ext = os.path.splitext(audio_file.filename)[1].lower() or ".wav"

        # Use a temporary file with the same extension as the input file
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            audio_tensor = whisperx.load_audio(tmp.name)

        result = self.transcribe(audio_tensor)

        return {"text": result}

transcriber_app = Transcriber.bind()

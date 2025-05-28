# serve_frontend.py
import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File
import asyncio

ray.init()
serve.start(detached=False)

# This is the Ray Serve deployment
@serve.deployment(ray_actor_options={"num_gpus": 0.5}, num_replicas=1)
class WhisperXModel:
    def __init__(self):
        import whisperx
        self.device = "cuda"
        self.model = whisperx.load_model("small", device=self.device, compute_type="float16")
        self.model_aligner = None

    def transcribe_and_align(self, audio_path):
        result = self.model.transcribe(audio_path)
        if self.model_aligner is None:
            import whisperx
            self.model_aligner = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result_aligned = whisperx.align(result["segments"], self.model.model_aud, self.model_aligner, self.device)
        return {
            "text": result_aligned["segments"][0]["text"],
            "word_timestamps": result_aligned["segments"][0]["words"]
        }

    def __call__(self, audio_path: str):
        return self.transcribe_and_align(audio_path)

WhisperXModel.deploy()

# FastAPI frontend
app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    handle = serve.get_deployment("WhisperXModel").get_handle()
    result = await handle.remote(file_path)
    return result

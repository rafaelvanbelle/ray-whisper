import ray
from ray import serve
import whisperx

ray.init()
serve.start(detached=True, http_options={"host": "0.0.0.0"})


@serve.deployment(ray_actor_options={'num_gpus':0.5}, num_replicas=1, name='whisperx')
class WhisperXModel:
    def __init__(self):
        # Adjust model size and compute type as needed
        self.device = "cuda"
        self.model = whisperx.load_model("small", device=self.device, compute_type="float16")

        # Optional: align model for word-level timestamps
        self.model_aligner = None

    def __call__(self, request):
        # You would stream/upload audio to disk or memory; here's a placeholder
        audio_file = "bdw_clip.mp3"  # Path to audio file

        result = self.model.transcribe(audio_file)

        if self.model_aligner is None:
            self.model_aligner = whisperx.load_align_model(language_code=result["language"], device=self.device)

        result_aligned = whisperx.align(result["segments"], self.model.model_aud, self.model_aligner, self.device)

        return {
            "text": result_aligned["segments"][0]["text"],
            "word_timestamps": result_aligned["segments"][0]["words"]
        }

serve.run(WhisperXModel.bind())
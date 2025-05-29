import gradio as gr
import httpx
import asyncio
import base64

async def transcribe(audio):
    timeout = httpx.Timeout(connect=60.0, read=None, write=None, timeout=9999)  # Increase both read and connect timeouts
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(audio, "rb") as f:
            files = {"file": (audio, f, "audio/wav")}
            response = await client.post("http://localhost:8000", files=files)
            data = response.json()
            formatted_text = data.get("formatted_text", "")
            srt_b64 = data.get("srt_b64", "")
            srt_filename = data.get("srt_filename", "output.srt")
            # Save SRT file locally for download
            srt_path = f"/tmp/{srt_filename}"
            with open(srt_path, "wb") as srt_file:
                srt_file.write(base64.b64decode(srt_b64))
            return formatted_text, srt_path

iface = gr.Interface(fn=transcribe, inputs=gr.Audio(type="filepath"), outputs=[
        gr.Textbox(label="Transcription"),
        gr.File(label="Download SRT")
    ], concurrency_limit=5)

if __name__ == "__main__":
    iface.launch()

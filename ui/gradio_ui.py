import gradio as gr
import httpx
import asyncio

async def transcribe(audio):
    timeout = httpx.Timeout(connect=60.0, read=None, write=None, timeout=9999)  # Increase both read and connect timeouts
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(audio, "rb") as f:
            files = {"file": (audio, f, "audio/wav")}
            response = await client.post("http://localhost:8000", files=files)
            return response.text

iface = gr.Interface(fn=transcribe, inputs=gr.Audio(type="filepath"), outputs="text", concurrency_limit=5)

if __name__ == "__main__":
    iface.launch()

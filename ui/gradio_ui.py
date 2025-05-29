import gradio as gr
import httpx
import asyncio
import base64
import time
import os
import whisperx

async def transcribe(audio):
    timeout = httpx.Timeout(connect=60.0, read=None, write=None, timeout=9999)  # Increase both read and connect timeouts
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(audio, "rb") as f:
            files = {"file": (audio, f, "audio/wav")}
            response = await client.post("http://localhost:8000", files=files)
            
            formatted_text = format_segments(response)
            srt_path = write_srt(response, audio, language=response['language'], output_dir="/tmp", extension="srt")
            
            
            return formatted_text, srt_path

iface = gr.Interface(fn=transcribe, inputs=gr.Audio(type="filepath"), outputs=[
        gr.Textbox(label="Transcription"),
        gr.File(label="Download SRT")
    ], concurrency_limit=5)


def format_segments(result):
# Format segments with timestamps
    formatted_text = ""
    for seg in result["segments"]:
        start_ts = time.strftime('%H:%M:%S', time.gmtime(seg['start']))
        end_ts = time.strftime('%H:%M:%S', time.gmtime(seg['end']))
        formatted_text += f"[{start_ts} --> {end_ts}] {seg['text'].strip()}\n"

    return formatted_text


def write_srt(result, audio, language, output_dir="/tmp", extension="srt"):

    writer = whisperx.utils.get_writer(extension, output_dir=output_dir)
    writer(result, audio, options={'max_line_width':None, 'max_line_count':None, 'highlight_words':None})

    audio_basename = os.path.basename(audio_path)
    audio_basename = os.path.splitext(audio_basename)[0]
    output_path = os.path.join(
            output_dir, audio_basename + "." + extension
        )
    
    return output_path

if __name__ == "__main__":
    iface.launch()

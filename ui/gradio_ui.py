import gradio as gr
import requests

def transcribe(audio):
    with open(audio, "rb") as f:
        response = requests.post("http://localhost:8000/transcribe", files={"audio_file": f})
        return response.text

iface = gr.Interface(fn=transcribe, inputs=gr.Audio(type="filepath"), outputs="text")

if __name__ == "__main__":
    iface.launch()

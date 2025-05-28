import gradio as gr
import requests

def transcribe(audio):
    with open(audio, "rb") as f:
        response = requests.post("http://127.0.0.1:8000/", files={"file": f})
        return response.text

iface = gr.Interface(fn=transcribe, inputs=gr.Audio(type="filepath"), outputs="text")

if __name__ == "__main__":
    iface.launch()

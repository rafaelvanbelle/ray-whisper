import gradio as gr
import httpx
import asyncio
import time
import os

async def transcribe_with_progress(audio):
    if not audio:
        yield "❌ No audio file provided"
        return
    
    try:
        # Show initial status
        file_size = os.path.getsize(audio) / (1024 * 1024)  # MB
        yield f"📁 File size: {file_size:.1f} MB"
        
        yield "📤 Uploading file to server..."
        
        # Set timeouts
        timeout = httpx.Timeout(
            connect=30.0,
            read=600.0,  # 10 minutes
            write=60.0,
            pool=None
        )
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            with open(audio, "rb") as f:
                files = {"file": (audio, f, "audio/wav")}
                
                yield "⏳ Processing audio with WhisperX..."
                yield "🔄 This may take several minutes for large files..."
                
                # Make the request
                response = await client.post("http://localhost:8000", files=files)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        text = result.get("text", str(result))
                        processing_time = result.get("processing_time", total_time)
                        
                        yield f"✅ Transcription completed in {processing_time:.1f} seconds!\n\n{text}"
                    except:
                        yield f"✅ Transcription completed in {total_time:.1f} seconds!\n\n{response.text}"
                else:
                    yield f"❌ Server error ({response.status_code}): {response.text}"
                    
    except httpx.TimeoutException:
        yield "⏰ Request timed out. Your file might be too large or the server is very busy."
    except Exception as e:
        yield f"❌ Error: {str(e)}"

# Alternative version with periodic updates during processing
async def transcribe_with_periodic_updates(audio):
    if not audio:
        yield "❌ No audio file provided"
        return
    
    try:
        file_size = os.path.getsize(audio) / (1024 * 1024)
        yield f"📁 Processing {file_size:.1f} MB file..."
        
        timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=None)
        
        # Start the transcription request in the background
        async def make_request():
            async with httpx.AsyncClient(timeout=timeout) as client:
                with open(audio, "rb") as f:
                    files = {"file": (audio, f, "audio/wav")}
                    return await client.post("http://localhost:8000", files=files)
        
        # Create the request task
        request_task = asyncio.create_task(make_request())
        
        # Show periodic updates while waiting
        start_time = time.time()
        update_interval = 10  # seconds
        
        while not request_task.done():
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            if minutes > 0:
                yield f"⏳ Still processing... {minutes}m {seconds}s elapsed"
            else:
                yield f"⏳ Processing... {seconds}s elapsed"
            
            # Wait for next update or completion
            try:
                await asyncio.wait_for(asyncio.shield(request_task), timeout=update_interval)
                break  # Request completed
            except asyncio.TimeoutError:
                continue  # Keep waiting and update progress
        
        # Get the result
        response = await request_task
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            try:
                result = response.json()
                text = result.get("text", str(result))
                processing_time = result.get("processing_time", total_time)
                
                yield f"✅ Completed in {processing_time:.1f}s (total: {total_time:.1f}s)\n\n{text}"
            except:
                yield f"✅ Completed in {total_time:.1f}s\n\n{response.text}"
        else:
            yield f"❌ Error ({response.status_code}): {response.text}"
            
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def create_interface():
    with gr.Blocks(title="Audio Transcription with Progress") as demo:
        gr.Markdown("# 🎵 Audio Transcription Service")
        gr.Markdown("Upload an audio file to get real-time transcription progress updates.")
        
        with gr.Tab("📊 With Progress Updates"):
            audio_input = gr.Audio(
                label="Upload Audio File", 
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            transcribe_btn = gr.Button("Start Transcription", variant="primary")
            
            output_text = gr.Textbox(
                label="Transcription Progress & Result",
                lines=15,
                max_lines=25,
                show_copy_button=True,
                interactive=False
            )
            
            # Use the periodic updates version for better UX
            transcribe_btn.click(
                fn=transcribe_with_periodic_updates,
                inputs=[audio_input],
                outputs=[output_text],
                show_progress=False  # We're handling progress ourselves
            )
        
        with gr.Tab("⚡ Simple Version"):
            audio_input_simple = gr.Audio(
                label="Upload Audio File", 
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            transcribe_btn_simple = gr.Button("Transcribe", variant="secondary")
            
            output_simple = gr.Textbox(
                label="Result",
                lines=10,
                show_copy_button=True
            )
            
            transcribe_btn_simple.click(
                fn=transcribe_with_progress,
                inputs=[audio_input_simple],
                outputs=[output_simple]
            )
        
        with gr.Tab("ℹ️ Tips"):
            gr.Markdown("""
            ## Progress Features:
            
            - **File size display** - Shows upload size
            - **Real-time status** - Updates every 10 seconds during processing
            - **Elapsed time tracking** - Shows how long transcription is taking
            - **Processing time** - Shows server-side processing duration
            
            ## Expected Processing Times:
            - **Small files (< 5 MB)**: 10-30 seconds
            - **Medium files (5-25 MB)**: 1-3 minutes  
            - **Large files (25+ MB)**: 3-10+ minutes
            
            ## Status Icons:
            - 📁 File info
            - 📤 Uploading
            - ⏳ Processing  
            - 🔄 Long operation in progress
            - ✅ Success
            - ❌ Error
            - ⏰ Timeout
            """)
    
    return demo

if __name__ == "__main__":
    iface = create_interface()
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
import PyPDF2
import pdfplumber
import os
from typing import Optional, Tuple
import json
from datetime import datetime
from pathlib import Path
import requests
import re


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    print(f"Loading model on device: {DEVICE}")
    try:
        model = ChatterboxTTS.from_pretrained(DEVICE)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def extract_text_from_pdf(pdf_path: str, method: str = "pdfplumber") -> str:
    """Extract text from PDF using either PyPDF2 or pdfplumber."""
    text = ""
    
    try:
        if method == "pdfplumber":
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        else:  # PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
    
    return text.strip()


def process_text_for_tts(text: str, max_chars: int = 300) -> list[str]:
    """Split text into chunks suitable for TTS processing."""
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Split into sentences (simple approach)
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group sentences into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def fix_grammar_with_ollama(text: str) -> str:
    """Fix grammar and spacing issues using Ollama (qwen3)."""
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            return fix_grammar_basic(text)
    except:
        return fix_grammar_basic(text)
    
    # Use Ollama with qwen3 for grammar correction
    prompt = f"""Fix the grammar, punctuation, and spacing in this text. Return ONLY the corrected text with no other words, explanations, or thinking:

{text}

Corrected text:"""
    
    try:
        data = {
            "model": "qwen3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": len(text) + 200
            }
        }
        
        response = requests.post("http://localhost:11434/api/generate", 
                               json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            corrected_text = result.get("response", "").strip()
            
            # Clean up any unwanted additions
            if corrected_text and len(corrected_text) > 0:
                # Remove any thinking tags or unwanted text
                corrected_text = re.sub(r'<think>.*?</think>', '', corrected_text, flags=re.DOTALL)
                corrected_text = re.sub(r'^.*?Corrected text:\s*', '', corrected_text, flags=re.IGNORECASE)
                corrected_text = corrected_text.strip()
                if corrected_text:
                    return corrected_text
        
    except Exception as e:
        print(f"Ollama grammar correction failed: {e}")
    
    # Fallback to basic grammar fixes
    return fix_grammar_basic(text)


def fix_grammar_basic(text: str) -> str:
    """Basic grammar and spacing fixes for extracted text."""
    if not text:
        return text
    
    # Fix common spacing issues from PDF extraction
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)     # Add space between letters and numbers
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)     # Add space between numbers and letters
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation
    text = re.sub(r'([,;:])([a-zA-Z])', r'\1 \2', text) # Add space after punctuation
    
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common word concatenations
    text = text.replace('isabout', 'is about')
    text = text.replace('givingup', 'giving up')
    text = text.replace('eviland', 'evil and')
    text = text.replace('practising', 'practicing')
    text = text.replace('andgoodness', 'and goodness')
    text = text.replace('isestablished', 'is established')
    text = text.replace('wemust', 'we must')
    text = text.replace('letgo', 'let go')
    text = text.replace('ofboth', 'of both')
    text = text.replace('goodand', 'good and')
    text = text.replace('Wehave', 'We have')
    text = text.replace('alreadyhear', 'already hear')
    text = text.replace('denough', 'd enough')
    text = text.replace('aboutwholesome', 'about wholesome')
    text = text.replace('andunwholesome', 'and unwholesome')
    text = text.replace('conditionsto', 'conditions to')
    text = text.replace('understandsomething', 'understand something')
    text = text.replace('aboutthem', 'about them')
    
    return text.strip()


def extract_pdf_text(
    pdf_file,
    extraction_method
):
    """Extract text from PDF and return it for editing."""
    if pdf_file is None:
        return "Please upload a PDF file", ""
    
    extracted_text = extract_text_from_pdf(pdf_file, extraction_method)
    
    if extracted_text.startswith("Error"):
        return extracted_text, ""
    
    # Process text into chunks for display
    text_chunks = process_text_for_tts(extracted_text)
    
    if not text_chunks:
        return "No text could be extracted from the PDF", ""
    
    # Format extracted text for display and editing
    display_text = "=== EXTRACTED TEXT (Edit below before generating audio) ===\n\n"
    full_text = ""
    for i, chunk in enumerate(text_chunks):
        display_text += f"Chunk {i+1}: {chunk}\n\n"
        full_text += chunk + " "
    
    return f"✅ Extracted {len(extracted_text)} characters from PDF", extracted_text


def fix_extracted_text(raw_text: str):
    """Fix grammar and spacing issues in extracted text."""
    if not raw_text or raw_text.startswith("Please upload") or raw_text.startswith("Error") or raw_text.startswith("✅"):
        return "Please extract text from a PDF first", ""
    
    print("🔧 Fixing grammar and spacing with Ollama...")
    fixed_text = fix_grammar_with_ollama(raw_text)
    
    status = f"✅ Grammar corrected - {len(fixed_text)} characters"
    
    return status, fixed_text


def split_text_for_tts(text: str, max_chunk_size: int = 250) -> list[str]:
    """Split text into chunks suitable for TTS while preserving sentence boundaries."""
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_chunk_size]]


def generate_audio_from_text(
    model,
    text_to_synthesize,
    audio_prompt_path,
    exaggeration,
    temperature,
    seed_num,
    cfgw,
    min_p,
    top_p,
    repetition_penalty
):
    """Generate audio from the provided text, handling long texts by chunking."""
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)
    
    if not text_to_synthesize or text_to_synthesize.strip() == "":
        return None, "Please provide text to synthesize", ""
    
    if seed_num != 0:
        set_seed(int(seed_num))
    
    try:
        # Split text into manageable chunks
        text_chunks = split_text_for_tts(text_to_synthesize, max_chunk_size=250)
        
        print(f"🔄 Processing {len(text_chunks)} text chunks...")
        
        # Generate audio for each chunk
        audio_segments = []
        total_chars = len(text_to_synthesize)
        
        for i, chunk in enumerate(text_chunks):
            print(f"🎙️  Generating audio for chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)")
            
            wav = model.generate(
                chunk,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfgw,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            
            # Add a small pause between chunks (0.3 seconds of silence)
            pause_samples = int(0.3 * model.sr)
            pause = np.zeros((1, pause_samples), dtype=np.float32)
            
            audio_segments.append(wav.squeeze(0).numpy())
            if i < len(text_chunks) - 1:  # Don't add pause after last chunk
                audio_segments.append(pause.squeeze(0))
        
        # Concatenate all audio segments
        final_audio = np.concatenate(audio_segments)
        
        status_msg = f"✅ Generated audio for {total_chars} characters in {len(text_chunks)} chunks"
        return (model.sr, final_audio), status_msg, text_to_synthesize
    
    except Exception as e:
        return None, f"❌ Error generating audio: {str(e)}", ""


def generate_from_text(
    model,
    text,
    audio_prompt_path,
    exaggeration,
    temperature,
    seed_num,
    cfgw,
    min_p,
    top_p,
    repetition_penalty
):
    """Generate audio from text with automatic chunking for longer texts."""
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    try:
        # For texts longer than 250 chars, use chunking
        if len(text) > 250:
            text_chunks = split_text_for_tts(text, max_chunk_size=250)
            print(f"🔄 Processing {len(text_chunks)} text chunks for direct TTS...")
            
            audio_segments = []
            for i, chunk in enumerate(text_chunks):
                print(f"🎙️  Generating chunk {i+1}/{len(text_chunks)}")
                
                wav = model.generate(
                    chunk,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfgw,
                    min_p=min_p,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                
                # Add small pause between chunks
                pause_samples = int(0.3 * model.sr)
                pause = np.zeros((1, pause_samples), dtype=np.float32)
                
                audio_segments.append(wav.squeeze(0).numpy())
                if i < len(text_chunks) - 1:
                    audio_segments.append(pause.squeeze(0))
            
            # Concatenate all segments
            final_audio = np.concatenate(audio_segments)
            return (model.sr, final_audio)
        else:
            # For short texts, generate directly
            wav = model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfgw,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            return (model.sr, wav.squeeze(0).numpy())
    
    except Exception as e:
        print(f"Error in direct TTS generation: {e}")
        # Fallback to basic generation with truncated text
        truncated_text = text[:250] if len(text) > 250 else text
        wav = model.generate(truncated_text, audio_prompt_path=audio_prompt_path)
        return (model.sr, wav.squeeze(0).numpy())


def save_text_to_file(text: str, filename: str = None, metadata: dict = None) -> str:
    """Save text content to a JSON file with metadata."""
    saved_texts_dir = Path("saved_texts")
    saved_texts_dir.mkdir(exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"text_{timestamp}.json"
    
    file_path = saved_texts_dir / filename
    
    save_data = {
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        return f"Text saved successfully to {file_path}"
    except Exception as e:
        return f"Error saving text: {str(e)}"


def load_text_from_file(file_path: str) -> Tuple[str, str]:
    """Load text content from a saved JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text = data.get("text", "")
        metadata = data.get("metadata", {})
        timestamp = data.get("timestamp", "Unknown")
        
        info = f"Loaded text from {timestamp}"
        if metadata:
            info += f"\nMetadata: {json.dumps(metadata, indent=2)}"
        
        return text, info
    except Exception as e:
        return "", f"Error loading text: {str(e)}"


def get_saved_text_files() -> list:
    """Get list of saved text files."""
    saved_texts_dir = Path("saved_texts")
    if not saved_texts_dir.exists():
        return []
    
    files = []
    for file_path in saved_texts_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            timestamp = data.get("timestamp", "Unknown")
            metadata = data.get("metadata", {})
            files.append({
                "path": str(file_path),
                "name": file_path.name,
                "timestamp": timestamp,
                "source": metadata.get("source", "Unknown")
            })
        except:
            continue
    
    return sorted(files, key=lambda x: x["timestamp"], reverse=True)


with gr.Blocks(title="PDF to Audio Converter") as demo:
    model_state = gr.State(None)
    
    gr.Markdown("# PDF to Audio Converter")
    gr.Markdown("Convert PDF documents to speech using Chatterbox TTS")
    
    with gr.Tabs():
        with gr.TabItem("PDF to Audio"):
            with gr.Row():
                with gr.Column():
                    pdf_file = gr.File(
                        label="Upload PDF File",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    
                    extraction_method = gr.Radio(
                        choices=["pdfplumber", "PyPDF2"],
                        value="pdfplumber",
                        label="PDF Extraction Method"
                    )
                    
                    gr.Markdown("**Step 1:** Upload PDF and click 'Extract PDF Text'")
                    gr.Markdown("**Step 2:** Click 'Fix Grammar' to correct spacing and grammar issues")
                    gr.Markdown("**Step 3:** Edit the corrected text below as needed")
                    gr.Markdown("**Step 4:** Click 'Generate Audio from Edited Text'")
                    
                    extract_btn = gr.Button("📄 Extract PDF Text", variant="secondary")
                    fix_grammar_btn = gr.Button("🔧 Fix Grammar with Ollama", variant="secondary")
                    
                    with gr.Row():
                        save_extracted_btn = gr.Button("💾 Save Extracted Text", size="sm")
                        save_synthesized_btn = gr.Button("💾 Save Synthesized Text", size="sm")
                        load_saved_btn = gr.Button("📁 Load Saved Text", size="sm")
                    
                    ref_wav = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Reference Audio File (for voice cloning)",
                        value=None
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration", value=.5)
                        cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)
                        seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                        temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)
                        min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.05)
                        top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=1.00)
                        repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)
                    
                    convert_btn = gr.Button("🔊 Generate Audio from Edited Text", variant="primary")
                
                with gr.Column():
                    audio_output = gr.Audio(label="Generated Audio")
                    status_output = gr.Textbox(label="Status", lines=2)
                    extracted_text_output = gr.Textbox(
                        label="Raw Extracted Text (from PDF)",
                        lines=6,
                        max_lines=10,
                        interactive=False
                    )
                    
                    grammar_status = gr.Textbox(
                        label="Grammar Correction Status",
                        lines=1,
                        interactive=False
                    )
                    
                    editable_text = gr.Textbox(
                        label="Grammar-Corrected Text (Edit this text before generating audio)",
                        placeholder="Grammar-corrected text will appear here. Edit as needed before generating audio...",
                        lines=8,
                        max_lines=15,
                        interactive=True
                    )
                    save_status = gr.Textbox(label="Save Status", lines=1, visible=False)
                    synthesized_text_store = gr.Textbox(visible=False)
        
        with gr.TabItem("Direct Text to Speech"):
            with gr.Row():
                with gr.Column():
                    direct_text = gr.Textbox(
                        value="Enter your text here for direct text-to-speech conversion. Long texts will be automatically chunked for optimal quality.",
                        label="Text to synthesize (supports long texts with automatic chunking)",
                        max_lines=10
                    )
                    
                    with gr.Row():
                        save_direct_text_btn = gr.Button("💾 Save Text", size="sm")
                        load_direct_text_btn = gr.Button("📁 Load Text", size="sm")
                    direct_ref_wav = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Reference Audio File",
                        value=None
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        direct_exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration", value=.5)
                        direct_cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)
                        direct_seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                        direct_temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)
                        direct_min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.05)
                        direct_top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=1.00)
                        direct_repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)
                    
                    direct_run_btn = gr.Button("Generate Speech", variant="primary")
                
                with gr.Column():
                    direct_audio_output = gr.Audio(label="Output Audio")
                    direct_save_status = gr.Textbox(label="Save Status", lines=1, visible=False)
        
        with gr.TabItem("Saved Texts Manager"):
            gr.Markdown("### Manage Your Saved Texts")
            with gr.Row():
                with gr.Column():
                    refresh_list_btn = gr.Button("🔄 Refresh List")
                    saved_files_display = gr.Dataframe(
                        headers=["File Name", "Timestamp", "Source"],
                        label="Saved Text Files",
                        interactive=False
                    )
                    selected_file_path = gr.Textbox(label="Selected File Path", visible=False)
                    
                    with gr.Row():
                        load_selected_btn = gr.Button("Load Selected", variant="primary")
                        delete_selected_btn = gr.Button("Delete Selected", variant="stop")
                
                with gr.Column():
                    loaded_text_preview = gr.Textbox(
                        label="Text Preview",
                        lines=15,
                        max_lines=30,
                        interactive=False
                    )
                    loaded_text_info = gr.Textbox(label="File Info", lines=3)
    
    # Don't load model on startup - load it when needed
    # demo.load(fn=load_model, inputs=[], outputs=model_state)
    
    # Save and load handlers
    def save_extracted_text(text, pdf_filename=None):
        if not text or text.strip() == "":
            return gr.update(value="No text to save", visible=True)
        
        metadata = {"source": "PDF extraction" if pdf_filename else "Direct input"}
        if pdf_filename:
            metadata["original_pdf"] = os.path.basename(pdf_filename)
        
        result = save_text_to_file(text, metadata=metadata)
        return gr.update(value=result, visible=True)
    
    def load_saved_text():
        files = get_saved_text_files()
        if not files:
            return "", "No saved texts found"
        
        # Load the most recent file
        most_recent = files[0]
        text, info = load_text_from_file(most_recent["path"])
        return text, info
    
    def refresh_saved_files():
        files = get_saved_text_files()
        if not files:
            return gr.update(value=[])
        
        data = [[f["name"], f["timestamp"], f["source"]] for f in files]
        return gr.update(value=data)
    
    def on_file_select(evt: gr.SelectData, files_data):
        if evt.index[0] < len(files_data):
            files = get_saved_text_files()
            if evt.index[0] < len(files):
                return files[evt.index[0]]["path"]
        return ""
    
    def load_selected_file(file_path):
        if not file_path:
            return "", "No file selected"
        text, info = load_text_from_file(file_path)
        return text, info
    
    def delete_selected_file(file_path):
        if not file_path:
            return gr.update(value=[]), "No file selected"
        
        try:
            Path(file_path).unlink()
            files = get_saved_text_files()
            data = [[f["name"], f["timestamp"], f["source"]] for f in files]
            return gr.update(value=data), f"Deleted {os.path.basename(file_path)}"
        except Exception as e:
            return gr.update(), f"Error deleting file: {str(e)}"
    
    save_extracted_btn.click(
        fn=lambda text, pdf_file: save_extracted_text(text, pdf_file),
        inputs=[editable_text, pdf_file],
        outputs=save_status
    )
    
    save_synthesized_btn.click(
        fn=lambda text: save_text_to_file(text, metadata={"source": "Synthesized from PDF"}),
        inputs=synthesized_text_store,
        outputs=save_status
    )
    
    load_saved_btn.click(
        fn=load_saved_text,
        inputs=[],
        outputs=[editable_text, save_status]
    )
    
    refresh_list_btn.click(
        fn=refresh_saved_files,
        inputs=[],
        outputs=saved_files_display
    )
    
    saved_files_display.select(
        fn=on_file_select,
        inputs=saved_files_display,
        outputs=selected_file_path
    )
    
    load_selected_btn.click(
        fn=load_selected_file,
        inputs=selected_file_path,
        outputs=[loaded_text_preview, loaded_text_info]
    )
    
    delete_selected_btn.click(
        fn=delete_selected_file,
        inputs=selected_file_path,
        outputs=[saved_files_display, loaded_text_info]
    )
    
    # Auto-refresh saved files list on tab change - disabled to prevent loading issues
    # demo.load(fn=refresh_saved_files, inputs=[], outputs=saved_files_display)
    
    # Direct text save/load handlers
    save_direct_text_btn.click(
        fn=lambda text: save_text_to_file(text, metadata={"source": "Direct input"}),
        inputs=direct_text,
        outputs=direct_save_status
    )
    
    load_direct_text_btn.click(
        fn=load_saved_text,
        inputs=[],
        outputs=[direct_text, direct_save_status]
    )
    
    # PDF extraction handler - now only outputs to raw extracted text
    extract_btn.click(
        fn=extract_pdf_text,
        inputs=[
            pdf_file,
            extraction_method
        ],
        outputs=[status_output, extracted_text_output]
    )
    
    # Grammar correction handler
    fix_grammar_btn.click(
        fn=fix_extracted_text,
        inputs=[extracted_text_output],
        outputs=[grammar_status, editable_text]
    )
    
    # Audio generation handler
    convert_btn.click(
        fn=generate_audio_from_text,
        inputs=[
            model_state,
            editable_text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=[audio_output, status_output, synthesized_text_store],
    )
    
    # Direct TTS handlers
    direct_run_btn.click(
        fn=generate_from_text,
        inputs=[
            model_state,
            direct_text,
            direct_ref_wav,
            direct_exaggeration,
            direct_temp,
            direct_seed_num,
            direct_cfg_weight,
            direct_min_p,
            direct_top_p,
            direct_repetition_penalty,
        ],
        outputs=direct_audio_output,
    )


if __name__ == "__main__":
    print("Starting PDF-to-Audio app...")
    print("The app will be available at:")
    print("- Local URL: http://localhost:7863")
    print("- Public URL: Will be generated if share=True")
    
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7863,        # Using different port
        share=False,             # Disable public link for now
        show_error=True
    )
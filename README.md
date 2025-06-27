# Gradio PDF to Audio Converter

A powerful web application that converts PDF documents to high-quality speech using ChatterboxTTS with integrated AI-powered grammar correction.

## 🌟 Features

- **PDF Text Extraction**: Support for both pdfplumber and PyPDF2 extraction methods
- **AI Grammar Correction**: Uses Ollama (qwen3) to fix spacing and grammar issues from PDF extraction
- **Voice Cloning**: Generate speech with custom voice prompts
- **Advanced TTS Controls**: Fine-tune temperature, exaggeration, CFG weight, and sampling parameters
- **Text Management**: Save, load, and manage extracted/corrected texts
- **Direct TTS**: Convert text directly to speech without PDF upload
- **Web Interface**: Easy-to-use Gradio interface accessible via web browser

## 🔄 Workflow

1. **Upload PDF** and click "Extract PDF Text"
2. **Click "Fix Grammar with Ollama"** to correct spacing and grammar issues (e.g., "isaboutgivingupevilandpractising" → "is about giving up evil and practicing")
3. **Edit the corrected text** as needed
4. **Generate audio** from the final text

## 🛠️ Requirements

- Python 3.8+
- ChatterboxTTS
- Gradio
- PyPDF2
- pdfplumber
- Ollama (with qwen3 model for grammar correction)

## 📦 Installation

1. Install ChatterboxTTS and dependencies:
```bash
pip install gradio PyPDF2 pdfplumber requests
```

2. Install and set up Ollama with qwen3:
```bash
# Install Ollama (see https://ollama.ai/)
ollama pull qwen3
ollama serve
```

3. Run the application:
```bash
python gradio_pdf_to_audio_app.py
```

4. Open your browser to: `http://localhost:7863`

## 🎯 Use Cases

- **Accessibility**: Convert academic papers, books, and documents to audio
- **Learning**: Listen to educational content while multitasking
- **Content Creation**: Generate audiobooks from text documents
- **Language Learning**: Practice pronunciation with voice cloning

## 🔧 Advanced Features

### Grammar Correction
- **Ollama Integration**: Uses qwen3 for intelligent text correction
- **Fallback System**: Regex-based fixes when Ollama is unavailable
- **Common Fixes**: Handles typical PDF extraction issues like merged words

### Voice Cloning
- Upload reference audio for custom voice generation
- Adjust exaggeration levels for different speaking styles
- Fine-tune sampling parameters for optimal output

### Text Management
- Save extracted and corrected texts as JSON files
- Load previously saved texts for reprocessing
- Organize texts with metadata and timestamps

## 📄 License

This project builds upon ChatterboxTTS. Please refer to the original license terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 🙏 Acknowledgments

- Built on top of [ChatterboxTTS](https://github.com/fishaudio/chatterbox-tts)
- Grammar correction powered by Ollama and qwen3
- Web interface created with Gradio
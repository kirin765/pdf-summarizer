# PDF Summarizer

AI-powered PDF summarization web application using Flask and OpenAI GPT.

## Features

- **PDF Text Extraction**: Extract text from PDF files using PyMuPDF
- **OCR Support**: Automatic OCR using Tesseract for scanned PDFs
- **AI Summarization**: Generate concise summaries using OpenAI GPT-4
- **Multi-language**: Supports Korean and English
- **Rate Limiting**: Daily usage limits per IP (20 requests/day)

## Tech Stack

- **Backend**: Flask
- **AI**: OpenAI GPT-4.1-mini
- **PDF Processing**: PyMuPDF (fitz)
- **OCR**: Tesseract
- **Deployment**: Gunicorn

## Requirements

- Python 3.8+
- Tesseract OCR installed on system
- OpenAI API key

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pdf-summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies (not in requirements.txt)
pip install pytesseract Pillow pymupdf
```

## Environment Variables

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Running the App

```bash
# Development
python app.py

# Production with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Project Structure

```
pdf-summarizer/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── uploads/            # Temporary PDF upload directory
└── templates/          # HTML templates
```

## License

MIT

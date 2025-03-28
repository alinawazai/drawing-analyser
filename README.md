# Construction Drawing Analysis System

This Streamlit application allows you to analyze and search through construction drawings in PDF format. It uses advanced AI models to extract information and enable semantic search capabilities.

## Features

- PDF to image conversion
- Automatic block detection using YOLO
- Semantic search using FAISS and BM25
- RAG pipeline for information retrieval
- Interactive web interface

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- API keys for:
  - OpenAI
  - Cohere
  - Google Gemini

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd construction_drawing_app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export COHERE_API_KEY="your-cohere-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

4. Download the YOLO model weights:
Place the `best_small_yolo11_block_etraction.pt` file in the application directory.

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a PDF file containing construction drawings

4. Click "Process PDF" to analyze the drawings

5. Use the search interface to find specific information in the drawings

## How It Works

1. **PDF Processing**:
   - Converts PDF pages to images at different resolutions
   - Uses YOLO model to detect different blocks in the drawings

2. **Information Extraction**:
   - Crops and processes detected blocks
   - Extracts text and metadata from the blocks

3. **Search Functionality**:
   - Uses FAISS for vector similarity search
   - Implements BM25 for keyword-based search
   - Combines both approaches using ensemble retrieval
   - Reranks results using Cohere's reranking model

## Notes

- The application requires significant computational resources, especially for processing large PDF files
- Processing time depends on the size and complexity of the drawings
- Make sure you have sufficient disk space for temporary files

## License

[Your chosen license]
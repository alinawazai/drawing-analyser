import nest_asyncio
nest_asyncio.apply()

import asyncio
# Ensure an active event loop exists.
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import os
import json
import time
import glob
import logging
import concurrent.futures
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from ultralytics import YOLO
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from nltk.tokenize import word_tokenize
import torch
import nltk

# Download required NLTK resource if needed.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except FileExistsError:
        pass

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Directory structure (adjust as needed)
DATA_DIR = "data"
LOW_RES_DIR = os.path.join(DATA_DIR, "40_dpi")   # For detection
HIGH_RES_DIR = os.path.join(DATA_DIR, "500_dpi")   # For cropping
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def log_message(msg):
    st.sidebar.write(msg)

# Initialize session state (if not already)
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.gemini_documents = None
    st.session_state.vector_store = None
    st.session_state.compression_retriever = None

# -------------------------
# Pipeline Functions (unchanged)
# -------------------------
def pdf_to_images(pdf_path, output_dir, fixed_length=1080):
    log_message(f"Converting PDF to images at fixed length {fixed_length}px...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_message(f"Created directory: {output_dir}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_message(f"Error opening PDF: {e}")
        raise
    def process_page(page_number):
        page = doc[page_number]
        scale = fixed_length / page.rect.width
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)
        image_filename = f"{base_name}_page_{page_number + 1}.jpg"
        image_path = os.path.join(output_dir, image_filename)
        pix.save(image_path)
        log_message(f"Saved image: {image_path}")
        return image_path
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, i) for i in range(len(doc))]
        file_paths = [future.result() for future in concurrent.futures.as_completed(futures)]
    doc.close()
    log_message("PDF conversion completed.")
    return file_paths

# (Other functions remain the same...)

# -------------------------
# UI Layout
# -------------------------
st.sidebar.title("PDF Processing")
# Simple file uploader (PDF only)
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    # Ensure DATA_DIR exists
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_path = os.path.join(DATA_DIR, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success("PDF uploaded successfully.")

if uploaded_pdf and not st.session_state.processed:
    if st.sidebar.button("Run Processing Pipeline"):
        log_message("PDF uploaded successfully.")
        log_message("Converting PDF to images concurrently...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_low = executor.submit(pdf_to_images, pdf_path, LOW_RES_DIR, 662)
            future_high = executor.submit(pdf_to_images, pdf_path, HIGH_RES_DIR, 4000)
            low_res_paths = future_low.result()
            high_res_paths = future_high.result()
        log_message("PDF conversion completed.")

        log_message("Running YOLO detection on low-res images...")
        yolo_model = BlockDetectionModel("best_small_yolo11_block_etraction.pt")
        detection_results = yolo_model.predict_batch(LOW_RES_DIR)
        log_message("Block detection completed.")

        log_message("Cropping detected regions using high-res images...")
        cropped_data = crop_and_save(detection_results, OUTPUT_DIR)
        log_message("Cropping completed.")

        ocr_prompt = """\
You are an advanced system specialized in extracting standardized metadata from construction drawing texts.
Within the images you receive, there will be details pertaining to a single construction drawing.
Your job is to identify and extract exactly below fields from this text:
- 1st image has details about the drawing_title and scale
- 2nd Image has details about the client or project
- 4th Images has Notes
- 3rd Images has rest of the informations
- last image is the full image from which the above image are cropped
1. Purpose_Type_of_Drawing (examples: 'Architectural', 'Structural', 'Fire Protection')
2. Client_Name
3. Project_Title
4. Drawing_Title
5. Floor
6. Drawing_Number
7. Project_Number
8. Revision_Number (must be a numeric value, or 'N/A' if it cannot be determined)
9. Scale
10. Architects (list of names; use ['Unknown'] if no names are identified)
11. Notes_on_Drawing (any remarks or additional details related to the drawing)

Key Requirements:
- If any field is missing, return an empty string ('') or 'N/A' for that field.
- Return only a valid JSON object containing these nine fields in the order listed, with no extra text.
- Preserve all text in its original language (no translation), apart from minimal cleaning if necessary.
- Do not wrap the final JSON in code fences.
- Return ONLY the final JSON object with these fields and no additional commentary.
Below is an example json format:
{
    "Purpose_Type_of_Drawing": "Architectural",
    "Client_Name": "문촌주공아파트주택  재건축정비사업조합",
    "Project_Title": "문촌주공아파트  주택재건축정비사업",
    "Drawing_Title": "분산 상가-7  단면도-3  (근린생활시설-3)",
    "Floor": "주단면도-3",
    "Drawing_Number": "A51-2023",
    "Project_Number": "EP-201",
    "Revision_Number": 0,
    "Scale": "A1 : 1/100, A3 : 1/200",
    "Architects": ["Unknown"],
    "Notes_on_Drawing": "• Example note text."
}
"""
        log_message("Extracting metadata using Gemini OCR asynchronously...")
        gemini_documents = process_all_pages(cropped_data, ocr_prompt)
        log_message("Metadata extraction completed.")
        gemini_json_path = os.path.join(DATA_DIR, "gemini_documents.json")
        with open(gemini_json_path, "w") as f:
            json.dump([doc.dict() for doc in gemini_documents], f, indent=4)
        log_message("Gemini documents saved.")

        log_message("Building vector store for semantic search...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        example_embedding = embeddings.embed_query("sample text")
        d = len(example_embedding)
        index = faiss.IndexFlatL2(d)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        uuids = [str(uuid4()) for _ in range(len(gemini_documents))]
        vector_store.add_documents(documents=gemini_documents, ids=uuids)
        log_message("Vector store built and documents indexed.")

        log_message("Setting up retrievers...")
        bm25_retriever = BM25Retriever.from_documents(gemini_documents, k=10, preprocess_func=word_tokenize)
        retriever_ss = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":10})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever_ss],
            weights=[0.6, 0.4]
        )
        log_message("Setting up RAG pipeline...")
        compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )
        log_message("RAG pipeline set up.")
        st.session_state.processed = True
        st.session_state.gemini_documents = gemini_documents
        st.session_state.vector_store = vector_store
        st.session_state.compression_retriever = compression_retriever
        log_message("Processing pipeline completed.")

st.title("Chat Interface")
st.info("Enter your query below to search the processed PDF data.")
query = st.text_input("Query:")
if query and st.session_state.processed:
    st.write("Searching...")
    try:
        results = st.session_state.compression_retriever.invoke(query)
        st.markdown("### Retrieved Documents:")
        for doc in results:
            drawing = doc.metadata.get("drawing_name", "Unknown")
            st.write(f"**Drawing:** {drawing}")
            try:
                st.json(json.loads(doc.page_content))
            except Exception:
                st.write(doc.page_content)
            img_path = doc.metadata.get("drawing_path", "")
            if img_path and os.path.exists(img_path):
                st.image(Image.open(img_path), width=400)
    except Exception as e:
        st.error(f"Search failed: {e}")

st.write("Streamlit app finished processing.")

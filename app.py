import nest_asyncio
import asyncio
import os
import json
import glob
import logging
from uuid import uuid4
import time
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
import streamlit as st
from prompts import OCR_PROMPT
from google import genai

import nltk
nltk.download('punkt')
# Download required NLTK resource if needed.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except FileExistsError:
        pass
# Asyncio setup to allow async calls
nest_asyncio.apply()

# Load environment variables (Streamlit secrets)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

# Directory structure (adjust as needed)
DATA_DIR = "data"
LOW_RES_DIR = os.path.join(DATA_DIR, "40_dpi")   # For detection
HIGH_RES_DIR = os.path.join(DATA_DIR, "500_dpi")   # For cropping
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
client = genai.Client(api_key=GEMINI_API_KEY)

# Set up basic logging (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Helper function to log messages
def log_message(msg):
    st.sidebar.write(msg)

# Initialize session state (only processed flag and cached results)
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.gemini_documents = None
    st.session_state.vector_store = None
    st.session_state.compression_retriever = None
    st.session_state.vector_db_path = None  # Store path for uploaded vector database


# Asynchronous PDF to Images Conversion
async def pdf_to_images_async(pdf_path, output_dir, fixed_length=1080):
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

    file_paths = []
    tasks = []

    for i in range(len(doc)):
        tasks.append(process_page_async(doc, i, output_dir, base_name, fixed_length))

    results = await asyncio.gather(*tasks)
    for result in results:
        file_paths.extend(result)

    doc.close()
    log_message("PDF conversion completed.")
    return file_paths

# Process a single PDF page and convert it to an image
async def process_page_async(doc, page_num, output_dir, base_name, fixed_length=1080):
    page = doc[page_num]
    scale = fixed_length / page.rect.width
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix)
    image_filename = f"{base_name}_page_{page_num + 1}.jpg"
    image_path = os.path.join(output_dir, image_filename)
    pix.save(image_path)
    return [image_path]

# Block Detection Model
class BlockDetectionModel:
    def __init__(self, weight, device=None):
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(weight).to(self.device)
        log_message(f"YOLO model loaded on {self.device}.")

    async def predict_batch_async(self, images_dir):
        if not os.path.exists(images_dir) or not os.listdir(images_dir):
            raise ValueError(f"Directory {images_dir} is empty or does not exist.")
        images = glob.glob(os.path.join(images_dir, "*.jpg"))
        log_message(f"Found {len(images)} low-res images for detection.")
        
        output = {}
        batch_size = 10  # Process 10 images at a time
        tasks = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tasks.append(self.process_batch(batch, i, len(images)))

        results = await asyncio.gather(*tasks)
        for result in results:
            output.update(result)

        log_message("Block detection completed.")
        return output

    async def process_batch(self, batch, start_idx, total_images):
        log_message(f"Processing images {start_idx + 1} to {min(start_idx + len(batch), total_images)} of {total_images}.")
        results = self.model(batch)
        output = {}
        for result in results:
            image_name = os.path.basename(result.path)
            labels = result.boxes.cls.tolist()
            boxes = result.boxes.xywh.tolist()
            output[image_name] = [{"label": label, "bbox": box} for label, box in zip(labels, boxes)]
        return output

# Cropping function (asynchronous)
async def crop_and_save_async(detection_output, output_dir):
    log_message("Cropping detected regions using high-res images asynchronously...")
    output_data = {}
    tasks = []

    for image_name, detections in detection_output.items():
        tasks.append(crop_image_async(image_name, detections, output_dir))

    results = await asyncio.gather(*tasks)
    for result in results:
        output_data.update(result)

    log_message("Cropping completed.")
    return output_data

def scale_bboxes(bbox, src_size=(662, 468), dst_size=(4000, 3000)):
    scale_x = dst_size[0] / src_size[0]
    scale_y = scale_x
    return bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y
async def crop_image_async(image_name, detections, output_dir):
    image_resource_path = os.path.join(output_dir, image_name.replace(".jpg", ""))
    image_path = os.path.join(HIGH_RES_DIR, image_name)
    
    if not os.path.exists(image_resource_path):
        os.makedirs(image_resource_path)
    if not os.path.exists(image_path):
        log_message(f"High-res image missing: {image_path}")
        return {}

    try:
        with Image.open(image_path) as image:
            image_data = {}
            for det in detections:
                label = det["label"]
                bbox = det["bbox"]
                label_dir = os.path.join(image_resource_path, str(label))
                os.makedirs(label_dir, exist_ok=True)
                x, y, w, h = scale_bboxes(bbox)
                cropped_img = image.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                cropped_name = f"{label}_{len(os.listdir(label_dir)) + 1}.jpg"
                cropped_path = os.path.join(label_dir, cropped_name)
                cropped_img.save(cropped_path)
                image_data.setdefault(label, []).append(cropped_path)
            image_data["Image_Path"] = image_path
            return {image_name: image_data}
    except Exception as e:
        log_message(f"Error cropping {image_name}: {e}")
        return {}

# Asynchronous processing for Gemini OCR
async def process_with_gemini_async(image_paths, prompt):
    log_message(f"Asynchronously processing {len(image_paths)} images with Gemini OCR...")
    contents = [prompt]
    
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img_resized = img.resize((int(img.width / 2), int(img.height / 2)))
                contents.append(img_resized)
        except Exception as e:
            log_message(f"Error opening {path}: {e}")
            continue
    
    response = await asyncio.to_thread(client.models.generate_content, model="gemini-2.0-flash", contents=contents)
    log_message("Gemini OCR bulk response received.")
    resp_text = response.text.strip()
    print(f"Gemini OCR response: {resp_text}")
    if not resp_text:
        log_message("Empty response from Gemini OCR.")
        return None
    return resp_text

    # try:
    #     return json.loads(resp_text)
    # except json.JSONDecodeError:
    #     log_message(f"Failed to parse JSON: {resp_text}")
    #     return None

# Process pages with metadata using Gemini OCR asynchronously
async def process_all_pages_async(data, prompt):
    log_message("Processing all pages asynchronously...")
    tasks = []

    for key, blocks in data.items():
        tasks.append(process_page_with_metadata_async(key, blocks, prompt))

    results = await asyncio.gather(*tasks)
    documents = [result for result in results if result]
    
    log_message(f"Total {len(documents)} documents processed asynchronously.")
    return documents

# Process page metadata extraction
async def process_page_with_metadata_async(page_key, blocks, prompt):
    log_message(f"Processing page: {page_key}")
    all_imgs = []
    for block_type, paths in blocks.items():
        if block_type != "Image_Path":
            all_imgs.extend(paths)

    if not all_imgs:
        log_message(f"No cropped images for {page_key}")
        return None

    raw_metadata = await process_with_gemini_async(all_imgs, prompt)

    if raw_metadata:
        doc = Document(
            page_content=json.dumps(raw_metadata),
            metadata={"drawing_path": blocks["Image_Path"], "drawing_name": page_key, "content": "everything"}
        )
        log_message(f"Document created for {page_key}")
        return doc
    else:
        log_message(f"No metadata extracted for {page_key}")
        return None
    
# Run the full processing pipeline asynchronously
async def run_processing_pipeline(pdf_path):
    log_message("Running PDF processing pipeline...")

    # Step 1: Convert PDF to images asynchronously
    low_res_paths = await pdf_to_images_async(pdf_path, LOW_RES_DIR, 662)
    high_res_paths = await pdf_to_images_async(pdf_path, HIGH_RES_DIR, 4000)

    log_message("Running YOLO detection on low-res images...")
    yolo_model = BlockDetectionModel("best_small_yolo11_block_etraction.pt")
    detection_results = await yolo_model.predict_batch_async(LOW_RES_DIR)
    log_message("Block detection completed.")

    log_message("Cropping detected regions using high-res images...")
    cropped_data = await crop_and_save_async(detection_results, OUTPUT_DIR)

    ocr_prompt = OCR_PROMPT
    log_message("Processing images with Gemini OCR...")
    gemini_documents = await process_all_pages_async(cropped_data, ocr_prompt)

    # Ensure that documents are not empty before proceeding
    gemini_documents = [doc for doc in gemini_documents if doc is not None]

    log_message("Metadata extraction completed.")

    if not gemini_documents:
        log_message("No valid documents processed.")
        return None, None

    # Save metadata to JSON
    gemini_json_path = os.path.join(DATA_DIR, "gemini_documents.json")
    with open(gemini_json_path, "w") as f:
        json.dump([doc.dict() for doc in gemini_documents], f, indent=4)

    log_message("Gemini documents saved.")

    # Step 2: Build the vector store asynchronously
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
    try:
        vector_store.add_documents(documents=gemini_documents, ids=uuids)
    except Exception as e:
        log_message(f"Error adding documents to FAISS: {e}")
        return None, None

    log_message("Vector store built and documents indexed.")

    # Set up BM25 Retriever
    log_message("Setting up retrievers...")
    bm25_retriever = BM25Retriever.from_documents(gemini_documents, k=10, preprocess_func=word_tokenize)
    retriever_ss = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever_ss],
        weights=[0.6, 0.4]
    )

    log_message("Retriever setup complete.")

    # Set up RAG pipeline
    log_message("Setting up RAG pipeline...")
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    log_message("RAG pipeline set up.")

    return vector_store, compression_retriever

# Function to handle saving the vector database
def save_vector_db(vector_store):
    try:
        vector_db_path = os.path.join(DATA_DIR, "vector_db_index.faiss")
        faiss.write_index(vector_store.index, vector_db_path)
        st.session_state.vector_db_saved = True
        st.session_state.vector_db_path = vector_db_path  # Save the path for future download
        st.sidebar.success("Vector database saved successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to save Vector Database: {e}")

# Streamlit UI
def run_streamlit():
    st.sidebar.title("PDF Processing")
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    # Adding Query Text Box for Searching
    query = st.text_input("Enter your query here:")

    # Upload existing vector database
    uploaded_vector_db = st.sidebar.file_uploader("Upload Vector Database", type=["faiss"])
    if uploaded_vector_db:
        vector_db_path = os.path.join(DATA_DIR, uploaded_vector_db.name)
        with open(vector_db_path, "wb") as f:
            f.write(uploaded_vector_db.getbuffer())
        st.sidebar.success("Vector Database uploaded successfully.")
        st.session_state.vector_db_path = vector_db_path

    if uploaded_pdf:
        os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the data directory exists.
        pdf_path = os.path.join(DATA_DIR, uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.sidebar.success("PDF uploaded successfully.")

    if uploaded_pdf and not st.session_state.processed:
        if st.sidebar.button("Run Processing Pipeline"):
            # Run the processing pipeline asynchronously
            loop = asyncio.get_event_loop()
            vector_store, compression_retriever = loop.run_until_complete(run_processing_pipeline(pdf_path))

    # # Option to save the vector database after processing is done
    # if st.session_state.processed and st.sidebar.button("Save Vector Database"):
    #     save_vector_db(st.session_state.vector_store)

    # # Option to download the vector database
    # if st.session_state.vector_db_path and st.session_state.vector_db_saved:
    #     st.sidebar.download_button("Download Vector Database", data=open(st.session_state.vector_db_path, "rb"), file_name="vector_db_index.faiss", mime="application/octet-stream")


    if query:
        # Perform the retrieval from the vector store
        log_message("Performing search query...")

        if st.session_state.vector_store and st.session_state.compression_retriever:
            try:
                results = st.session_state.compression_retriever.invoke(query)
                st.markdown("### Retrieved Documents:")

                for doc in results:
                    drawing = doc.metadata.get("drawing_name", "Unknown")
                    st.write(f"**Drawing:** {drawing}")
                    # try:
                    #     st.json(json.loads(doc.page_content))  # Display content in JSON format
                    # except Exception:
                    #     st.write(doc.page_content)  # Fallback if JSON parsing fails

                    img_path = doc.metadata.get("drawing_path", "")
                    if img_path and os.path.exists(img_path):
                        st.image(Image.open(img_path), caption=f"Image for {drawing}", width=400)
            except Exception as e:
                st.error(f"Error during retrieval: {e}")
        else:
            st.error("No vector database or retriever available. Please process the PDF first.")

# Execute Streamlit UI
run_streamlit()
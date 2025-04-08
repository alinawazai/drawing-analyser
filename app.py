import nest_asyncio
nest_asyncio.apply()

import asyncio
# Ensure an active event loop exists.
# (Consider if nest_asyncio makes this manual loop handling redundant)
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
import shutil # <-- Import shutil for zip operations

# Download required NLTK resource if needed.
try:
    nltk.data.find('tokenizers/punkt') # Check for 'punkt' which word_tokenize uses
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
         st.error(f"Failed to download NLTK 'punkt': {e}")
         st.stop() # Stop if essential resource download fails


# Load environment variables
# Use st.secrets for deployment
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
COHERE_API_KEY = st.secrets.get("COHERE_API_KEY")

# Basic check if keys are loaded
if not all([GEMINI_API_KEY, OPENAI_API_KEY, COHERE_API_KEY]):
    st.error("API Keys (GEMINI, OPENAI, COHERE) are missing in Streamlit secrets!")
    st.stop()

# Directory structure (adjust as needed)
DATA_DIR = "data"
LOW_RES_DIR = os.path.join(DATA_DIR, "40_dpi")   # For detection
HIGH_RES_DIR = os.path.join(DATA_DIR, "500_dpi") # For cropping
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
VECTOR_STORE_UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded_vs") # For uploaded/extracted stores

# Ensure base directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOW_RES_DIR, exist_ok=True)
os.makedirs(HIGH_RES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_UPLOAD_DIR, exist_ok=True)


# Initialize Gemini Client (handle potential errors)
try:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
except ImportError:
    st.error("google.generativeai library not installed.")
    st.stop()
except Exception as e:
    st.error(f"Failed to initialize Gemini client: {e}")
    st.stop()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__) # Use a named logger

# Sidebar logger function
log_messages = []
def log_message(msg, level=logging.INFO):
    logger.log(level, msg)
    log_messages.append(msg)
    # Update sidebar dynamically (might impact performance if logs are very frequent)
    st.sidebar.text_area("Logs", value="\n".join(log_messages), height=200, key="log_area")


# --- Initialize Embeddings Early ---
# This needs to be available for both processing and loading vector stores
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    # Test embedding to catch potential API key or setup issues early
    _ = embeddings.embed_query("test")
    logger.info("OpenAI Embeddings initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize OpenAI Embeddings: {e}")
    logger.error(f"Failed to initialize OpenAI Embeddings: {e}", exc_info=True)
    st.stop()


# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
if "gemini_documents" not in st.session_state:
    st.session_state.gemini_documents = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "compression_retriever" not in st.session_state:
    st.session_state.compression_retriever = None

# -------------------------
# Pipeline Functions (Keep as defined in your original code)
# (Make sure they use the logger for messages instead of st.sidebar.write directly)
# -------------------------

def pdf_to_images(pdf_path, output_dir, fixed_length=1080):
    log_message(f"Converting PDF to images at fixed length {fixed_length}px...")
    # ... (rest of your function, use log_message)
    if not os.path.exists(pdf_path):
        log_message(f"PDF not found: {pdf_path}", level=logging.ERROR)
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_message(f"Created directory: {output_dir}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_message(f"Error opening PDF: {e}", level=logging.ERROR)
        raise

    file_paths = []
    total_pages = len(doc)
    log_message(f"Processing {total_pages} pages...")
    for i, page in enumerate(doc): # Use enumerate for progress
        # page = doc[i] # No need for this line if using enumerate
        scale = fixed_length / page.rect.width
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)
        image_filename = f"{base_name}_page_{i + 1}.jpg"
        image_path = os.path.join(output_dir, image_filename)
        try:
            pix.save(image_path)
            # log_message(f"Saved image: {image_path}") # Log only on success or less frequently
            file_paths.append(image_path)
        except Exception as e:
            log_message(f"Error saving image {image_path}: {e}", level=logging.ERROR)
        if (i + 1) % 10 == 0 or (i + 1) == total_pages: # Log progress every 10 pages
             log_message(f"Converted page {i + 1}/{total_pages}")

    doc.close()
    log_message(f"PDF conversion completed. Saved {len(file_paths)} images to {output_dir}.")
    return file_paths

class BlockDetectionModel:
    def __init__(self, weight, device=None):
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else "cpu"
        try:
            self.model = YOLO(weight).to(self.device)
            log_message(f"YOLO model '{weight}' loaded on {self.device}.")
        except Exception as e:
            log_message(f"Failed to load YOLO model from '{weight}': {e}", level=logging.ERROR)
            raise

    def predict_batch(self, images_dir):
        if not os.path.exists(images_dir) or not os.listdir(images_dir):
            log_message(f"Directory {images_dir} is empty or does not exist for YOLO prediction.", level=logging.WARNING)
            return {} # Return empty dict if no images
        images = glob.glob(os.path.join(images_dir, "*.jpg"))
        if not images:
            log_message(f"No .jpg images found in {images_dir}.", level=logging.WARNING)
            return {}
        log_message(f"Found {len(images)} low-res images for detection in {images_dir}.")

        output = {}
        batch_size = 16 # Adjust based on GPU memory

        try:
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                log_message(f"Detecting blocks in images {i + 1} to {min(i + batch_size, len(images))} of {len(images)}.")
                results = self.model(batch, verbose=False) # Reduce console spam
                for result in results:
                    image_name = os.path.basename(result.path)
                    labels = result.boxes.cls.tolist() # Class indices
                    confidences = result.boxes.conf.tolist() # Confidence scores
                    boxes = result.boxes.xywh.tolist() # Bounding boxes [x_center, y_center, width, height]

                    # Map class indices to names if available (optional but good practice)
                    class_names = result.names if hasattr(result, 'names') else {int(l): f'class_{int(l)}' for l in set(labels)}

                    output[image_name] = [
                        {"label": class_names.get(int(label), f'class_{int(label)}'), "confidence": conf, "bbox": box}
                        for label, conf, box in zip(labels, confidences, boxes)
                    ]
        except Exception as e:
            log_message(f"Error during YOLO prediction: {e}", level=logging.ERROR)
            raise # Re-raise the exception after logging

        log_message(f"Block detection completed. Processed {len(output)} images.")
        return output

# Add src_size parameter dynamically or make it more robust
def scale_bboxes(bbox, src_width, src_height, dst_width, dst_height):
    # bbox is [x_center, y_center, width, height]
    x_center, y_center, w, h = bbox
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height # Allow different scales if needed

    new_x_center = x_center * scale_x
    new_y_center = y_center * scale_y
    new_w = w * scale_x
    new_h = h * scale_y

    # Return coordinates for cropping (top-left x, top-left y, bottom-right x, bottom-right y)
    x1 = new_x_center - new_w / 2
    y1 = new_y_center - new_h / 2
    x2 = new_x_center + new_w / 2
    y2 = new_y_center + new_h / 2

    return x1, y1, x2, y2

def crop_and_save(detection_output, high_res_dir, output_base_dir):
    log_message(f"Cropping detected regions using high-res images from '{high_res_dir}'...")
    output_data = {}
    processed_count = 0

    # --- Get dimensions of one low-res image to calculate scale ---
    # Assuming all low-res images have the same dimensions used for detection
    low_res_img_path_example = None
    if detection_output:
        example_low_res_name = next(iter(detection_output))
        # Need to find the corresponding low-res image path used by YOLO
        # Let's assume it was in LOW_RES_DIR
        low_res_img_path_example = os.path.join(LOW_RES_DIR, example_low_res_name)

    if not low_res_img_path_example or not os.path.exists(low_res_img_path_example):
         log_message("Cannot determine low-res image dimensions for scaling. Using default (662x?).", level=logging.WARNING)
         # Provide defaults or raise error if scaling is critical
         # This default might be incorrect if pdf_to_images changed dimensions
         low_res_width, low_res_height = 662, 468 # Fallback, adjust as needed
    else:
        try:
             with Image.open(low_res_img_path_example) as low_img:
                 low_res_width, low_res_height = low_img.size
                 log_message(f"Determined low-res dimensions for scaling: {low_res_width}x{low_res_height}")
        except Exception as e:
             log_message(f"Error reading low-res image {low_res_img_path_example} for dimensions: {e}. Using default.", level=logging.WARNING)
             low_res_width, low_res_height = 662, 468 # Fallback

    # --- Iterate through detections ---
    for image_name, detections in detection_output.items():
        high_res_image_path = os.path.join(high_res_dir, image_name)
        page_output_dir = os.path.join(output_base_dir, image_name.replace(".jpg", "")) # e.g., data/output/pdfname_page_1

        if not os.path.exists(high_res_image_path):
            log_message(f"High-res image missing, skipping cropping: {high_res_image_path}", level=logging.WARNING)
            continue

        try:
            with Image.open(high_res_image_path) as high_res_image:
                high_res_width, high_res_height = high_res_image.size
                image_data = {"Image_Path": high_res_image_path, "cropped_regions": {}} # Store cropped paths under a key
                os.makedirs(page_output_dir, exist_ok=True) # Ensure page-specific output dir exists

                for i, det in enumerate(detections):
                    label = det["label"] # Use the actual class name/index
                    bbox_low_res = det["bbox"] # [x_center, y_center, width, height] relative to low-res

                    # Scale bbox coordinates
                    x1, y1, x2, y2 = scale_bboxes(bbox_low_res, low_res_width, low_res_height, high_res_width, high_res_height)

                    # Ensure coordinates are within image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(high_res_width, x2), min(high_res_height, y2)

                    # Check if the resulting box has valid dimensions
                    if x2 <= x1 or y2 <= y1:
                         log_message(f"Skipping invalid bbox after scaling/clipping for label '{label}' in {image_name}", level=logging.WARNING)
                         continue

                    # Create label-specific subdirectory if it doesn't exist
                    # Use a safe filename for the label directory if needed
                    safe_label_name = str(label).replace("/", "_").replace("\\", "_")
                    label_dir = os.path.join(page_output_dir, safe_label_name)
                    os.makedirs(label_dir, exist_ok=True)

                    # Crop the image
                    cropped_img = high_res_image.crop((int(x1), int(y1), int(x2), int(y2)))

                    # Save the cropped image
                    # Use a more descriptive name: pagename_label_index.jpg
                    cropped_name = f"{image_name.replace('.jpg','')}_{safe_label_name}_{i+1}.jpg"
                    cropped_path = os.path.join(label_dir, cropped_name)
                    cropped_img.save(cropped_path)

                    # Store the path relative to the label
                    if label not in image_data["cropped_regions"]:
                        image_data["cropped_regions"][label] = []
                    image_data["cropped_regions"][label].append(cropped_path)

                output_data[image_name] = image_data
                processed_count += 1
                if processed_count % 10 == 0:
                     log_message(f"Cropped images saved for {processed_count} pages...")

        except Exception as e:
            log_message(f"Error cropping {image_name}: {e}", level=logging.ERROR)

    log_message(f"Cropping completed. Processed {processed_count} pages.")
    return output_data


def process_with_gemini(image_paths, prompt):
    # Function expects a list of image paths
    if not image_paths:
        log_message("No images provided to Gemini.", level=logging.WARNING)
        return None

    log_message(f"Processing {len(image_paths)} image parts with Gemini...")
    contents = [prompt] # Start with the text prompt

    # Add images, handling potential errors
    processed_image_count = 0
    for path in image_paths:
        if not os.path.exists(path):
            log_message(f"Image file not found, skipping: {path}", level=logging.WARNING)
            continue
        try:
            # Consider resizing based on Gemini recommendations/limits if needed
            with Image.open(path) as img:
                # Example resize (optional, adjust as needed):
                # max_size = (1024, 1024)
                # img.thumbnail(max_size, Image.Resampling.LANCZOS)
                contents.append(img) # Add PIL image object directly
                processed_image_count += 1
        except Exception as e:
            log_message(f"Error opening or processing image {path}: {e}", level=logging.WARNING)

    if processed_image_count == 0:
        log_message("No valid images could be processed for Gemini.", level=logging.ERROR)
        return None

    # Add the full image path as the last element if needed by the prompt logic
    # Example: if full_image_path: contents.append(full_image_path) # Or the PIL image if preferred

    log_message(f"Sending {processed_image_count} image parts to Gemini model...")
    try:
        # Use generate_content for multi-modal input
        # Choose appropriate model (Flash is fast, Pro might be better for complex OCR)
        response = client.generate_content(model="models/gemini-1.5-flash", contents=contents)
        # Check for safety ratings or blocks if necessary
        # if response.prompt_feedback.block_reason:
        #     log_message(f"Gemini request blocked: {response.prompt_feedback.block_reason}", level=logging.ERROR)
        #     return None

        log_message("Gemini response received.")
        resp_text = response.text.strip()

        # Clean potential markdown code fences
        if resp_text.startswith("```"):
            resp_text = resp_text[3:]
            if resp_text.lower().startswith("json"):
                resp_text = resp_text[4:]
            if resp_text.endswith("```"):
                resp_text = resp_text[:-3]
            resp_text = resp_text.strip()

        # Attempt to parse JSON
        return json.loads(resp_text)

    except json.JSONDecodeError:
        log_message(f"Failed to parse JSON response from Gemini: {resp_text}", level=logging.ERROR)
        return None
    except Exception as e:
        # Catch other potential API errors (rate limits, config issues, etc.)
        log_message(f"Error calling Gemini API: {e}", level=logging.ERROR)
        return None


def process_page_with_metadata(page_key, page_data, prompt):
    log_message(f"Extracting metadata for page: {page_key}")

    # --- Prepare image inputs for Gemini ---
    # The prompt expects images in a specific order.
    # Modify this logic based on your actual `cropped_data` structure and prompt requirements.
    # Example: Assuming `page_data['cropped_regions']` is a dict like {'0': [path1], '1': [path2], ...}
    # And you have the full image path in `page_data['Image_Path']`
    all_cropped_paths = []
    cropped_regions = page_data.get("cropped_regions", {})

    # Get paths based on the prompt's expected order (adjust indices/keys as needed)
    # This requires knowing which cropped label corresponds to which part in the prompt
    # Example mapping (replace with your actual label meanings):
    prompt_order_labels = {
         'drawing_title_scale_label': 0, # Example: label '0' has title/scale
         'client_project_label': 1,      # Example: label '1' has client/project
         'notes_label': 3,               # Example: label '3' has notes
         'other_info_label': 2           # Example: label '2' has the rest
    }

    # Retrieve paths based on labels - handle missing labels gracefully
    img1_paths = cropped_regions.get(str(prompt_order_labels['drawing_title_scale_label']), [])
    img2_paths = cropped_regions.get(str(prompt_order_labels['client_project_label']), [])
    img3_paths = cropped_regions.get(str(prompt_order_labels['other_info_label']), [])
    img4_paths = cropped_regions.get(str(prompt_order_labels['notes_label']), [])

    # Combine paths in the order expected by the prompt
    # Add only the first image found for each label if multiple exist, or adjust logic
    ordered_image_paths = []
    if img1_paths: ordered_image_paths.append(img1_paths[0])
    if img2_paths: ordered_image_paths.append(img2_paths[0])
    if img3_paths: ordered_image_paths.append(img3_paths[0])
    if img4_paths: ordered_image_paths.append(img4_paths[0])

    # Add the full drawing image path (last, as per prompt)
    full_image_path = page_data.get("Image_Path")
    if full_image_path and os.path.exists(full_image_path):
        ordered_image_paths.append(full_image_path)
    else:
         log_message(f"Full image path missing or invalid for {page_key}", level=logging.WARNING)


    if not ordered_image_paths:
        log_message(f"No valid cropped or full images found for {page_key}, skipping Gemini.", level=logging.WARNING)
        return None

    # Call Gemini with the ordered image paths
    raw_metadata = process_with_gemini(ordered_image_paths, prompt)

    if raw_metadata and isinstance(raw_metadata, dict):
        # Create LangChain Document
        doc = Document(
            page_content=json.dumps(raw_metadata, ensure_ascii=False), # Store metadata dict as JSON string
            metadata={
                "drawing_path": page_data.get("Image_Path", "N/A"),
                "drawing_name": page_key,
                "source_pdf": page_key.split('_page_')[0] + ".pdf" # Example: derive PDF name
                # Add any other relevant metadata, e.g., page number
                # "page_number": int(page_key.split('_page_')[-1]) if '_page_' in page_key else 0
            }
        )
        log_message(f"Metadata extracted and Document created for {page_key}")
        return doc
    else:
        log_message(f"No valid metadata dictionary extracted by Gemini for {page_key}", level=logging.WARNING)
        return None

def process_all_pages(cropped_data, prompt):
    # Sequentially process each page's data
    documents = []
    total_pages = len(cropped_data)
    log_message(f"Starting sequential metadata extraction for {total_pages} pages...")
    start_time = time.time()

    for i, (page_key, page_data) in enumerate(cropped_data.items()):
        log_message(f"Processing page {i+1}/{total_pages}: {page_key}")
        doc = process_page_with_metadata(page_key, page_data, prompt)
        if doc:
            documents.append(doc)
        # Optional: add a small delay if hitting API rate limits
        # time.sleep(0.5)

    end_time = time.time()
    duration = end_time - start_time
    log_message(f"Metadata extraction finished for {len(documents)} pages in {duration:.2f} seconds.")
    return documents


# Simple tokenizer function (replace NLTK if causing issues)
def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    # Basic split on whitespace and punctuation, convert to lowercase
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# -------------------------
# Streamlit UI Layout
# -------------------------
st.set_page_config(layout="wide") # Use wider layout

st.sidebar.title("PDF Drawing Analyzer")

# File Uploaders
uploaded_pdf = st.sidebar.file_uploader("1. Upload PDF Drawing", type=["pdf"], key="pdf_uploader")
uploaded_vectorstore_zip = st.sidebar.file_uploader("2. OR Upload Existing Vector Store (.zip)", type=["zip"], key="vs_uploader")

# Processing Trigger
pdf_path = None
if uploaded_pdf:
    # Save uploaded PDF temporarily
    pdf_path = os.path.join(DATA_DIR, uploaded_pdf.name)
    try:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.sidebar.success(f"PDF '{uploaded_pdf.name}' uploaded.")
        log_message(f"PDF '{uploaded_pdf.name}' uploaded and saved to '{pdf_path}'.")
    except Exception as e:
        st.sidebar.error(f"Error saving uploaded PDF: {e}")
        log_message(f"Error saving uploaded PDF: {e}", level=logging.ERROR)
        pdf_path = None # Ensure path is None if save failed

# --- Load Vector Store Logic ---
if uploaded_vectorstore_zip and not st.session_state.processed:
    vs_name = uploaded_vectorstore_zip.name
    vs_base_name = vs_name.replace(".zip", "")
    saved_zip_path = os.path.join(VECTOR_STORE_UPLOAD_DIR, vs_name)
    extracted_vs_path = os.path.join(VECTOR_STORE_UPLOAD_DIR, vs_base_name) # Directory to extract into

    st.sidebar.info(f"Processing uploaded Vector Store '{vs_name}'...")
    try:
        # Save the uploaded zip
        with open(saved_zip_path, "wb") as f:
            f.write(uploaded_vectorstore_zip.getbuffer())
        log_message(f"Saved uploaded vector store zip to '{saved_zip_path}'")

        # Clean existing extraction dir if it exists
        if os.path.exists(extracted_vs_path):
            log_message(f"Removing existing directory: {extracted_vs_path}")
            shutil.rmtree(extracted_vs_path)

        # Extract the zip file
        log_message(f"Extracting '{saved_zip_path}' to '{extracted_vs_path}'...")
        shutil.unpack_archive(saved_zip_path, extracted_vs_path, 'zip')
        log_message("Extraction complete.")

        # Check if expected files exist after extraction
        faiss_file = os.path.join(extracted_vs_path, "index.faiss")
        pkl_file = os.path.join(extracted_vs_path, "index.pkl")
        if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
             raise FileNotFoundError(f"index.faiss or index.pkl not found in '{extracted_vs_path}' after extraction.")

        # Load the vector store (embeddings object is already initialized)
        log_message(f"Loading FAISS vector store from '{extracted_vs_path}'...")
        st.session_state.vector_store = FAISS.load_local(
            folder_path=extracted_vs_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True # Required for loading pickle file
        )
        log_message("FAISS vector store loaded successfully.")

        # --- Rebuild Retriever pipeline after loading VS ---
        # We need the documents to build BM25, which aren't in the saved VS.
        # Option 1: Save/Load documents alongside VS (e.g., in a separate JSON/pickle)
        # Option 2: Use only the FAISS retriever if documents aren't available.
        # Option 3: Re-run Gemini *if* cropped data is available/saved (less ideal)

        # Let's go with Option 2 for simplicity when loading a VS: Use only FAISS retriever.
        log_message("Setting up retriever from loaded Vector Store (FAISS only)...")
        retriever_ss = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k":10})

        # Setup compression (Cohere Reranker)
        try:
            compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5, api_key=COHERE_API_KEY)
            st.session_state.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever_ss # Use ONLY the FAISS retriever here
            )
            log_message("RAG pipeline (FAISS + Reranker) set up from loaded store.")
            st.session_state.processed = True # Mark as ready for search
            st.sidebar.success(f"Vector Store '{vs_base_name}' loaded and ready.")
        except Exception as e:
            log_message(f"Failed to set up Cohere Reranker: {e}", level=logging.ERROR)
            st.sidebar.error(f"Error setting up Reranker: {e}")
            # Reset state if retriever setup fails
            st.session_state.processed = False
            st.session_state.vector_store = None
            st.session_state.compression_retriever = None


        # Clean up uploaded zip file (optional)
        # try:
        #     os.remove(saved_zip_path)
        #     log_message(f"Removed temporary zip file: {saved_zip_path}")
        # except OSError as e:
        #     log_message(f"Could not remove zip file {saved_zip_path}: {e}", level=logging.WARNING)


    except Exception as e:
        log_message(f"Error loading vector store from zip: {e}", level=logging.ERROR)
        st.sidebar.error(f"Failed to load Vector Store: {e}")
        # Clean up extraction directory if loading failed
        if os.path.exists(extracted_vs_path):
            try:
                shutil.rmtree(extracted_vs_path)
                log_message(f"Cleaned up potentially incomplete extraction: {extracted_vs_path}")
            except OSError:
                pass
        # Reset state
        st.session_state.processed = False
        st.session_state.vector_store = None


# --- Processing Pipeline Trigger ---
# Only show button if a PDF is uploaded AND we haven't processed OR loaded a VS yet
if pdf_path and not st.session_state.processed:
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run Full Processing Pipeline", key="run_pipeline"):
        with st.spinner("Processing PDF... This may take several minutes."):
            st.session_state.processed = False # Ensure reset before starting
            st.session_state.gemini_documents = None
            st.session_state.vector_store = None
            st.session_state.compression_retriever = None
            log_messages.clear() # Clear logs for new run
            st.sidebar.text_area("Logs", value="", height=200, key="log_area") # Clear display too
            log_message("Starting PDF Processing Pipeline...")

            try:
                # 1. Convert PDF to Images
                log_message("Step 1/6: Converting PDF to images...")
                # Clear previous run's images
                if os.path.exists(LOW_RES_DIR): shutil.rmtree(LOW_RES_DIR); os.makedirs(LOW_RES_DIR)
                if os.path.exists(HIGH_RES_DIR): shutil.rmtree(HIGH_RES_DIR); os.makedirs(HIGH_RES_DIR)
                _ = pdf_to_images(pdf_path, LOW_RES_DIR, 662) # Low res for detection
                _ = pdf_to_images(pdf_path, HIGH_RES_DIR, 4000) # High res for cropping

                # 2. Detect Blocks with YOLO
                log_message("Step 2/6: Detecting blocks using YOLO...")
                # Ensure model path is correct
                yolo_model_path = "best_small_yolo11_block_etraction.pt"
                if not os.path.exists(yolo_model_path):
                     raise FileNotFoundError(f"YOLO model file not found: {yolo_model_path}")
                yolo_model = BlockDetectionModel(yolo_model_path)
                detection_results = yolo_model.predict_batch(LOW_RES_DIR)

                # 3. Crop Detected Regions
                log_message("Step 3/6: Cropping detected regions from high-res images...")
                # Clear previous output
                if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR); os.makedirs(OUTPUT_DIR)
                cropped_data = crop_and_save(detection_results, HIGH_RES_DIR, OUTPUT_DIR)

                # 4. Extract Metadata with Gemini
                log_message("Step 4/6: Extracting metadata using Gemini...")
                # Define the OCR prompt (ensure it matches your needs)
                ocr_prompt = """
                You are an advanced system specialized in extracting standardized metadata from construction drawing texts.
                Analyze the provided image parts, which are cropped regions from a single construction drawing page, plus the full drawing image itself (last).
                Identify and extract the following fields based on the typical location of information in title blocks and notes:
                - The first image part likely contains drawing_title and scale.
                - The second image part likely contains client or project information.
                - The fourth image part likely contains Notes.
                - The third image part likely contains other miscellaneous information.
                - The last image is the full drawing page for context.

                Extract these fields precisely:
                1. Purpose_Type_of_Drawing (e.g., 'Architectural', 'Structural', 'MEP', 'Fire Protection', 'Civil') - Infer if possible, else 'Unknown'.
                2. Client_Name (Name of the client or owning entity)
                3. Project_Title (Overall name of the construction project)
                4. Drawing_Title (Specific title of this drawing sheet)
                5. Floor (Floor level or area depicted, e.g., '1st Floor Plan', 'Roof Plan', 'Section A-A')
                6. Drawing_Number (The unique identifier code for this drawing)
                7. Project_Number (Identifier for the overall project)
                8. Revision_Number (The revision index, letter, or number. Use 0 or 'N/A' if none found.)
                9. Scale (Drawing scale, e.g., '1:100', '1/8" = 1\'-0"', 'N.T.S.')
                10. Architects (List of architect names or firm names; use ['Unknown'] if none)
                11. Notes_on_Drawing (Transcribe any general notes or important remarks found on the drawing, often in a dedicated 'Notes' section)

                Key Requirements:
                - Return ONLY a valid JSON object containing these fields.
                - If a field cannot be found, use "" (empty string) or "N/A" or ['Unknown'] as appropriate for the field type.
                - Preserve original language and text as much as possible.
                - Do NOT include ```json ``` code fences or any text outside the JSON object.

                Example JSON Output Format:
                {
                    "Purpose_Type_of_Drawing": "Architectural",
                    "Client_Name": "Sample Construction Inc.",
                    "Project_Title": "New Office Building",
                    "Drawing_Title": "First Floor Plan - West Wing",
                    "Floor": "First Floor",
                    "Drawing_Number": "A-101",
                    "Project_Number": "P-2024-01",
                    "Revision_Number": "B",
                    "Scale": "1/4\" = 1'-0\"",
                    "Architects": ["Design Firm XYZ"],
                    "Notes_on_Drawing": "1. All dimensions are to face of stud unless noted otherwise. 2. Verify all existing conditions on site."
                }
                """
                gemini_documents = process_all_pages(cropped_data, ocr_prompt)
                if not gemini_documents:
                    raise ValueError("No documents were generated by the Gemini processing step.")
                st.session_state.gemini_documents = gemini_documents # Save for potential later use

                # Save intermediate results (optional but good for debugging)
                # gemini_json_path = os.path.join(DATA_DIR, "gemini_extracted_docs.json")
                # try:
                #     with open(gemini_json_path, "w", encoding='utf-8') as f:
                #         json.dump([doc.dict() for doc in gemini_documents], f, indent=4, ensure_ascii=False)
                #     log_message(f"Gemini documents saved to {gemini_json_path}")
                # except Exception as e:
                #     log_message(f"Error saving Gemini documents: {e}", level=logging.WARNING)


                # 5. Build Vector Store & Retrievers
                log_message("Step 5/6: Building Vector Store and Retrievers...")
                # FAISS Vector Store
                log_message("Initializing FAISS index...")
                example_embedding = embeddings.embed_query("sample text for dimension check")
                d = len(example_embedding)
                index = faiss.IndexFlatL2(d) # Using L2 distance
                log_message(f"FAISS index created with dimension {d}.")
                vector_store = FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore=InMemoryDocstore(), # Simple in-memory storage
                    index_to_docstore_id={}
                )
                log_message("Adding documents to FAISS...")
                # Generate unique IDs for documents if they don't have them
                doc_ids = [str(uuid4()) for _ in gemini_documents]
                vector_store.add_documents(documents=gemini_documents, ids=doc_ids)
                st.session_state.vector_store = vector_store # Save to session state
                log_message("Documents added to FAISS vector store.")

                # BM25 Retriever (Keyword search)
                log_message("Setting up BM25 retriever...")
                # Use the simple tokenizer defined earlier
                bm25_retriever = BM25Retriever.from_documents(
                    gemini_documents,
                    k=10, # Retrieve top 10 keyword matches
                    preprocess_func=simple_tokenize # Use the simple tokenizer
                 )
                log_message("BM25 retriever ready.")

                # FAISS Retriever (Semantic search)
                retriever_ss = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
                log_message("FAISS similarity retriever ready.")

                # Ensemble Retriever (Combine BM25 and FAISS)
                log_message("Setting up Ensemble retriever...")
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, retriever_ss],
                    weights=[0.5, 0.5] # Adjust weights as needed (e.g., 0.4 keyword, 0.6 semantic)
                )
                log_message("Ensemble retriever ready.")

                # 6. Setup RAG Pipeline with Reranking
                log_message("Step 6/6: Setting up RAG pipeline with Cohere Reranker...")
                try:
                    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5, api_key=COHERE_API_KEY)
                    compression_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor,
                        base_retriever=ensemble_retriever # Use the combined retriever
                    )
                    st.session_state.compression_retriever = compression_retriever
                    log_message("RAG pipeline with reranking is set up.")
                except Exception as e:
                     log_message(f"Failed to set up Cohere Reranker: {e}", level=logging.ERROR)
                     raise ValueError(f"Failed to initialize Cohere Reranker: {e}")


                # Mark processing as complete
                st.session_state.processed = True
                log_message("✅ Processing pipeline completed successfully!")
                st.success("Processing complete! You can now query the document or download the vector store.")

            except FileNotFoundError as e:
                log_message(f"Error: A required file was not found: {e}", level=logging.ERROR)
                st.error(f"Processing failed: File not found - {e}")
                st.session_state.processed = False # Reset state on failure
            except ValueError as e:
                log_message(f"Error: A value error occurred: {e}", level=logging.ERROR)
                st.error(f"Processing failed: {e}")
                st.session_state.processed = False # Reset state on failure
            except Exception as e:
                log_message(f"An unexpected error occurred during processing: {e}", level=logging.ERROR)
                logger.error("Pipeline Error", exc_info=True) # Log full traceback
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.processed = False # Reset state on failure

            # Rerun to update the UI (show download/query sections)
            st.rerun()


# --- Download Vector Store Section (Corrected) ---
if st.session_state.processed and st.session_state.vector_store:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Download Vector Store")

    # Text input for the user to specify the base name
    vs_download_base_name = st.sidebar.text_input(
        "Enter a base name for download:",
        value="my_drawing_vectorstore", # Sensible default
        key="vs_download_name"
    )

    if vs_download_base_name:
        # Define directory path for saving locally before zipping
        vs_save_dir_path = os.path.join(DATA_DIR, vs_download_base_name)
        # Define the path/name for the zip archive
        zip_archive_path_base = os.path.join(DATA_DIR, vs_download_base_name) # e.g., data/my_vs
        zip_file_name = f"{vs_download_base_name}.zip" # e.g., my_vs.zip
        zip_file_full_path = f"{zip_archive_path_base}.zip" # e.g., data/my_vs.zip

        # Button to trigger saving and zipping
        if st.sidebar.button(f"💾 Save & Prepare '{zip_file_name}'", key="save_download_vs"):
            try:
                log_message(f"Saving vector store to directory: {vs_save_dir_path}...")
                # Ensure the target directory exists and is empty (optional, prevents merging issues)
                if os.path.exists(vs_save_dir_path):
                    shutil.rmtree(vs_save_dir_path)
                # os.makedirs(vs_save_dir_path) # save_local creates the directory

                # --- 1. Save the vector store ---
                st.session_state.vector_store.save_local(vs_save_dir_path)
                log_message("Vector Store saved locally.")

                # --- 2. Zip the directory ---
                log_message(f"Creating zip archive: {zip_file_full_path}...")
                shutil.make_archive(
                    base_name=zip_archive_path_base, # Path to the archive file (without .zip)
                    format='zip',
                    root_dir=DATA_DIR,          # Root for archive paths
                    base_dir=vs_download_base_name # Directory within root_dir to zip
                )
                log_message("Zip archive created successfully.")

                # --- 3. Provide the zip file for download ---
                with open(zip_file_full_path, "rb") as fp:
                    st.sidebar.download_button(
                        label=f"⬇️ Download {zip_file_name}",
                        data=fp,
                        file_name=zip_file_name,
                        mime="application/zip"
                    )
                log_message(f"Download button ready for {zip_file_name}.")

                # Optional: Clean up the unzipped save directory after creating the zip
                # try:
                #     shutil.rmtree(vs_save_dir_path)
                #     log_message(f"Removed temporary save directory: {vs_save_dir_path}")
                # except OSError as e:
                #     log_message(f"Could not remove temp dir {vs_save_dir_path}: {e}", level=logging.WARNING)

            except Exception as e:
                log_message(f"Error during vector store save/zip: {e}", level=logging.ERROR)
                logger.error("Save/Zip Error", exc_info=True)
                st.sidebar.error(f"Failed to save/zip vector store: {e}")
                # Clean up potentially incomplete zip file
                if os.path.exists(zip_file_full_path):
                    try: os.remove(zip_file_full_path)
                    except OSError: pass
    else:
        st.sidebar.warning("Please enter a base name for the download file.")


# -------------------------
# Main Chat Interface Area
# -------------------------
st.title("💬 Drawing Query Interface")

if not st.session_state.processed:
    st.info("Please upload a PDF and run the processing pipeline, or upload an existing Vector Store (.zip) using the sidebar.")
else:
    st.success("Vector store loaded. Ready for queries.")
    query = st.text_input("Enter your query about the drawing(s):", key="query_input")

    if query and st.session_state.compression_retriever:
        st.markdown("---")
        st.subheader("Search Results")
        with st.spinner("Searching and reranking relevant documents..."):
            try:
                start_time = time.time()
                # Use the compression retriever (which includes ensemble + reranking)
                results = st.session_state.compression_retriever.invoke(query)
                end_time = time.time()
                log_message(f"Query '{query}' processed in {end_time - start_time:.2f} seconds. Found {len(results)} results after reranking.")

                if results:
                    st.write(f"Found {len(results)} relevant sections:")
                    for i, doc in enumerate(results):
                        with st.expander(f"Result {i+1}: Drawing '{doc.metadata.get('drawing_name', 'Unknown')}'", expanded=i==0):
                            col1, col2 = st.columns([1,1]) # Two columns

                            with col1:
                                st.write("**Extracted Data:**")
                                try:
                                    # Parse the JSON content for display
                                    metadata_dict = json.loads(doc.page_content)
                                    st.json(metadata_dict) # Display JSON nicely
                                except json.JSONDecodeError:
                                    st.text(doc.page_content) # Fallback to raw text
                                st.caption(f"Source: {doc.metadata.get('drawing_name', 'N/A')}")
                                st.caption(f"Original PDF: {doc.metadata.get('source_pdf', 'N/A')}")


                            with col2:
                                img_path = doc.metadata.get("drawing_path")
                                if img_path and os.path.exists(img_path):
                                    st.write("**Drawing Image:**")
                                    try:
                                        image = Image.open(img_path)
                                        st.image(image, use_column_width=True)
                                    except Exception as e:
                                        st.warning(f"Could not load image {img_path}: {e}")
                                else:
                                    st.write("*(No image path found in metadata)*")
                else:
                    st.warning("No relevant documents found for your query after reranking.")

            except Exception as e:
                log_message(f"Error during query execution: {e}", level=logging.ERROR)
                logger.error("Query Error", exc_info=True)
                st.error(f"An error occurred during search: {e}")

# Display logs in the sidebar at the end
# st.sidebar.text_area("Logs", value="\n".join(log_messages), height=200, key="log_area_final")

# log_message("Streamlit app script execution finished.") # Log script end
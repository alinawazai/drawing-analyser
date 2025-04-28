import nest_asyncio
nest_asyncio.apply()

import asyncio
# Ensure an active event loop exists.
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
import shutil
import os
import faiss
import pickle
from io import BytesIO
import json
import time
import glob
import zipfile
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
from prompts import COMBINED_PROMPT
# Download required NLTK resource if needed.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except FileExistsError:
        pass

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

# Directory structure (adjust as needed)
DATA_DIR = "data"
LOW_RES_DIR = os.path.join(DATA_DIR, "40_dpi")   # For detection
HIGH_RES_DIR = os.path.join(DATA_DIR, "500_dpi")   # For cropping
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
from google import genai
client = genai.Client(api_key=GEMINI_API_KEY)
# Set up basic logging (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def log_message(msg):
    st.sidebar.write(msg)

# Initialize session state (only processed flag and cached results)
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.gemini_documents = None
    st.session_state.vector_store = None
    st.session_state.compression_retriever = None
    st.session_state.previous_pdf_uploaded = None  # Track the last uploaded PDF

# -------------------------
# Pipeline Functions (Sequential Version)
# -------------------------

def pdf_to_images(pdf_path, output_dir, fixed_length=1080):
    log_message(f"Converting PDF to images at fixed length {fixed_length}px...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    log_message(f"Created directory: {output_dir}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_message(f"Error opening PDF: {e}")
        raise

    file_paths = []
    # Optional: Limit number of pages (e.g., process only first 10 pages) 
    # max_pages = min(len(doc), 10)
    # for i in range(max_pages):
    for i in range(len(doc)):
        page = doc[i]
        scale = fixed_length / page.rect.width
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)
        image_filename = f"{base_name}_page_{i + 1}.jpg"
        image_path = os.path.join(output_dir, image_filename)
        pix.save(image_path)
        log_message(f"Saved image: {image_path}")
        file_paths.append(image_path)
    doc.close()
    log_message("PDF conversion completed.")
    return file_paths

class BlockDetectionModel:
    def __init__(self, weight, device=None):
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(weight).to(self.device)
        log_message(f"YOLO model loaded on {self.device}.")

    def predict_batch(self, images_dir):
        if not os.path.exists(images_dir) or not os.listdir(images_dir):
            raise ValueError(f"Directory {images_dir} is empty or does not exist.")
        images = glob.glob(os.path.join(images_dir, "*.jpg"))
        log_message(f"Found {len(images)} low-res images for detection.")
        
        output = {}
        batch_size = 10  # Process 10 images at a time
        
        # Process images in batches of 10
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            log_message(f"Processing images {i + 1} to {min(i + batch_size, len(images))} of {len(images)}.")
            results = self.model(batch)
            for result in results:
                image_name = os.path.basename(result.path)
                labels = result.boxes.cls.tolist()
                boxes = result.boxes.xywh.tolist()
                output[image_name] = [{"label": label, "bbox": box} for label, box in zip(labels, boxes)]
        
        log_message("Block detection completed.")
        return output


def scale_bboxes(bbox, src_size=(662, 468), dst_size=(4000, 3000)):
    scale_x = dst_size[0] / src_size[0]
    scale_y = scale_x
    return bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y

def crop_and_save(detection_output, output_dir):
    log_message("Cropping detected regions using high-res images...")
    output_data = {}
    for image_name, detections in detection_output.items():
        image_resource_path = os.path.join(output_dir, image_name.replace(".jpg", ""))
        image_path = os.path.join(HIGH_RES_DIR, image_name)
        if not os.path.exists(image_resource_path):
            os.makedirs(image_resource_path)
        if not os.path.exists(image_path):
            log_message(f"High-res image missing: {image_path}")
            continue
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
                output_data[image_name] = image_data
                log_message(f"Cropped images saved for {image_name}")
        except Exception as e:
            log_message(f"Error cropping {image_name}: {e}")
    log_message("Cropping completed.")
    return output_data

def process_with_gemini(image_paths, prompt):
    log_message(f"Asynchronously processing {len(image_paths)} images with Gemini OCR in bulk...")
    # Even though this step is originally asynchronous, processing sequentially reduces load.
    contents = [prompt]
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img_resized = img.resize((int(img.width / 2), int(img.height / 2)))
                contents.append(img_resized)
        except Exception as e:
            log_message(f"Error opening {path}: {e}")

    # time.sleep(4)  # Simple rate-limiting
    response = client.models.generate_content(model="gemini-2.0-flash", contents=contents)
    log_message("Gemini OCR bulk response received.")
    resp_text = response.text.strip()
    if resp_text.startswith("```"):
        resp_text = resp_text.replace("```", "").strip()
        if resp_text.lower().startswith("json"):
            resp_text = resp_text[4:].strip()
    try:
        return json.loads(resp_text)
    except json.JSONDecodeError:
        log_message(f"Failed to parse JSON: {resp_text}")
        return None

def process_page_with_metadata(page_key, blocks, prompt):
    log_message(f"Processing page: {page_key}")
    all_imgs = []
    for block_type, paths in blocks.items():
        if block_type != "Image_Path":
            all_imgs.extend(paths)
        if block_type == "Image_Path":
            all_imgs.append(paths)
    if not all_imgs:
        log_message(f"No cropped images for {page_key}")
        return None
    raw_metadata = process_with_gemini(all_imgs, prompt)
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

def process_all_pages(data, prompt):
    documents = []
    for key, blocks in data.items():
        doc = process_page_with_metadata(key, blocks, prompt)
        if doc:
            documents.append(doc)
        else:
            log_message(f"No document returned for {key}")
    log_message(f"Total {len(documents)} documents processed sequentially.")
    return documents

def save_vector_store_as_zip(vector_store, documents, zip_filename, high_res_images_dir=HIGH_RES_DIR):
    # Create a temporary directory to store the files
    temp_dir = os.path.join(DATA_DIR, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the FAISS index
    faiss_index_path = os.path.join(temp_dir, "faiss_index.index")
    faiss.write_index(vector_store.index, faiss_index_path)
    
    # Save the docstore using pickle
    docstore_path = os.path.join(temp_dir, "docstore.pkl")
    with open(docstore_path, "wb") as f:
        pickle.dump(vector_store.docstore, f)

    # Save the documents using pickle
    document_path = os.path.join(temp_dir, "document.pkl")
    with open(document_path, "wb") as f:
        pickle.dump(documents, f)

    # Include the high-resolution images
    high_res_image_dir = os.path.join(temp_dir, "high_res_images")
    os.makedirs(high_res_image_dir, exist_ok=True)

    # Copy all high-res images to the temporary directory
    for image_name in os.listdir(high_res_images_dir):
        image_path = os.path.join(high_res_images_dir, image_name)
        if os.path.isfile(image_path):
            shutil.copy(image_path, os.path.join(high_res_image_dir, image_name))
    
    # Create a zip file containing all necessary files
    zip_file_path = zip_filename
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(faiss_index_path, "faiss_index.index")
        zipf.write(docstore_path, "docstore.pkl")
        zipf.write(document_path, "document.pkl")
        
        # Add the images to the zip file
        for image_name in os.listdir(high_res_image_dir):
            image_path = os.path.join(high_res_image_dir, image_name)
            zipf.write(image_path, os.path.join("high_res_images", image_name))

    # Clean up temporary files with debugging output
    for temp_file in os.listdir(temp_dir):
        temp_file_path = os.path.join(temp_dir, temp_file)
        # Debug: Print the file path before removing
        print(f"Attempting to remove: {temp_file_path}")
        try:
            if os.path.exists(temp_file_path):  # Ensure the file exists before removing
                os.remove(temp_file_path)
            else:
                print(f"File not found: {temp_file_path}")
        except Exception as e:
            print(f"Failed to remove {temp_file_path}: {e}")
    
    shutil.rmtree(temp_dir)  # Remove the temporary directory

    return zip_file_path



st.image_dir_for_vector_db = DATA_DIR

def load_vector_store_from_zip(zip_filename, extraction_dir=DATA_DIR):
    # Create a temporary directory to extract the zip content
    temp_dir = os.path.join(extraction_dir, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        zipf.extractall(temp_dir)
    
    # Load the FAISS index
    faiss_index_path = os.path.join(temp_dir, "faiss_index.index")
    faiss_index = faiss.read_index(faiss_index_path)
    
    # Load the docstore
    docstore_path = os.path.join(temp_dir, "docstore.pkl")
    with open(docstore_path, "rb") as f:
        docstore = pickle.load(f)

    # Load the documents
    document_path = os.path.join(temp_dir, "document.pkl")
    with open(document_path, "rb") as f:
        documents = pickle.load(f)

    # Extract high-resolution images to a directory
    high_res_images_dir = os.path.join(extraction_dir, "high_res_images")
    st.image_dir_for_vector_db = high_res_images_dir
    os.makedirs(high_res_images_dir, exist_ok=True)

    for image_name in os.listdir(os.path.join(temp_dir, "high_res_images")):
        image_path = os.path.join(temp_dir, "high_res_images", image_name)
        if os.path.isfile(image_path):
            shutil.move(image_path, os.path.join(high_res_images_dir, image_name))


    # # Clean up the temporary directory
    # for temp_file in os.listdir(temp_dir):
    #     temp_file_path = os.path.join(temp_dir, temp_file)
    #     os.remove(temp_file_path)

    shutil.rmtree(temp_dir)  # Remove the temporary directory

    return faiss_index, docstore, documents
# -------------------------
# UI Layout
# -------------------------
st.sidebar.title("PDF Processing")
# It is recommended to add a file named `.streamlit/config.toml` in your project root with:
# [server]
# fileWatcherType = "none"

# Track if a new PDF is uploaded
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Reset the session state when a new PDF is uploaded
if uploaded_pdf:
    if uploaded_pdf.name != st.session_state.previous_pdf_uploaded:
        # Reset the session state for a new PDF upload
        st.session_state.processed = False
        st.session_state.gemini_documents = None
        st.session_state.vector_store = None
        st.session_state.compression_retriever = None
        st.session_state.previous_pdf_uploaded = uploaded_pdf.name  # Store the name of the newly uploaded PDF

    os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the data directory exists.
    pdf_path = os.path.join(DATA_DIR, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success("PDF uploaded successfully.")

if uploaded_pdf and not st.session_state.processed:
    if st.sidebar.button("Run Processing Pipeline"):
        log_message("PDF uploaded successfully.")

        log_message("Converting PDF to images sequentially...")
        low_res_paths = pdf_to_images(pdf_path, LOW_RES_DIR, 662)
        high_res_paths = pdf_to_images(pdf_path, HIGH_RES_DIR, 4000)
        log_message("PDF conversion completed.")
        # uploaded_pdf = None
        log_message("Running YOLO detection on low-res images...")
        yolo_model = BlockDetectionModel("best_small_yolo11_block_etraction.pt")
        detection_results = yolo_model.predict_batch(LOW_RES_DIR)
        log_message("Block detection completed.")

        log_message("Cropping detected regions using high-res images...")
        cropped_data = crop_and_save(detection_results, OUTPUT_DIR)
        log_message("Cropping completed.")

        ocr_prompt = COMBINED_PROMPT
        log_message("Extracting metadata using Gemini OCR sequentially...")
        gemini_documents = process_all_pages(cropped_data, ocr_prompt)
        log_message("Metadata extraction completed.")
        gemini_json_path = os.path.join(DATA_DIR, "gemini_documents.json")
        with open(gemini_json_path, "w") as f:
            json.dump([doc.dict() for doc in gemini_documents], f, indent=4, ensure_ascii=False)
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

if uploaded_pdf and st.session_state.processed:
    # Add the "Download Vector Store" button
    vector_store_filename = st.text_input("Enter the name for the vector store file:", "vector_store.zip")

    if st.button("Download Vector Store"):
        # Save the FAISS index and docstore into a zip file with images
        zip_file_path = save_vector_store_as_zip(
            st.session_state.vector_store, 
            st.session_state.gemini_documents, 
            os.path.join(DATA_DIR, vector_store_filename)
        )
        
        # Offer the zip file for download
        with open(zip_file_path, "rb") as f:
            zip_data = f.read()

        st.download_button(
            label="Download FAISS Vector Store as Zip",
            data=zip_data,
            file_name=vector_store_filename,
            mime="application/zip"
        )

# Add the "Upload Vector Store" button
uploaded_vector_store = st.file_uploader("Upload a vector store", type=[".zip"])

if uploaded_vector_store:
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        # Load the vector store from the uploaded files, including high-res images
        faiss_index, inmdocs, docs = load_vector_store_from_zip(uploaded_vector_store)

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
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)
        st.session_state.vector_store = vector_store
        bm25_retriever = BM25Retriever.from_documents(docs, k=10, preprocess_func=word_tokenize)
        retriever_ss = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever_ss],
            weights=[0.6, 0.4]
        )
        compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )
        st.session_state.compression_retriever = compression_retriever

        st.success(f"Vector store loaded successfully from {uploaded_vector_store.name}.")
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")

# st.title("Chat Interface")
# st.info("Enter your query below")
# query = st.text_input("Query Here:")
# if (uploaded_pdf and st.session_state.processed) or uploaded_vector_store:
#     if query:
#         st.write("Searching...")
#         try:
#             results = st.session_state.compression_retriever.invoke(query)
#             st.markdown("### Retrieved Documents:")
#             for doc in results:
#                 drawing = doc.metadata.get("drawing_name", "Unknown")
#                 st.write(f"**Drawing:** {drawing}")
#                 try:
#                     st.json(json.loads(doc.page_content))
#                 except Exception:
#                     st.write(doc.page_content)
#                 img_path = doc.metadata.get("drawing_path", "")
#                 extraction_dir=DATA_DIR
#                 img_path2 = os.path.join(st.image_dir_for_vector_db , img_path.split("/")[-1])
#                 if img_path and os.path.exists(img_path):
#                     st.image(Image.open(img_path), width=400)
#                 elif img_path2 and os.path.exists(img_path2):
#                     st.image(Image.open(img_path2), width=400)
#                 else:
#                     st.write(img_path2)
#         except Exception as e:
#             st.error(f"Search failed: {e}")

# st.write("Streamlit app finished processing.")

# ---------------------------------------
# Chat Interface (LLM-powered Q&A)
# ---------------------------------------
# ‹BEGIN NEW CHAT MODULE› ────────────────────────────────────────────────
import json, textwrap, re

# 1 ─── Few-shot helper: reformulate the user’s question for best recall
def reformulate_query(original_q: str) -> str:
    """
    Uses gemini-2.0-flash to turn the user question into a concise
    search query that maximises recall in the vector DB.
    Falls back to the original question if parsing fails.
    """
    system = (
        """
        "You are a search-assistant.\n"
        "Rewrite the user's question so it can be used as a retrieval query.\n"
        The chunks in my vector DB are JSON objects like this.\n":
        {
            "Drawing_Type": "Floor_Plan",
            "Purpose_of_Building": "Commercial",
            "Client_Name": "배상승공아파트주택재건축사업조합",
            "Project_Title": "배상승공아파트 재건축사업",
            "Drawing_Title": "분상상가-1 지하2층 평면도 (근린생활시설-3)",
            "Space_Classification": {
                "Communal": [],
                "Private": [],
                "Service": []
            },
            "Details": {
                "Drawing_Number": "A51-2000",
                "Project_Number": "N/A",
                "Revision_Number": 0,
                "Scale": "A1 : 1/100, A3 : 1/200",
                "Architects": ["SAMOO"]
            },
            "Additional_Details": {
                "Number_of_Units": 0,
                "Number_of_Stairs": 0,
                "Number_of_Elevators": 0,
                "Number_of_Hallways": 0,
                "Unit_Details": [],
                "Stairs_Details": [],
                "Elevator_Details": [],
                "Hallways": [],
                "Other_Common_Areas": []
            },
            "Notes_on_Drawing": "1. 해당 설계는 동지내부가 포함된 탁시형구조 및 지구단위계획에 따라 미설계될 수 있음.\n2. 해당도면 및 부속 자료는 건축허가 등 절차적 요건을 고려하여 결정할 것."
        }
        "Keep it short, and give it a proper structure to maximize the correct retrival, remove pronouns & page numbers, \n"
        "PLEASE REMOVE EXTRA INFORMATION, SPACES, LINE BREAKS, AND UNNECESSARY CHARACTERS.\n"
        "and keep important technical terms.\n"
        
        """
    )
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[system + "\n\nUser: " + original_q],
    )

    data = json.loads(resp.text)
    log_message(f"Reformulated query: {data.get('rewritten_query', original_q)}")
    return data.get("rewritten_query", original_q)
    

# 2 ─── Retrieve docs with the rewritten query
def retrieve_docs(search_q: str, k: int = 5):
    return st.session_state.compression_retriever.invoke(search_q)[:k]


# 3 ─── Build the final prompt for Gemini answer
def build_answer_prompt(user_q: str, docs):
    context_parts, used_tokens = [], 0
    for i, d in enumerate(docs, 1):
        # Pretty-print JSON, keep Unicode readable
        try:
            pretty = json.dumps(json.loads(d.page_content),
                                ensure_ascii=False, indent=2)
        except Exception:
            pretty = d.page_content
        snippet = pretty[:2000]  # trim long docs
        used_tokens += len(snippet)
        if used_tokens > 12000:      # leave room for instructions
            break
        context_parts.append(f"### Source {i}\n{snippet}")

    context_block = "\n\n".join(context_parts)

    system_msg = textwrap.dedent("""
        You are an expert civil-engineering assistant.
        • Answer the user's question using ONLY the sources below.
        • Cite sources like [1], [2].  If information is missing, say 
          "I couldn’t find that information." – do NOT guess.
    """).strip()

    return f"{system_msg}\n\n{context_block}\n\n### Question\n{user_q}\n\n### Answer"


# 4 ─── Two-stage RAG pipeline
def answer_with_rag(user_q: str):
    log_message("Searching...")
    retrieval_q = reformulate_query(user_q)
    log_message(f"Retrieval query: {retrieval_q}")
    docs = retrieve_docs(retrieval_q)

    if not docs:
        return "I couldn’t find any information related to your question.", []

    prompt = build_answer_prompt(user_q, docs)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return resp.text.strip(), docs


# 5 ─── Streamlit chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Drawing-AI Chat")

user_query = st.chat_input("Ask about the drawing, specification …")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    answer_text, source_docs = answer_with_rag(user_query)

    with st.chat_message("assistant"):
        st.markdown(answer_text)
        with st.expander("🔍 Sources used"):
            for i, d in enumerate(source_docs, 1):
                st.markdown(f"**[{i}]** *{d.metadata.get('drawing_name','?')}*")
                st.code(d.page_content[:1200], language="json")

    st.session_state.chat_history.extend([
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": answer_text}
    ])
# ‹END NEW CHAT MODULE› ─────────────────────────────────────────────────

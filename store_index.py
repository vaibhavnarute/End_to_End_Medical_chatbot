from src.helper import load_pdf_file, split_documents, download_hugging_face_embeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os

load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_API_TOKEN')

# Initialize HuggingFace Pipeline
model_id = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=huggingface_token)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=500,
    temperature=0.4,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)

extracted_data = load_pdf_file(data='Data/')
text_chunks = split_documents(extracted_data)  # Use split_documents here
embeddings = download_hugging_face_embeddings()

import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from src.helper import load_pdf_file, split_documents, download_hugging_face_embeddings
from langchain.vectorstores import FAISS

# Ensure the environment variable is set
huggingface_token = os.getenv('HUGGINGFACE_API_TOKEN')
os.environ["HUGGINGFACE_API_TOKEN"] = huggingface_token

# Initialize HuggingFace Pipeline
model_id = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=huggingface_token)

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
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know. "
    "Use 4 sentences maximum to keep the answer concise.\n\n"
    "{context}"
)

# Load and split documents
extracted_data = load_pdf_file(data='Data/')
text_chunks = split_documents(extracted_data)
embeddings = download_hugging_face_embeddings()

# Create the vector store and retriever
vectorstore = FAISS.from_documents(text_chunks, embeddings)
retriever = vectorstore.as_retriever()

# Define the retrieval chain
from langchain.chains import RetrievalQA
question_answer_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Define Streamlit UI
st.title("AI-Powered Question Answering System")
st.write("Ask a question based on the uploaded documents.")

# Input text box for user question
user_input = st.text_input("Enter your question:")

if user_input:
    # Run the retrieval chain
    response = question_answer_chain.run(user_input)
    st.write("Answer:", response)

# For debugging and insight into the loaded documents
if st.checkbox("Show Extracted Documents"):
    st.write(extracted_data)

if st.checkbox("Show Text Chunks"):
    st.write(text_chunks)

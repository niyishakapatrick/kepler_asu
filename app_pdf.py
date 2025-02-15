from langchain_community.chat_models import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_groq.chat_models import ChatGroq
import os
import time
from pathlib import Path
from PyPDF2 import PdfReader  # Replaced PdfFileReader with PdfReader


# Create .streamlit directory if it doesn't exist
os.makedirs('.streamlit', exist_ok=True)

# Create or overwrite config.toml
with open('.streamlit/config.toml', 'w') as f:
    f.write('[client]\ntoolbarMode = "minimal"')

# Define the PDF folder path (same directory as the script)
PDF_FOLDER = Path(__file__).parent / "pdf_folder"  # Use the parent directory of the script
os.makedirs(PDF_FOLDER, exist_ok=True)

# Function to handle PDF metadata issues
def process_pdfs_from_folder(pdf_folder):
    # Collect all PDF files in the folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise ValueError("No PDF files found in the specified folder.")

    temp_files = []
    all_documents = []

    try:
        # Loop through each PDF file and process it
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)

            # Check metadata before loading PDF
            print(f"Checking metadata for file: {pdf_path}")
            try:
                with open(pdf_path, "rb") as file:
                    pdf_reader = PdfReader(file)
                    metadata = pdf_reader.metadata
                    print("PDF Metadata:", metadata)
            except Exception as e:
                print(f"Warning: Failed to read metadata for {pdf_path}. Continuing without metadata. Error: {str(e)}")

            # Use PDFMinerLoader to process the PDF
            try:
                loader = PDFMinerLoader(pdf_path)
                documents = loader.load()
                all_documents.extend(documents)  # Add the extracted documents from this PDF
            except Exception as e:
                print(f"Error loading PDF with PDFMinerLoader: {str(e)}. Attempting to process without metadata.")
                documents = []  # Fallback if PDFMinerLoader fails

        if not all_documents:
            raise ValueError("No documents extracted from any PDF. Please check the file content.")

        # Split the extracted text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_documents)

        # Create a vector store from all documents
        vectorstore = FAISS.from_documents(splits, GPT4AllEmbeddings())

        return vectorstore

    finally:
        # Cleanup any temporary files (if any were created)
        for temp_file in temp_files:
            safe_remove(temp_file)

def safe_remove(filepath):
    """Safely remove a file with retries."""
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
            break
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(0.1)
            else:
                print(f"Warning: Could not remove temporary file {filepath}")

def get_response(user_query, chat_history, vectorstore=None):
    # llm = ChatOllama(model="mistral")
    llm = ChatGroq(
        model_name="mixtral-8x7b-32768",
        api_key="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"
    )
    
    if vectorstore:
        relevant_docs = vectorstore.similarity_search(user_query, k=3)
        relevant_content = "\n".join([doc.page_content for doc in relevant_docs])
        
        template = """
        You are a helpful assistant. Use the following pieces of context from the uploaded document to help answer the question.
        If the context isn't relevant, just answer based on your knowledge.
        
        Context from document: {context}
        
        Chat history: {chat_history}
        User question: {user_question}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        return chain.stream({
            "context": relevant_content,
            "chat_history": chat_history,
            "user_question": user_query
        })
    else:
        template = """
        You are a helpful assistant. 
        Answer the following questions considering the history of the conversation:
        Chat history: {chat_history}
        User question: {user_question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        return chain.stream({
            "chat_history": chat_history,
            "user_question": user_query
        })

# Streamlit UI
st.set_page_config(page_title="Kepler ASU Chatbot", page_icon="🤖")
st.image("logo.png", width=800)
st.markdown("### Chatbot Assistant", unsafe_allow_html=True)

# Sidebar info
#st.sidebar.image("logo_kepler.jpeg", width=80)
# st.sidebar.markdown("### Supported Formats")
# st.sidebar.markdown("""
# <small>
# - Only PDFs are processed automatically from the 'pdf_folder' directory.
# </small>
# """, unsafe_allow_html=True)

# Process PDFs from the specified folder (no upload option)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

try:
    # Process PDF files from the folder
    with st.spinner("Processing document..."):
        st.session_state.vectorstore = process_pdfs_from_folder(PDF_FOLDER)
    st.sidebar.success("ChatBot Initialized OK!")
except Exception as e:
    st.sidebar.error(f"Error processing PDFs: {str(e)}")

st.sidebar.markdown("<small>keplerasuscholars@asu.edu </small>", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi, I'm a bot. How can I help you? Ask questions about this scholar's program, and I'll answer them!")
    ]

# Display chat history with delete buttons
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.write(message.content)
        with col2:
            if isinstance(message, HumanMessage):  # Only show delete button for user messages
                if st.button("🗑️", key=f"delete_{idx}", help="Delete this message"):
                    delete_message(idx)

# Chat input
user_query = st.chat_input("Type your message here...")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        response = st.write_stream(get_response(
            user_query, 
            st.session_state.chat_history,
            st.session_state.vectorstore
        ))
    
    st.session_state.chat_history.append(AIMessage(content=response))

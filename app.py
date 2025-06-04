import streamlit as st
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import pinecone
import tempfile
import os
from io import BytesIO
import re
import streamlit.components.v1 as components

# Initialize Streamlit app
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    .sidebar-content {
        background-color: #fafafa;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# MathJax configuration for LaTeX rendering
components.html("""
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
""", height=0)

def init_api_keys():
    """Initialize API keys from Streamlit secrets or sidebar input"""
    with st.sidebar:
        st.markdown("## 🔑 API Configuration")
        
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        pinecone_key = st.text_input(
            "Pinecone API Key",
            type="password",
            help="Enter your Pinecone API key"
        )
        
        pinecone_env = st.text_input(
            "Pinecone Environment",
            placeholder="e.g., us-west1-gcp-free",
            help="Enter your Pinecone environment"
        )
        
        index_name = st.text_input(
            "Pinecone Index Name",
            placeholder="pdf-qa-index",
            help="Enter your Pinecone index name"
        )
    
    return openai_key, pinecone_key, pinecone_env, index_name

def process_pdf(pdf_file, openai_key, pinecone_key, pinecone_env, index_name):
    """Process uploaded PDF and create vector embeddings"""
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        
        # Initialize Pinecone
        pinecone.init(
            api_key=pinecone_key,
            environment=pinecone_env
        )
        
        # Create or connect to Pinecone index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine'
            )
        
        # Create vector store
        vectorstore = Pinecone.from_documents(
            texts,
            embeddings,
            index_name=index_name
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vectorstore, len(texts)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, 0

def create_qa_chain(vectorstore, openai_key):
    """Create conversational retrieval chain"""
    
    llm = OpenAI(
        temperature=0.7,
        openai_api_key=openai_key,
        model_name="gpt-3.5-turbo-instruct"
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return qa_chain

def render_latex_markdown(text):
    """Render LaTeX and Markdown content"""
    # Convert markdown-style formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    # Wrap text in MathJax-compatible div
    return f'<div class="arithmatex">{text}</div>'

def main():
    st.markdown('<h1 class="main-header">📚 LLM-Powered PDF Q&A Bot</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    # Get API keys
    openai_key, pinecone_key, pinecone_env, index_name = init_api_keys()
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("### 📄 Upload PDF Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to ask questions about"
        )
        
        if uploaded_file and all([openai_key, pinecone_key, pinecone_env, index_name]):
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing PDF and creating embeddings..."):
                    vectorstore, chunk_count = process_pdf(
                        uploaded_file, openai_key, pinecone_key, pinecone_env, index_name
                    )
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.qa_chain = create_qa_chain(vectorstore, openai_key)
                        st.success(f"✅ PDF processed successfully! Created {chunk_count} text chunks.")
                    else:
                        st.error("❌ Failed to process PDF. Please check your inputs.")
        
        # Display processing status
        if st.session_state.qa_chain:
            st.success("🤖 Bot is ready to answer questions!")
        else:
            st.info("👆 Please upload a PDF and configure API keys to start.")
    
    with col2:
        st.markdown("### 💬 Chat with your PDF")
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {question}
                </div>
                """, unsafe_allow_html=True)
                
                rendered_answer = render_latex_markdown(answer)
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Bot:</strong> {rendered_answer}
                </div>
                """, unsafe_allow_html=True)
        
        # Question input
        if st.session_state.qa_chain:
            question = st.text_input(
                "Ask a question about your PDF:",
                placeholder="e.g., What is the main topic of this document?",
                key="question_input"
            )
            
            col_ask, col_clear = st.columns([3, 1])
            
            with col_ask:
                if st.button("🚀 Ask Question", type="primary"):
                    if question:
                        with st.spinner("Thinking..."):
                            try:
                                result = st.session_state.qa_chain({
                                    "question": question
                                })
                                
                                answer = result['answer']
                                st.session_state.chat_history.append((question, answer))
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            
            with col_clear:
                if st.button("🗑️ Clear"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        else:
            st.info("Please process a PDF first to start asking questions.")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ How it works")
        st.markdown("""
        1. **Upload PDF**: Choose your document
        2. **Process**: Text is split and embedded
        3. **Ask**: Questions are answered using RAG
        4. **Render**: LaTeX/Markdown is displayed
        """)
        
        st.markdown("### 🔧 Features")
        st.markdown("""
        - OpenAI GPT integration
        - Pinecone vector storage
        - LangChain prompt chains
        - LaTeX/Markdown rendering
        - Conversational memory
        """)
        
        if st.session_state.chat_history:
            st.markdown("### 📊 Session Stats")
            st.metric("Questions Asked", len(st.session_state.chat_history))

if __name__ == "__main__":
    main()
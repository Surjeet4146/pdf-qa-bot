# LLM-Powered PDF Q&A Bot

A powerful chatbot that answers questions about PDF documents using OpenAI GPT and Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **PDF Processing**: Upload and process PDF documents
- **Vector Embeddings**: Integration with Pinecone for efficient similarity search
- **LangChain Integration**: Conversational retrieval chains with memory
- **LaTeX/Markdown Rendering**: MathJax support for mathematical expressions
- **Streamlit UI**: Clean, interactive web interface
- **Conversational Memory**: Maintains context across questions

## Technology Stack

- **LLM**: OpenAI GPT-3.5-turbo-instruct
- **Vector Database**: Pinecone
- **Framework**: LangChain
- **UI**: Streamlit
- **Math Rendering**: MathJax

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys Required**
   - OpenAI API Key
   - Pinecone API Key
   - Pinecone Environment name
   - Pinecone Index name

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Configure API Keys**: Enter your OpenAI and Pinecone credentials in the sidebar
2. **Upload PDF**: Choose a PDF document to analyze
3. **Process Document**: Click "Process PDF" to create embeddings
4. **Ask Questions**: Type questions about the document content
5. **Get Answers**: Receive contextual answers with LaTeX/Markdown rendering

## How It Works

1. **Document Processing**: PDF is loaded and split into chunks
2. **Embedding Creation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in Pinecone vector database
4. **Question Processing**: User questions are embedded and matched with relevant chunks
5. **Answer Generation**: LLM generates answers using retrieved context
6. **Rendering**: Responses are rendered with LaTeX/Markdown support

## Project Structure

```
pdf-qa-bot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Example Questions

- "What is the main topic of this document?"
- "Can you summarize the key findings?"
- "What mathematical formulas are mentioned?"
- "Explain the methodology used in this paper"

## Customization

- Adjust chunk size and overlap in `RecursiveCharacterTextSplitter`
- Modify LLM temperature for different response styles
- Change the number of retrieved chunks (`k` parameter)
- Customize the UI styling in the CSS section
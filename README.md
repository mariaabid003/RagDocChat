# RagDocChat

DocChat AI is a full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF, TXT, DOCX) and interactively ask questions about their content. The system uses advanced embeddings and Anthropic's Claude AI to provide accurate, context-aware answers with citations to the original documents.

## Features

- **User Authentication**: Secure signup and login with JWT-based authentication.
- **Document Processing**: Upload and parse PDF, TXT, and DOCX files.
- **RAG Capabilities**: Automatically chunks and embeds document text for semantic search.
- **Interactive Chat**: Ask questions across your uploaded documents. 
- **Source Citations**: AI responses include citations and relevance scores to the original document chunks.
- **Chat History**: Saves chat sessions and history using MongoDB.

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: MongoDB for user data and chat history
- **Vector Database**: ChromaDB (Local persistent storage)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Anthropic Claude API (`claude-haiku-4-5-20251001`)
- **Frontend**: Vanilla HTML/JS/CSS

## Getting Started

### Prerequisites

- Python 3.8+
- MongoDB instance (local or Atlas)
- Anthropic API Key

### Backend Setup

1. Navigate to the `backend` directory.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have a `.env` file in the `backend` directory with the following variables:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key
   MONGODB_URI=your_mongodb_connection_string
   JWT_SECRET=your_jwt_secret
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
   The backend will be available at `http://localhost:8000`.

### Frontend Setup

1. Navigate to the `frontend` directory.
2. Open `index.html` in your web browser, or serve it using a local web server (e.g., `python -m http.server 3000`).

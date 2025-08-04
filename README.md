
# Finance RAG

A sophisticated Retrieval-Augmented Generation (RAG) system built with Streamlit for financial document analysis and question-answering. This application combines the power of Google's Gemini Flash 2.5 model and GPT-4o model with advanced embedding techniques and vector search capabilities to provide accurate, context-aware responses to financial queries.

## 🏗️ System Architecture

![FinanceRAG System Architecture](https://github.com/Chandi713/Finance-RAG/blob/main/FinanceRAG%20System%20Architecture%20(2).jpeg)

### Core Components

- **Streamlit Frontend**: Interactive web interface for user queries and document management
- **Gemini Flash 2.5 Model**: Handles metadata extraction and user query comprehension
- **GPT-4o (Azure OpenAI)**: Primary language model for generating final responses to users
- **GTE-Finance-Embedding Model**: Specialized financial embeddings for document vectorization
- **Qdrant Vector Database**: High-performance vector storage and similarity search
- **Jina-Reranker-m0**: Advanced reranking model for improving search result relevance

### Data Flow

1. **Document Ingestion**: Users upload financial documents through the Streamlit interface
2. **Metadata Extraction**: Gemini Flash 2.5 processes documents to extract relevant metadata
3. **Query Understanding**: Gemini Flash 2.5 analyzes and comprehends user queries
4. **Embedding Generation**: GTE-Finance-Embedding model creates vector representations
5. **Vector Storage**: Embeddings and metadata stored in Qdrant vector database
6. **Query Processing**: User queries are embedded and matched against the vector database
7. **Result Reranking**: Jina-Reranker-m0 optimizes search results for relevance
8. **Response Generation**: GPT-4o (Azure OpenAI) generates final contextual answers using retrieved documents

## 🚀 Features

- **Financial Document Analysis**: Specialized for financial reports, statements, and market data
- **Intelligent Query Processing**: Advanced RAG pipeline with reranking capabilities
- **Conversation History**: Maintains context across multiple queries
- **Metadata Filtering**: Enhanced search with document metadata
- **Real-time Processing**: Fast document ingestion and query response

## 📋 Prerequisites

- Python 3.8+
- Streamlit
- Access to Google Gemini API
- Qdrant vector database instance
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chandi713/Finance-RAG.git
   cd Finance-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file with your API keys
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_URL=your_qdrant_instance_url
   JINA_URL=your_jina_reranker_url
   JINA_API=your_jina_api_key
   JINA_MODEL=jina-reranker-m0
   GEMINI_API_KEY=your_gemini_api_key
   AZURE_OPENAI_API_KEY=your_azure_openai_key
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   ```

4. **Configure the embedding model**
   - Ensure GTE-Finance-Embedding model is accessible
   - Set up Jina-Reranker-m0 model endpoints

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload Documents**
   - Navigate to the document upload section
   - Upload your financial documents (PDF, TXT, etc.)
   - Wait for processing and embedding generation

3. **Query Your Documents**
   - Enter your financial questions in the chat interface
   - The system will retrieve relevant context and generate informed responses
   - View conversation history and manage previous queries

## 📁 Project Structure

```
Finance-RAG/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
├── RAG_Pipeline/          # RAG processing modules
├── gte-finance-model/     # Embedding model components
└── .DS_Store             # System file (should be ignored)
```

## 🔧 Configuration

### Model Configuration
- **Embedding Model**: GTE-Finance-Embedding (optimized for financial content)
- **Query Understanding**: Google Gemini Flash 2.5 (metadata extraction and query comprehension)
- **Response Generation**: GPT-4o deployed on Azure OpenAI
- **Reranking Model**: Jina-Reranker-m0

### Vector Database
- **Database**: Qdrant
- **Collection**: Configured for financial document embeddings
- **Similarity Metric**: Cosine similarity (recommended)

## 🎯 Key Features Explained

### Metadata Extraction & Query Understanding
The system uses Gemini Flash 2.5 to:
- Automatically extract relevant metadata from financial documents
- Understand and comprehend user queries for better context matching
- Process document types, key financial metrics, date ranges, and entity information

### Advanced Reranking
Jina-Reranker-m0 improves search quality by:
- Reordering initial similarity search results
- Considering query-document semantic alignment
- Optimizing for financial domain relevance

### Response Generation
GPT-4o on Azure OpenAI provides:
- High-quality, contextual responses based on retrieved documents
- Financial domain expertise for accurate interpretations
- Natural language generation optimized for financial content

### Conversation Memory
The application maintains conversation history to:
- Provide context-aware responses
- Enable follow-up questions
- Maintain session continuity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

# Finance RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system specifically designed for financial document understanding and question-answering. This research project implements and evaluates state-of-the-art embedding models fine-tuned on diverse financial datasets to enhance retrieval accuracy in finance-specific contexts.

## 🎯 Project Description and Purpose

### Overview

The Finance RAG System addresses the critical challenge of accurate information retrieval from complex financial documents by implementing domain-specific fine-tuning of embedding models. This study explores the application of retrieval-augmented generation (RAG) methodologies to improve the accuracy and reliability of large language models (LLMs) in financial document analysis contexts.

### Research Objectives

- **Domain-Specific Embedding Enhancement**: Fine-tune state-of-the-art embedding models (BGE-M3, GTE-multilingual) on comprehensive financial Q&A datasets to improve domain-specific retrieval performance
- **Comprehensive Performance Evaluation**: Conduct systematic evaluation across multiple financial benchmarks to assess model performance improvements and generalization capabilities
- **Methodological Innovation**: Implement Low-Rank Adaptation (LoRA) techniques for efficient fine-tuning while preserving computational efficiency and model stability
- **Production-Ready Implementation**: Deliver trained models with detailed performance analytics and reproducible training pipelines

### Key Research Contributions

Our research demonstrates significant performance improvements through domain-specific fine-tuning:

- **BGE-M3 Finance Model**: Achieves performance improvements ranging from **3.76% to 5.41%** across key information retrieval metrics
- **GTE-Multilingual Finance Model**: Demonstrates performance gains of **6.08% to 8.62%** on financial document retrieval tasks
- **Comprehensive Benchmarking**: Systematic evaluation across 7 financial datasets comprising 8,583+ training samples and 905+ evaluation instances
- **Novel Methodological Application**: First comprehensive study applying multi-functional embedding models to financial domain question-answering with rigorous comparative analysis

### Technical Innovation

The system leverages BGE M3-Embedding, distinguished for its versatility in Multi-Linguality (100+ languages), Multi-Functionality (dense, sparse, multi-vector retrieval), and Multi-Granularity (up to 8192 tokens). Additionally, the GTE-multilingual-base model supports 70+ languages with 305 million parameters, delivering efficient resource utilization while maintaining high-quality embeddings.

**Multi-Modal Retrieval Capabilities:**
- **Dense Retrieval**: Traditional embedding-based semantic search with cosine similarity
- **Sparse Retrieval**: Lexical matching capabilities for precise term-based retrieval
- **Multi-Vector Retrieval**: Advanced representation learning for complex document understanding

## 📊 Dataset Details

### Training Data Sources

The system leverages a comprehensive collection of financial Q&A datasets from established research benchmarks:

#### Primary Datasets

- **ConvFinQA**: Multi-turn conversational financial question answering with contextual reasoning capabilities
- **FinanceBench**: A specialized test suite related to publicly traded companies with corresponding evidence strings
- **FinQA**: Dataset containing questions on financial reports, annotated by domain experts with numerical reasoning processes
- **TATQA**: Question answering benchmark on hybrid tabular and textual content in finance, focusing on structured data interpretation
- **FinDER**:  Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation
- **MultiHiertt**: Multi-hop reasoning on financial tables and hierarchical data structures
- **Financial-QA-10K**: Custom financial 10-K document Q&A pairs for regulatory filing comprehension


#### Data Accessibility

**Public Datasets**: ConvFinQA, FinanceBench, FinQA, TATQA, and MultiHiertt are publicly available through their respective research publications and repositories.

**Processed Data Structure**: Organized in `Datasets/` directory with standardized JSONL format:
- Query files: `{dataset}_queries.jsonl/queries.jsonl`
- Corpus files: `{dataset}_corpus.jsonl/corpus.jsonl`
- Relevance judgments: `{dataset}_qrels.tsv`

**Training Data Consolidation**: `TrainingData_V3.csv` contains the preprocessed and consolidated training dataset ready for model fine-tuning.

## 🚀 Instructions to Clone Repository and Run Code

#### 1. Environment Configuration

Create and activate a Python virtual environment:

```bash
python -m venv finance-rag-env
source finance-rag-env/bin/activate  # On Windows: finance-rag-env\Scripts\activate
```

#### 2. Dependency Installation

Install required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Execution Pipeline

#### Data Preprocessing

Execute data preprocessing and preparation:

```bash
jupyter notebook Pre-Process.ipynb
```
#### Model Training

**BGE-M3 Model Fine-tuning:**
```bash
jupyter notebook train_bge-m3_model.ipynb
```

**GTE-Multilingual Model Fine-tuning:**
```bash
jupyter notebook train_gte-multilingual-base_model.ipynb
```

#### Model Evaluation

**BGE-M3 Performance Assessment:**
```bash
jupyter notebook eval_bge-m3_model.ipynb
```

**GTE-Multilingual Performance Assessment:**
```bash
jupyter notebook eval_gte-multilingual-base_model.ipynb
```

#### Inference Usage

Load and utilize fine-tuned models for financial document retrieval:

```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned models
bge_model = SentenceTransformer('./bge-m3-finance-model')
gte_model = SentenceTransformer('./gte-finance-model')

# Example query and document corpus
query = "What were the primary revenue drivers in Q3 2023?"
documents = [
    "Revenue increased by 15% in Q3 2023 primarily due to strong performance in our core business segments.",
    "Operating expenses were effectively controlled during the third quarter reporting period."
]

# Generate embeddings and compute similarity
query_embedding = bge_model.encode(query)
doc_embeddings = bge_model.encode(documents)
similarities = bge_model.similarity(query_embedding, doc_embeddings)
```

### Hardware Requirements

**Minimum Configuration:**
- **GPU**: CUDA-compatible GPU with 16GB+ VRAM
- **RAM**: 16GB+ system memory
- **Storage**: 15GB+ available disk space

**Recommended Configuration:**
- **GPU**: NVIDIA RTX 3080/4080 or better (12GB+ VRAM)
- **RAM**: 32GB+ system memory
- **Storage**: SSD with 50GB+ available space for optimal performance

### Training Configuration Parameters

**BGE-M3 Model Training:**
- **Base Model**: `BAAI/bge-m3`
- **LoRA Rank**: 16, Alpha: 32, Dropout: 0.1
- **Learning Rate**: 2e-5
- **Batch Size**: 4 (training), 16 (evaluation)
- **Max Sequence Length**: 256 tokens

**GTE-Multilingual Model Training:**
- **Base Model**: `Alibaba-NLP/gte-multilingual-base`
- **LoRA Rank**: 16, Alpha: 32, Dropout: 0.1
- **Learning Rate**: 2e-5
- **Batch Size**: 8 (training), 16 (evaluation)
- **Max Sequence Length**: 512 tokens

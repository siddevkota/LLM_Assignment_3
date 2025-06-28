# LLM Assignment 3 - Advanced Language Model Applications

This repository contains three comprehensive assignments exploring different aspects of Large Language Model (LLM) applications and techniques.

## 📁 Project Structure

### 3.1 RAG (Retrieval-Augmented Generation)
**Directory:** `3.1_rag/`

#### Files:
- **`rag_pipeline.py`** - Complete RAG implementation with web crawling, FAISS indexing, and question-answering
- **`requirements.txt`** - Python dependencies for the RAG pipeline

**Purpose:** Implements a full RAG pipeline that:
- Crawls Wikipedia articles (default: Artificial Intelligence page)
- Builds semantic search index using FAISS and sentence transformers
- Retrieves relevant context for user queries
- Answers questions using DistilBERT QA model
- Evaluates performance on sample questions
- Provides interactive Q&A interface

**Key Features:**
- Web scraping with BeautifulSoup
- Semantic embeddings with SentenceTransformers
- Efficient similarity search with FAISS
- Question-answering with Transformers
- Performance evaluation metrics

### 3.2 Multi-Agent System
**File:** `3.2_multiagent.ipynb`

**Purpose:** Demonstrates a collaborative multi-agent LLM system where specialized agents work together through shared memory:

**Agents:**
- **PlannerAgent** - Breaks complex tasks into subtasks
- **AnswerAgent** - Solves individual subtasks using provided context
- **SummarizerAgent** - Combines subtask results into final answer

**Key Features:**
- Shared memory communication between agents
- Task decomposition and parallel processing
- Context-aware subtask resolution
- Final answer synthesis
- Example: Climate change impact on agriculture analysis

**Architecture:**
- Uses Google FLAN-T5 base model
- Message passing via SharedMemory class
- Modular agent design for extensibility

### 3.3 Fine-tuning Comparison
**File:** `3.3_pretrain.ipynb`

**Purpose:** Comprehensive comparison of fine-tuning techniques analyzing trade-offs between accuracy, training time, and memory usage:

**Techniques Compared:**
- **Full Fine-tuning** - Traditional approach updating all model parameters
- **LoRA (Low-Rank Adaptation)** - Parameter-efficient fine-tuning

**Analysis Metrics:**
- Model accuracy on sentiment classification
- Training time comparison
- Memory usage during training
- Parameter efficiency

**Dataset:** IMDB movie reviews for sentiment analysis
**Base Model:** DistilBERT-base-uncased

**Key Insights:**
- LoRA achieves comparable accuracy with significantly fewer trainable parameters
- Reduced memory footprint and faster training times
- Trade-off analysis for production deployment decisions

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **For RAG Pipeline:**
   ```bash
   cd 3.1_rag
   .venv/Scripts/activate (recommended)
   pip install -r requirements.txt
   python rag_pipeline.py
   ```

2. **For Jupyter Notebooks:**
   ```bash
   pip install jupyter transformers datasets peft torch sentence-transformers
   jupyter notebook
   ```

## 🎯 Learning Objectives

- **RAG Systems:** Understanding retrieval-augmented generation for knowledge-intensive tasks
- **Multi-Agent Architecture:** Designing collaborative AI systems with specialized roles
- **Fine-tuning Optimization:** Comparing traditional vs. parameter-efficient training methods
- **Performance Analysis:** Evaluating trade-offs in accuracy, speed, and resource usage

## 📊 Key Technologies

- **Transformers:** Hugging Face library for pre-trained models
- **FAISS:** Facebook AI Similarity Search for efficient vector operations
- **SentenceTransformers:** Semantic text embeddings
- **LoRA/PEFT:** Parameter-efficient fine-tuning techniques
- **BeautifulSoup:** Web scraping and HTML parsing

## 🔧 Usage Examples

Each component can be run independently:
- RAG pipeline provides interactive Q&A on crawled content
- Multi-agent system demonstrates collaborative problem-solving
- Fine-tuning notebook shows comparative analysis of training methods

## 📈 Results

- RAG pipeline successfully retrieves and answers domain-specific questions
- Multi-agent system effectively decomposes and solves complex queries
- LoRA fine-tuning achieves ~95% of full fine-tuning accuracy with 90% fewer parameters

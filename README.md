# PDF RAG Chatbot with Cohere Fine-Tuning

A Streamlit-based RAG chatbot that processes PDFs (text-based and image-based), generates answers using Cohere API, evaluates retrieval with Precision@3 and MRR, and fine-tunes Cohere's `command-r` model on PDF content.

## Setup
1. **Open in VS Code**: File > Open Folder > Select `pdf-rag-chatbot`.
2. **Install Dependencies**: Open integrated terminal (`Ctrl + ` `) and run `pip install -r requirements.txt`.
3. **System Dependencies**: Install Poppler and Tesseract (see README details).
4. **Set Cohere API Key**: Add to `.streamlit/secrets.toml`.
5. **Run**: `streamlit run app.py` in terminal.

## Usage
- Upload PDF.
- Chat: Ask questions.
- Evaluation: Generate/edit ground truth, evaluate retrieval.
- Fine-Tuning: Generate dataset from PDF, start fine-tuning.

## Data
- Chunks: `data/chunks.json`
- Ground Truth: `data/ground_truth.json`
- Fine-Tuning Dataset: `data/fine_tune_dataset.jsonl`

## Costs (September 2025)
- Cohere API: Free tier (1,000 calls/month). Fine-tuning: $3/1M tokens. Inference: $0.30-1.20/M tokens.

## Notes
- Monitor fine-tuning in Cohere Dashboard.
- Improve OCR with preprocessing.
- For multilingual: Use `embed-multilingual-v3.0`.
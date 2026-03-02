# Legal LLM Workflow

This project has two main notebooks:
- `data_collection_preprocessing.ipynb`
- `legal_llm_model.ipynb.ipynb`

They implement a full pipeline: **collect legal text -> clean/format -> tokenize -> batch cache -> train custom decoder LLM -> run inference**.

## 1) Data collection and preprocessing (`data_collection_preprocessing.ipynb`)

### 1.1 Batch cache system
The notebook defines `SimpleBatchCache`, which stores training batches as sequential files:
- `batch001.pt`, `batch002.pt`, ...
- A `master.json` file tracks global and per-file metadata (`total_tokens`, `total_sequences`, timestamps, custom source metadata, etc.).

This cache is the bridge between preprocessing and training.

### 1.2 Source A: Client-Lawyer Q&A text
Processor: `QADataProcessor`
- Expected input format per record:
  - `N.Client: <question>`
  - `Lawyer: <answer>`
- Parsing: `parse_qa_text()` extracts `client` and `lawyer` pairs.
- Training format: `Client: ... </s> Lawyer: ...`
- Tokenization: RoBERTa tokenizer (`roberta-base`), fixed max length.
- Labels: left-shifted causal-LM labels, padding masked to `-100`.
- Batching: fixed example count per batch.

Pipeline wrapper: `process_qa_file_to_cache(...)`
- Reads source text file
- Parses Q&A pairs
- Tokenizes
- Creates batches
- Saves to cache with metadata (`data_type: legal_qa`, tokenizer, max_length, counts)

Notebook usage example:
- Input: `/content/client_lawer.txt`
- Cache dir: `/content/drive/MyDrive/cache`
- `batch_size=16`, `max_length=250`

### 1.3 Source B: Raw legal documents (txt from PDFs)
PDF extraction utility: `pdf_to_text(...)`
- Uses `PyPDF2.PdfReader`
- Combines text from multiple PDFs into one text output
- Optional newline stripping for smoother plain text

Document processor: `SimpleDocumentProcessor`
- Tokenizes full text
- Splits into overlapping chunks (`max_length`, `overlap`)
- Builds causal-LM labels (shifted, pad masked)
- Creates tensor batches

Pipeline wrapper: `process_legal_documents_simple(...)`
- Typical tokenizer: `roberta-base`
- In function: processor configured with `max_length=2048`, `overlap=50`
- Saves batch files with metadata (`data_type: legal_document_clm`)

Notebook usage example:
- `file_paths = ['supreme_court_1885.txt']`
- `cache_dir = '/content/drive/MyDrive/cache'`

### 1.4 Source C: Generic JSONL legal corpora
Processor: `JSONLDataProcessor`
- Reads each JSONL line as JSON object
- Serializes object to text (`json.dumps`)
- Tokenizes full object
- Creates overlapping fixed-length chunks
- Builds causal-LM labels and attention masks
- Smart-batches by token budget (`tokens_per_batch`)

Pipeline wrapper: `process_jsonl_smart(...)`
- Stores batches + metadata in cache (`data_type: jsonl_data`)

Notebook examples include combining JSONL files and caching with token-budget batching (`tokens_per_batch` values like 3600/9000).

### 1.5 Source D: Hugging Face instruction dataset
Dataset loaded in notebook:
- `Alignment-Lab-AI/Lawyer-Instruct`

Flow used there:
- Format records to single text field
- Tokenize with RoBERTa
- Build labels for CLM
- Pad/stack into batches
- Save to cache (`cache0` example)

### 1.6 Source E: Wikipedia legal-domain crawling
Processor: `WikipediaPageProcessor`
- Extracts structured page text (`TITLE`, `SUMMARY`, `SECTION`)
- Filters non-content sections
- Cleans references and whitespace
- Uses internal links for iterative topic discovery

Iterative crawl loop:
- `initial_topic = 'Foreign relations of India'`
- `max_links_to_follow_per_page = 20`
- `target_data_size_gb = 1`
- `max_pages_to_process = 3000`
- Writes extracted pages as JSONL (e.g., `wikipedia_data_Prisons_in_India.jsonl`)
- Then processes that JSONL via `process_jsonl_smart(...)`

### 1.7 Specialized legal judgment QA formatting
Processor: `QueryAnswerContextProcessor`
- Accepts JSON/JSONL with fields like `case_name`, `judgment_date`, `question`, `answer`
- Prompt template:
  - `Case Name: ...`
  - `Judgment Date: ...`
  - `Question: ...`
  - `Answer:` + target answer
- Labels mask prompt tokens with `-100` (loss only on answer tokens)
- Batches by token budget

Pipeline wrapper: `process_jsonl_qa(...)`
- Example call in notebook: `process_jsonl_qa('/content/IndicLegalQA Dataset_10K.json')`

### 1.8 Cleaning functions
`clean_constitution_text(...)` and `clean_file(...)` are used to normalize legal text by removing:
- amendment/procedural footnote artifacts
- citation markers (`[1]`, `[citation needed]`)
- noisy unicode markers and excess whitespace

## 2) Model and training (`legal_llm_model.ipynb.ipynb`)

### 2.1 Model type
A custom decoder-only Transformer (`SimpleLLM`) with:
- RoBERTa-size vocab (`vocab_size=50265`)
- Hidden size `512`
- `8` decoder layers
- `8` attention heads, `4` KV heads (Grouped Query Attention)
- SwiGLU feed-forward blocks (`intermediate_size=2048`)
- Rotary positional embedding (`max_position_embeddings=2048`)
- RMSNorm + residual dropout
- Weight-tied output head
- Optional QK norm and gradient checkpointing enabled in config

### 2.2 Core modules defined
- `SimpleRotaryEmbedding`
- `GroupedQueryAttention`
- `SwiGLUFFN`
- `TransformerBlock`
- `MultiTransformerBlock`
- `EmbeddingLayer`
- `SimpleLLM`

### 2.3 Training configuration (notebook values)
`train_config` includes:
- checkpoint dir: `/content/drive/MyDrive/checkpoints`
- preload checkpoint: `epoch23`
- epochs: ``
- learning rate: `3e-4`
- AdamW betas: `(0.9, 0.97)`
- weight decay: `0.01`
- warmup steps: `1000`
- max grad norm: `1.0`
- AMP: enabled
- gradient accumulation steps: `4`
- selected cache files: `batch075.pt`, `batch076.pt`

### 2.4 Trainer flow (`SimpleLLMTrainer`)
- Loads only configured batch files (or all if not specified)
- Builds optimizer + linear warmup scheduler
- Supports mixed precision (`GradScaler`, `autocast`)
- Supports checkpoint resume/load
- Training loop computes and logs:
  - loss
  - token-level accuracy (ignoring `-100`)
  - perplexity
  - learning rate
  - GPU memory (if CUDA)
- Saves checkpoint each epoch and best model by validation loss

### 2.5 Inference in notebook
- Loads saved checkpoint (`llm_epoch24.pt`)
- Tokenizes query with `RobertaTokenizer`
- Runs forward pass
- Includes `generate_text(...)` for autoregressive generation with:
  - greedy next-token decode
  - optional EOS stopping
  - repetition penalty

## 3) End-to-end workflow summary
1. Collect corpora (Q&A text, PDFs->text, JSONL, HF dataset, Wikipedia crawl, judgment QA JSON/JSONL).
2. Normalize/clean legal text where needed.
3. Tokenize with RoBERTa tokenizer.
4. Build CLM labels (`-100` masking for ignore regions/padding).
5. Create and save `.pt` batch files via `SimpleBatchCache`.
6. Train `SimpleLLM` from cached batches with checkpointing and AMP.
7. Validate, save best weights, and run generation on legal prompts.

## 4) Minimal reproducibility checklist
- Python with `torch`, `transformers`, `datasets`, `tensorboard`, `tqdm`
- Optional data tools: `PyPDF2`, `wikipedia`, `wikipedia-api`, `torchinfo`
- Storage paths in notebooks are currently Colab/Drive-style (`/content/...`, `/content/drive/...`)
- Ensure cache directories exist and are consistent between preprocessing and training notebooks

## 5) Notes specific to this repository
- `legal_llm.py` is not present in the current project folder; the implemented workflow is in notebooks.
- Main workflow sources used for this README are:
  - `data_collection_preprocessing.ipynb`
  - `legal_llm_model.ipynb.ipynb`

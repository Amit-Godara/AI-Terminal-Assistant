# AI Terminal Assistant

An NLP-based Intelligent Terminal Command Assistant that understands natural-language instructions, identifies user intent, extracts relevant entities, generates appropriate Windows PowerShell commands, explains them, and executes approved commands safely.

## Architecture Overview

The system uses a modular, multi-stage hybrid NLP pipeline:
1. **Text Preprocessor**: Cleans, tokenizes, lemmatizes, and normalizes input while retaining critical entities such as paths, filenames, and extensions.
2. **Intent Classifier**: TF-IDF vectorization paired with Logistic Regression (with UNKNOWN handling) to identify the intended action across 12 core intent classes.
3. **Entity Extractor**: Rule-based and spaCy pattern extraction for file names, paths, extensions, keywords, version targets, and numeric parameters, preserving exact original casing and character offsets.
4. **Semantic Similarity Engine**: Sentence Transformers (`all-MiniLM-L6-v2`) with a robust TF-IDF cosine similarity fallback to identify paraphrases and handle low-confidence predictions.
5. **Command Generator**: Safe, template-based generation of Windows PowerShell commands with human-readable explanations.
6. **Command Validator & Safe Executor**: Strict command allowlisting, dangerous argument blocking, path validation, user confirmation gating (`[y/N]`), and timeout-protected subprocess execution (`shell=False`).
7. **Rich Terminal UI**: Formatted CLI providing real-time feedback, confidence scores, command diffs, and execution logs.

## Directory Structure

```text
AI-Terminal-Assistant/
├── app.py                      # Main application entry point
├── config.py                   # Central configuration & path resolution
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview & quickstart
├── DEVELOPMENT_LOG.md          # Chronological development log (Phases 1–9)
├── FINAL_PROJECT_REPORT.md     # Comprehensive academic & technical report
├── PRESENTATION_GUIDE.md       # 12-slide presentation structure & speaker notes
├── LIVE_DEMO_GUIDE.md          # Step-by-step live demonstration script
│
├── data/
│   ├── intents_dataset.csv     # 12-class labeled dataset (240 examples)
│   ├── create_dataset.py       # Dataset generator script
│   ├── validator.py            # Reusable dataset validation utility
│   ├── nltk_data/              # Local NLTK corpora & tokenizers
│   ├── command_history.json    # Persistent command execution history
│   └── manual_test_results.json# 18-case manual integration verification
│
├── models/                     # Saved model artifacts (.joblib, metadata, reports)
│   ├── intent_classifier.joblib# Trained Logistic Regression classifier
│   ├── tfidf_vectorizer.joblib # Fitted TF-IDF feature extractor
│   ├── model_metadata.json     # Configuration and hyperparameter metadata
│   ├── evaluation_report.json  # Held-out test metrics and confusion matrix
│   ├── confusion_matrix.png    # High-resolution confusion matrix plot
│   ├── reference_embeddings.joblib # Cached SentenceTransformer reference vectors
│   ├── semantic_evaluation_report.json # Semantic engine evaluation metrics
│   └── semantic_comparison_report.json # 36-query model comparison report
│
├── core/
│   ├── __init__.py
│   ├── preprocessor.py         # NLTK RegexpTokenizer, stopwords & lemmatization
│   ├── intent_classifier.py    # TF-IDF + Logistic Regression Classifier
│   ├── entity_extractor.py     # Hybrid spaCy EntityRuler & Regex Entity Extractor
│   ├── semantic_engine.py      # Sentence Transformers + Fallback
│   ├── command_generator.py    # Controlled PowerShell template builder
│   ├── command_validator.py    # Allowlist & safety checks
│   ├── executor.py             # Controlled subprocess execution
│   ├── project_detector.py     # Read-only software project & technology detector
│   └── project_analyzer.py     # Read-only project intelligence & manifest analyzer
│
├── utils/
│   ├── __init__.py
│   ├── terminal_ui.py          # Rich-based terminal UI
│   └── history.py              # Command history manager
│
├── tests/
│   ├── __init__.py
│   ├── test_initialization.py  # Initialization and import tests (4 tests)
│   ├── test_preprocessor.py    # Preprocessing & tokenization tests (12 tests)
│   ├── test_dataset.py         # Dataset validation & loading tests (4 tests)
│   ├── test_intent_classifier.py# Intent classifier & ML pipeline tests (12 tests)
│   ├── test_entity_extractor.py # Entity extraction & missing parameter tests (33 tests)
│   ├── test_semantic_engine.py  # Semantic similarity & hybrid inference tests (13 tests)
│   ├── test_command_generator.py# PowerShell command generation & safety tests (37 tests)
│   ├── test_command_validator.py# Security validation & path safety tests (18 tests)
│   ├── test_executor.py         # Controlled execution & safety tests (34 tests)
│   ├── test_terminal_integration.py # End-to-end integration & UI tests (18 tests)
│   ├── test_project_detector.py # Project detection & technology tests (25 tests)
│   └── test_project_analyzer.py # Project intelligence & manifest tests (20 tests)
│
└── scripts/
    ├── train_model.py          # Model training, evaluation & verification pipeline
    ├── evaluate_semantic.py    # Unseen paraphrase benchmark & model comparison
    └── manual_test_suite.py    # Automated 18-case manual end-to-end test suite
```

## NLP Entity Extraction (Phase 4)

### Purpose & Architecture
While intent classification identifies *what* operation the user wants to perform (e.g., `MOVE_FILE` or `DELETE_FILE`), **Entity Extraction** parses *what resources* the operation acts upon (e.g., which file, what source path, what target directory, what search keyword, or what runtime version).

The `EntityExtractor` (`core/entity_extractor.py`) combines:
1. **spaCy `EntityRuler`**: Controlled pattern matcher running on a blank English model (`spacy.blank("en")`) for canonical software tools and runtimes (`VERSION_NAME`: `Python`, `Node.js`, `git`, `Docker`, `PowerShell`, `npm`, `rust`, `gcc`, `go`, `java`, `dotnet`, `code`, `pip`).
2. **Deterministic Regular Expressions**: Tailored patterns capturing:
   - Windows absolute paths (`C:\Projects\report.pdf`, `C:\Users\Amit\My Documents`)
   - Relative paths (`.\subfolder`, `..\backup`)
   - Complete filenames with single or multiple dots (`notes.txt`, `annual_report.pdf`, `data_v1.0.final.csv`)
   - File extensions (`.py`, `.pdf`, `.csv`, `.json`, `*.json`)
   - Technology-to-extension mappings (`Python files` $\rightarrow$ `.py`, `markdown files` $\rightarrow$ `.md`)
   - Quoted entities (`'Client Assets'`, `"Financial Reports 2026"`)
   - Wildcard search patterns (`config_*.json` $\rightarrow$ keyword: `config`, extension: `.json`)
   - Standalone numeric parameters (`10`, `42`)
3. **Intent-Guided Context**:
   - `MOVE_FILE` / `COPY_FILE`: Automatically splits queries into `<source>` and `<destination>`.
   - `CREATE_DIRECTORY`: Resolves target directory names from `named`, `called`, or direct phrasing.
   - `CREATE_FILE`: Identifies full filename and extension.
   - `DELETE_FILE`: Identifies target file without capturing generic words.
   - `SEARCH_FILES`: Extracts search keywords and extensions.
   - `CHECK_VERSION`: Extracts software runtime without confusing tools like `Node.js` with `.js` files.
4. **Original Casing & Offset Preservation**:
   - Operates directly on the raw user input query string.
   - Preserves exact casing (`Report.PDF`, `ImportantData.XLSX`, `C:\Users\Amit\My Documents`) and exact character offsets (`start`, `end`).
5. **False Positive Prevention**:
   - Generic command filler (`file`, `folder`, `directory`, `path`, `contents`, `system`, `machine`) is protected from being mistakenly parsed as filenames or folder names.
6. **Missing Parameter Detection**:
   - Detects when required arguments are omitted (e.g. `Move report.pdf` without a destination) and outputs structured clarification requests for subsequent interactive prompts.

### Supported Entity Types

| Entity Type | Description | Sample Query & Extracted Value |
|---|---|---|
| `FILE_NAME` | Complete filename with extension | `Delete file notes.txt` $\rightarrow$ `notes.txt` |
| `FILE_EXTENSION` | Target file extension | `Find all Python files` $\rightarrow$ `.py` |
| `DIRECTORY_NAME` | Target folder or directory | `Create folder MachineLearning` $\rightarrow$ `MachineLearning` |
| `SOURCE_PATH` | Source file or folder path | `Move C:\Projects\report.pdf into D:\Backup` $\rightarrow$ `C:\Projects\report.pdf` |
| `DESTINATION_PATH` | Target destination directory | `Copy notes.txt to Documents` $\rightarrow$ `Documents` |
| `KEYWORD` | Search query or pattern | `Find files with pattern config_*.json` $\rightarrow$ `config`, `config_*.json` |
| `VERSION_NAME` | Software / runtime target | `Show installed Node.js version` $\rightarrow$ `Node.js` |
| `NUMBER` | Numeric argument | `Show the first 10 files` $\rightarrow$ `10` |

### Sample Extraction Results

```json
{
  "query": "Move Report.PDF to C:\\Users\\Amit\\My Documents",
  "intent": "MOVE_FILE",
  "entities": [
    {
      "entity_type": "FILE_NAME",
      "value": "Report.PDF",
      "start": 5,
      "end": 15,
      "confidence": 1.0,
      "context": "move_copy_source_filename"
    },
    {
      "entity_type": "DESTINATION_PATH",
      "value": "C:\\Users\\Amit\\My Documents",
      "start": 19,
      "end": 46,
      "confidence": 1.0,
      "context": "move_copy_destination"
    }
  ],
  "missing_parameters": [],
  "status": "SUCCESS"
}
```

## NLP Semantic Similarity & Hybrid Inference (Phase 5)

### Purpose & Architecture
Supervised classifiers (TF-IDF + Logistic Regression) excel at learning discriminative boundaries on known vocabulary, but can struggle when users phrase requests using completely new synonyms, questions, or paraphrased structures.

The **Semantic Similarity Engine** (`core/semantic_engine.py`) provides:
1. **Dense Sentence Transformers**: Uses `all-MiniLM-L6-v2` generating 384-dimensional dense semantic embeddings to calculate cosine similarity against reference intent utterances.
2. **Offline-First & Lazy Loading**: Uses cached local weights with fallback handlers to prevent remote network hangs in secure/sandboxed environments.
3. **Reference Embedding Caching**: Persists computed reference vectors to `models/reference_embeddings.joblib` for rapid subsequent initializations.
4. **No Test Data Leakage**: Reference utterances are strictly sampled from the 80% stratified training split ($N=192$), preserving the test split untouched.
5. **Resilient Sparse Fallback**: Built-in TF-IDF cosine similarity fallback engine (`force_fallback=True` or on missing dependencies) ensuring zero crashes.
6. **Intent Aggregation**: Supports `"top_k_mean"` (average of top 2 per intent), `"max"`, and `"mean"` candidate scoring.
7. **Hybrid Inference (`predict_hybrid`)**:
   - **Consensus (`HYBRID_ENSEMBLE`)**: High confidence agreement between supervised classifier and semantic engine.
   - **Paraphrase Rescue (`SEMANTIC_ENGINE` / `BOOSTED`)**: Rescues unseen queries where the classifier lacks n-grams.
   - **Disagreement Arbitration**: Prioritizes learned decision boundaries or high semantic similarity appropriately.
   - **Mutual Uncertainty (`UNKNOWN` / `FALLBACK`)**: Safely rejects out-of-scope/conversational queries.

### Comparative Evaluation Benchmark (36 Unseen Paraphrases)

| Model Architecture | Top-1 Accuracy | Top-3 Accuracy | Avg Latency |
|---|---|---|---|
| **SentenceTransformer (Dense)** | **94.44%** | **100.00%** | **13.7 ms** |
| **TF-IDF Cosine (Fallback)** | **55.56%** | **94.44%** | **2.6 ms** |
| **Logistic Regression (Baseline Classifier)** | **8.33%** | **27.78%** | **1.3 ms** |
| **Hybrid Model (Classifier + Semantic Engine)** | **91.67%** | **100.00%** | **14.6 ms** |

## Intelligent PowerShell Command Generation (Phase 6)

### Purpose & Architecture
The **Command Generator** (`core/command_generator.py`) bridges NLP intent predictions and extracted parameters into controlled, safe PowerShell command operations. It does **not** execute commands, but constructs structured representations and readable previews for downstream validation (Phase 7).

Key features:
1. **Pydantic Data Model (`CommandOperation`)**: Contains `operation_type`, `intent`, `arguments`, `powershell_preview`, `explanation`, `risk_level`, `requires_confirmation`, `missing_parameters`, `clarification_prompt`, `generation_status`, and `inference_metadata`.
2. **Controlled Operation Templates**:
   - `LIST_FILES`: `Get-ChildItem` (with `-LiteralPath` for directory, optional `-Force` for hidden items).
   - `CREATE_DIRECTORY`: `New-Item -ItemType Directory -Path "<directory>"`.
   - `DELETE_FILE`: `Remove-Item -LiteralPath "<file>"` (**HIGH** risk, requires confirmation, no `-Recurse` or `-Force`).
   - `SEARCH_FILES`: `Get-ChildItem -Path "<dir>" -Filter "<pattern>" -File` (optional `-Recurse`).
   - `CHECK_VERSION`: Strict allowlist for approved tools (`python`, `node`, `git`, `docker`, `powershell`, `npm`, `java`, `rust`, `go`, `gcc`, `dotnet`, `pip`, `code`).
   - `CURRENT_DIRECTORY`: `Get-Location`.
   - `MOVE_FILE`: `Move-Item -LiteralPath "<source>" -Destination "<destination>"`.
   - `COPY_FILE`: `Copy-Item -LiteralPath "<source>" -Destination "<destination>"`.
   - `SYSTEM_INFO`: `Get-ComputerInfo`, `Get-CimInstance Win32_OperatingSystem`, or `hostname`.
   - `NETWORK_INFO`: `ipconfig`, `Get-NetAdapter`, or `Get-NetTCPConnection`.
   - `CREATE_FILE`: `New-Item -ItemType File -Path "<filename>"`.
   - `UNKNOWN`: Produces no preview; returns `UNSUPPORTED_INTENT`.
3. **Safe Path Quoting & Injection Neutralization**:
   - All file and directory arguments use `-LiteralPath` where appropriate to prevent wildcard bracket injection (`[`, `]`, `*`, `?`).
   - Paths with spaces and internal quotes are safely escaped with backticks (`` `" ``).
4. **Missing Parameter Gating**:
   - Incomplete queries (e.g. `move report.pdf` without destination) are gated with `GenerationStatus.NEEDS_CLARIFICATION` and a human-readable prompt.
5. **Strict No-Execution Boundary**:
   - Command execution is deferred entirely to Phase 7.

## Safety Validation and Controlled Execution (Phase 7)

### Purpose & Architecture
Phase 7 introduces the security core of the AI Terminal Assistant: **Independent Safety Validation** (`core/command_validator.py`) and **Controlled Command Execution** (`core/executor.py`).

Crucially, the generator's `powershell_preview` string is **never** executed. All operations are strictly interpreted from the strongly-typed `CommandOperation` object and executed via Python standard libraries or fixed-argument subprocesses.

Key features:
1. **Independent Security Validation (`CommandValidator`)**:
   - **Path Traversal & Sandboxing**: Detects and blocks `..` path traversal. Confines operations within `WORKSPACE_DIR` (or blocks access to `PROTECTED_SYSTEM_DIRS` such as `C:\Windows`, `C:\Program Files`, `C:\ProgramData`).
   - **Filename Sanitization**: Rejects forbidden Windows filename characters (`:`, `*`, `?`, `"`, `<`, `>`, `|`) and control characters.
   - **Target Conflict Prevention**: Blocks silent overwrites on `CREATE_FILE`, `CREATE_DIR`, `MOVE_FILE`, and `COPY_FILE` if the target destination already exists (`CONFLICT_TARGET_EXISTS`).
   - **Allowlist Enforcement**: Strictly validates tool executables (`CHECK_VERSION`), system metrics (`SYSTEM_INFO`), and network commands (`NETWORK_INFO`).
2. **Controlled Execution Engine (`CommandExecutor`)**:
   - **Python Standard Library Filesystem Handlers**: Executes file operations natively using `pathlib.Path` (`iterdir`, `mkdir`, `unlink`, `glob`/`rglob`) and `shutil` (`move`, `copy2`, exclusive open `x`).
   - **No Shell Injection**: Subprocesses run with `shell=False`, fixed executable arrays (e.g. `["python", "--version"]`), and a strict `10` second timeout. `shell=True`, `os.system()`, and `eval()` are strictly banned.
   - **Two-Stage Confirmation Gating**:
     - `LOW` Risk: Read-only operations (`LIST_FILES`, `CURRENT_DIRECTORY`, `CHECK_VERSION`, `SYSTEM_INFO`, `NETWORK_INFO`, `SEARCH_FILES`) run automatically.
     - `MEDIUM` Risk: State-modifying operations (`CREATE_DIRECTORY`, `CREATE_FILE`, `COPY_FILE`, `MOVE_FILE`) require user approval.
     - `HIGH` Risk: Destructive operations (`DELETE_FILE`) require **double confirmation**: explicit approval prompt AND the exact confirmation token `"DELETE"`. Directory deletion via `DELETE_FILE` is completely blocked.
3. **Persistent Audit Logging**:
   - Logs every operation attempt (success, failure, cancellation) with timestamps, arguments, outputs, and status to `data/command_history.json`.

## Terminal UI & End-to-End Integration (Phase 8)

### Purpose & Architecture
Phase 8 integrates the complete multi-stage hybrid NLP pipeline into an interactive terminal application (`app.py`) powered by a Rich terminal user interface (`utils/terminal_ui.py`).

The end-to-end flow executes:
1. **Natural Language Input**: User enters a request via `$ask "<instruction>"` or direct natural language.
2. **Text Preprocessing**: Tokenizes, strips punctuation, and normalizes queries while preserving critical entities and paths.
3. **Intent Classification & Hybrid Inference**: Evaluates intent using the trained TF-IDF + Logistic Regression model and dense Sentence Transformer (`all-MiniLM-L6-v2`) semantic similarity engine.
4. **Entity Extraction**: Uses the spaCy EntityRuler and deterministic regex on the unmodified input string to extract target paths, file names, tools, keywords, and parameters.
5. **Command Generation**: Constructs a structured `CommandOperation` and generates an informational PowerShell syntax preview.
6. **Independent Safety Validation**: Validates arguments, verifies workspace sandboxing, detects traversal (`..`), checks system path protection, and flags target conflicts.
7. **Multi-Stage Confirmation**:
   - **LOW Risk**: Read-only operations (`LIST`, `SEARCH`, `CHECK_VERSION`, `SYSTEM_INFO`, `NETWORK_INFO`, `GET_LOCATION`) execute automatically.
   - **MEDIUM Risk**: State-modifying operations (`CREATE_FILE`, `CREATE_DIR`, `COPY`, `MOVE`) prompt for confirmation: `Execute this operation? [y/N]` (defaulting to NO).
   - **HIGH Risk**: Destructive file operations (`DELETE_FILE`) require two-stage authorization: approval prompt + exact token confirmation (`Type DELETE to permanently delete this file:`).
8. **Controlled Execution**: Executes operations using native Python standard library handlers (`pathlib`, `shutil`) or allowlisted subprocesses (`shell=False`, `timeout=10`). The PowerShell preview is **never** executed directly.
9. **Visual Rich UI Cards**: Formatted output displays confidence, inference method, extracted entities, preview syntax card, validation status, execution duration, and stdout/stderr.
10. **Persistent Audit History**: Appends execution records to `data/command_history.json`.

### Supported Terminal Commands

| Command | Description | Example |
|---|---|---|
| `$ask <request>` | Natural-language terminal command instruction | `$ask "list all files in workspace"` |
| `<request>` | Direct natural-language instruction without prefix | `show current directory` |
| `$history` | Display execution history and audit log in a Rich table | `$history` |
| `$clear` | Clear the terminal screen | `$clear` |
| `$help` | Display interactive command reference and safety policies | `$help` |
| `$exit` / `quit` | Exit the assistant cleanly without traceback | `$exit` |

## Project Detection Engine (Phase 10)

### Purpose & Architecture
Phase 10 introduces the **Project Detection Engine** (`core/project_detector.py`), an extensible, strictly read-only scanner that detects software technologies, build systems, package managers, frameworks, and monorepos from project directories.

Key capabilities:
- **8 Core Technology Stacks**: Python, JavaScript/TypeScript, Java, C/C++, Flutter/Dart, .NET, Docker/Infrastructure, and Git/Monorepo setups.
- **Manifest-Based Detection**: Recognizes `pyproject.toml`, `requirements.txt`, `package.json`, `pom.xml`, `build.gradle`, `CMakeLists.txt`, `pubspec.yaml`, `*.csproj`, `Dockerfile`, and more.
- **Strict Read-Only & Security Isolation**: Performs no execution, package installation, or file modification. Enforces symlink boundaries (skips symlinks resolving outside the root) and directory exclusions (`node_modules`, `.git`, `.venv`, `dist`, `build`, `target`).

## Project Intelligence Engine (Phase 11)

### Purpose & Architecture
Phase 11 extends the assistant with the **Project Intelligence Engine** (`core/project_analyzer.py`), which performs deep read-only inspection of project structure, configuration files, manifests, dependencies, scripts, entry points, and documentation.

Key capabilities:
- **Manifest Intelligence**: Parses project metadata, dependencies, devDependencies, and build configurations across Python (`pyproject.toml`, `requirements.txt`, `Pipfile`, `setup.cfg`, `setup.py` regex parsing), Node (`package.json`, `tsconfig.json`), Java (`pom.xml`, `build.gradle`), C/C++ (`CMakeLists.txt`, `Makefile`), Flutter (`pubspec.yaml`), .NET (`*.csproj`), and Docker (`Dockerfile`, `docker-compose.yml`).
- **Safe Script Extraction**: Extracts exact script definitions (e.g. `npm run dev`, `test`, `build`) strictly as data/strings without executing them.
- **Structure Discovery**: Detects source, test, and documentation directories, identifies entry points (`app.py`, `index.ts`, `main.py`), and constructs a depth-limited directory tree.
- **Safe README Analysis**: Extracts project descriptions, installation instructions, usage, and test commands from markdown/rst/txt files while sanitizing untrusted inputs against prompt injection.
- **Strict Secret Protection**: Never opens or reads `.env*`, `*secret*`, `*credential*`, `*.pem`, `*.key`, or SSH key files.

## Quickstart & Execution

### 1. Activate the Python Environment
```powershell
# In PowerShell (Windows)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 2. Launch the Interactive Assistant
```powershell
python app.py
```

### 3. Run a Single Command Directly
```powershell
python app.py --query "show current directory"
python app.py --query "check python version"
python app.py --query "list all files"
```

### 4. Run the Complete Test Suite (227 Passed, 3 Skipped Across All 11 Phases)
```powershell
python -m pytest tests/ -v
```

### 5. Run the End-to-End Manual Test Harness
```powershell
python scripts/manual_test_suite.py
```

### 6. Run the Semantic Benchmark
```powershell
python scripts/evaluate_semantic.py
```

## Project Documentation & Presentation Materials

Comprehensive technical and presentation deliverables are available in the project root:
- [`FINAL_PROJECT_REPORT.md`](file:///d:/AI%20Terminal%20Assistance/FINAL_PROJECT_REPORT.md): Complete 23-section technical report with Mermaid architecture diagrams, NLP algorithms, and experimental results.
- [`PRESENTATION_GUIDE.md`](file:///d:/AI%20Terminal%20Assistance/PRESENTATION_GUIDE.md): 12-slide presentation structure with slide bullet points, suggested graphics, and detailed speaker notes.
- [`LIVE_DEMO_GUIDE.md`](file:///d:/AI%20Terminal%20Assistance/LIVE_DEMO_GUIDE.md): Step-by-step 9-part live demonstration script with exact commands and expected terminal cards.
- [`DEVELOPMENT_LOG.md`](file:///d:/AI%20Terminal%20Assistance/DEVELOPMENT_LOG.md): Chronological engineering log across all 11 development phases.

## Security & Operational Limitations

- **Not an OS-Level Sandbox**: Controlled execution uses Python standard library functions and `subprocess.run(shell=False)` to prevent shell injection, but does not implement hardware virtualization or an operating-system level container.
- **Windows File System Symlinks**: On non-admin Windows accounts, symlink creation requires elevated rights (`SeCreateSymbolicLinkPrivilege`). The application enforces symlink validation, but the corresponding unit test is skipped on standard accounts.
- **Controlled Operation Scope**: Only operations corresponding to the 12 supported intent categories are executed. Arbitrary administrative scripts and unapproved utilities are blocked by design.


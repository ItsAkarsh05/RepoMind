<div align="center">
  <img src="https://via.placeholder.com/150" alt="RepoMind Logo" width="150"/>
  <h1>🧠 RepoMind</h1>
  <p><strong>A complete, context-aware AI coding assistant that helps you talk to, understand, and visualize your GitHub codebases.</strong></p>

  <p>
    <a href="https://github.com/yourusername/RepoMind/issues"><img alt="Issues" src="https://img.shields.io/github/issues/yourusername/RepoMind?style=for-the-badge&color=2563eb"></a>
    <a href="https://github.com/yourusername/RepoMind/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/yourusername/RepoMind?style=for-the-badge&color=2563eb"></a>
    <a href="https://github.com/yourusername/RepoMind/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/yourusername/RepoMind?style=for-the-badge&color=eab308"></a>
    <a href="https://github.com/yourusername/RepoMind/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/yourusername/RepoMind?style=for-the-badge&color=16a34a"></a>
  </p>

  <p>
    <em>Powered by Qwen 2.5 Coder, RAG (FAISS + LlamaIndex), and a Custom Visualization Pipeline.</em>
  </p>
</div>

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Graphic Demostrations](#-graphic-demonstrations)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API](#-api)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Overview

### What problem it solves
Navigating and understanding a large, unfamiliar, or legacy GitHub codebase can take hours or even days. Developers struggle to figure out where functions are defined, how dependencies trace back, and what the overall architecture looks like.

### What the project does
RepoMind clones any public GitHub repository, parses the entire AST (Abstract Syntax Tree), and embeds the code into a vector space using FAISS. You can then **chat directly with the code** and **generate graphical visualizations** (Structure, Dependencies, and Call Graphs) to instantly understand how it works.

### Who it is for
- **Software Engineers** joining a new project or onboarding
- **Code Reviewers** needing extra context on large pull requests
- **Open Source Contributors** trying to understand a massive repository

---

## 🌟 Features
- ✨ **RAG-Powered Chat**: Ask questions about your code, and the local LLM will scan your files to give you exact, grounded answers.
- 🧠 **Conversational Memory**: Ask follow-up questions! The bot remembers recent messages, so you can naturally converse about specific functions.
- 🗺️ **Architecture Graphs**: Visualize your codebase entirely. Includes:
  - Repository Directory Tree
  - Function Call Graphs (who calls whom)
  - Modular Dependency Tracking (imports)
- 🔒 **100% Local Privacy**: Runs entirely on your hardware via Ollama. No proprietary code is sent to external API providers.

---

## 🎨 Graphic Demonstrations

*(Replace these placeholder links with actual images and GIFs of your project)*

### 1. Chat Interface
> *A clean, modern chat interface where you can ask complex questions about the architecture.*
> 
> <img src="https://via.placeholder.com/800x400.png?text=Add+Screenshot+of+Streamlit+Chat+UI+Here" alt="Chat Interface Demo" width="100%"/>

### 2. Dependency Graph Visualization
> *A dynamic node-graph showing how files interact and import each other.*
> 
> <img src="https://via.placeholder.com/800x400.png?text=Add+Screenshot+of+Architecture+Graph+Here" alt="Graph Interface Demo" width="100%"/>

---

## 🛠️ Tech Stack

**Language:**
- Python 3.10+

**Frameworks & Libraries:**
- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **API Backend:** [Flask](https://flask.palletsprojects.com/)
- **LLM Engine:** [Ollama](https://ollama.com/) (Qwen 2.5 Coder)
- **RAG / Vector Database:** [LlamaIndex](https://www.llamaindex.ai/) & [FAISS](https://faiss.ai/)
- **Embeddings:** [Langchain](https://python.langchain.com/) (`HuggingFaceBgeEmbeddings`)
- **Visualizations:** [Graphviz](https://graphviz.org/)

---

## 🚀 Installation

### Prerequisites
- [Python 3.10+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/) (Must be installed and running in the background)
- **Git** (For cloning repositories)

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/RepoMind.git
cd RepoMind
```

2. **Create a Virtual Environment**
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Mac/Linux:
source .venv/bin/activate
```

3. **Install Dependencies**
```bash
# Recommended for standard CPU use or if you have CUDA fully configured:
pip install -r requirements.txt

# IF YOU HAVE AN NVIDIA GPU (for much faster FAISS embedding indexing):
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Start the Local LLM**
```bash
# In a separate terminal window, ensure Ollama is running, then pull the model
ollama run qwen2.5-coder:3b
```

---

## 💻 Usage

RepoMind consists of a **Frontend UI** and a **Backend Visualization API**. 

### 1. Running the Chat Interface (Streamlit)
To interact with the conversational AI and load GitHub repositories:

```bash
python -m streamlit run app.py
```
> **Tip:** Open `http://localhost:8501` to view your dashboard. Paste a GitHub URL, let it ingest, and ask *"Where is the core logic localized?"*

### 2. Running the Visualization Backend (Flask)
To expose raw JSON endpoints for code architecture (useful if extending the app to React Flow / D3.js):

```bash
# Run this in a separate terminal
python -m visualization.api
```
> **Tip:** The server boots on `http://localhost:5000`. 

---

## 🔌 API

You can ping the Flask API directly if you want raw analysis data. *(Make sure to replace `C:\path\to\repo` with a valid cloned path).*

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/repo/structure?repo_path=<path>` | Gets a hierarchical file tree directory. |
| `GET` | `/repo/call-graph?repo_path=<path>` | Returns node/edge data for function calls. |
| `GET` | `/repo/dependencies?repo_path=<path>` | Returns file import dependencies. |
| `GET` | `/chat/history` | Retrieves current session chat memory. |
| `POST`| `/chat/reset` | Clears conversation memory for a fresh start. |

---

## 📂 Project Structure

```text
RepoMind/
├── .venv/                     # Virtual Environment (ignored)
├── cloned_repos/              # Where target GitHub repos are downloaded
├── faiss_indices/             # FAISS Vector Embeddings storage
├── weights/                   # HuggingFace Embeddings local cache
├── rag_101/                   # Core RAG retrieval orchestration
│   └── retriever.py           
├── repo_ingestion/            # Pipeline to clone, chunk, and embed code
│   ├── chunker.py
│   ├── embedding_store.py
│   └── github_handler.py     
├── visualization/             # AST Parsing and Graph generation
│   ├── api.py                 # Flask server
│   ├── graph_builder.py
│   └── streamlit_viz.py       # Visual render functions for Streamlit
├── app.py                     # Main Streamlit Chat Interface
├── memory.py                  # Conversation state tracker
└── requirements.txt           # Python dependency locks
```

---

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

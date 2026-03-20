# RepoMind 🧠

RepoMind is a complete, context-aware AI coding assistant that helps you talk to, understand, and visualize your GitHub codebases. 

It is powered by **Qwen 2.5 Coder** (via Ollama), **RAG** (Retrieval-Augmented Generation using FAISS and LlamaIndex), and includes a custom **Codebase Visualization API** to reverse-engineer project architectures.

---

## 🌟 Features

1. **RAG-Powered Chat**: Ask questions about your code, and the LLM will scan your files to give you exact, grounded answers.
2. **Conversational Memory**: Ask follow-up questions! The bot remembers recent messages, so you can say *"Can you optimize this?"* or *"Explain it better"*.
3. **Architecture Graphs**: Visualize your codebase. The Flask API generates:
   - **Repository Structure** (Hierarchical folder/file tree)
   - **Call Graphs** (Which functions call which other functions)
   - **Dependency Graphs** (File-to-file import tracking)

---

## 🚀 Step-by-Step Setup Guide

Follow these steps exactly to get RepoMind running on your machine.

### Step 1: Create a Virtual Environment (Recommended)
It's best practice to use a virtual environment to keep dependencies isolated. Open a terminal in this repository folder and run:
```bash
python -m venv .venv
```
Activate it (Windows):
```bash
.venv\\Scripts\\activate
```
*(Mac/Linux: `source .venv/bin/activate`)*

### Step 2: Install Python Dependencies
With your virtual environment activated, install the required packages:
```bash
pip install -r requirements.txt
```

### Step 3: Install and Start Ollama
RepoMind uses a local LLM under the hood so your code never leaves your machine.
1. Download Ollama from [ollama.com](https://ollama.com/)
2. Open a terminal and download the Qwen 2.5 Coder 3B model:
   ```bash
   ollama run qwen2.5-coder:3b
   ```
*(Leave Ollama running in the background).*

---

## 💻 How to Run the App

RepoMind consists of two separate applications: a **Frontend UI** and a **Backend Visualization API**. You can run one or both depending on what you want to do.

### Running the Chat Interface (Streamlit)
To talk to your code, run the Streamlit app. Open a terminal in the repo folder:
```bash
python -m streamlit run app.py
```
*This will open `http://localhost:8501` in your browser.*

**How to use the UI:**
1. Paste a GitHub repository URL into the sidebar (e.g., `https://github.com/user/repo`).
2. Click **Start Ingesting**. Wait for it to clone, parse, and index the code.
3. Start chatting! Ask things like:
   - *"Where is the authentication logic?"*
   - *"What does the `utils.py` file do?"*
   - *"Can you rewrite that function to be faster?"*

### Running the Visualization API (Flask)
To get raw JSON graphs of your codebase architecture (perfect for building frontend DAG visualizations with D3.js or React Flow).

Open a **separate** terminal in the repo folder:
```bash
python -m visualization.api
```
*This will start the server on `http://localhost:5000`.*

**Available Endpoints:**
*(Replace `C:\path\to\repo` with pointing to an actual folder on your machine)*
- **File Structure:** `curl "http://localhost:5000/repo/structure?repo_path=C:\path\to\repo"`
- **Call Graph:** `curl "http://localhost:5000/repo/call-graph?repo_path=C:\path\to\repo"`
- **Dependencies:** `curl "http://localhost:5000/repo/dependencies?repo_path=C:\path\to\repo"`
- **Chat History:** `curl "http://localhost:5000/chat/history"`
- **Clear Chat:** `curl -X POST "http://localhost:5000/chat/reset"`

---

## 🧪 Running Tests

We have a robust test suite for the visualization and memory modules. To run them:
```bash
python -m pytest tests/ -v
```

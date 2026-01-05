---
title: AI Code Explainer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: src/streamlit_app.py
pinned: false
license: mit
tags:
  - code-analysis
  - ai
  - llm
  - groq
  - streamlit
  - education
short_description: AI-powered code analysis, explanation, and visualization
---

# 🧠 AI Code Explainer

**Analyze, understand, and improve your code with AI**

A production-ready tool that uses AI to explain code, analyze complexity, detect security issues, generate documentation, and visualize code flow.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)
![Groq](https://img.shields.io/badge/Powered%20by-Groq-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

### 📖 Code Explanation
- **High-Level Summary**: Understand what code does at a glance
- **Line-by-Line Walkthrough**: Step-by-step explanation of each line
- **ELI5 Mode**: Simple explanations with real-world analogies
- **Audience Levels**: Beginner, Intermediate, Expert

### 📊 Code Analysis
- **Complexity Analysis**: Time & Space complexity (Big O notation)
- **Security Scan**: Detect common vulnerabilities (SQL injection, hardcoded keys, etc.)
- **Best Practices Review**: PEP8 compliance, naming conventions, code smells

### 📈 Visualization
- **Flowchart Generation**: Visual representation of code logic
- **Dependency Graphs**: See which functions call which
- **Mermaid Diagrams**: Interactive, copy-ready diagrams

### 🔄 Code Improvement
- **Refactoring Suggestions**: Improve readability, performance, or maintainability
- **Before/After Comparison**: Side-by-side view of changes
- **Docstring Generation**: Auto-generate documentation in multiple styles

### 💬 Interactive Chat
- **Ask Questions**: "Why did you use a while loop here?"
- **Context-Aware**: Remembers the conversation history
- **Deep Understanding**: Get detailed answers about specific code sections

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com) (free tier available)

### Option 1: Run Locally with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/ai-code-explainer.git
cd ai-code-explainer

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Copy environment template and add your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the application
streamlit run src/streamlit_app.py
```

### Option 2: Run with Docker

```bash
# Clone the repository
git clone https://github.com/your-repo/ai-code-explainer.git
cd ai-code-explainer

# Copy environment template
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run with Docker Compose (Direct Mode - Default)
docker-compose up --build

# Access at http://localhost:7860
```

### Option 3: Deploy to Hugging Face Spaces

1. Fork this repository
2. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
3. Select "Docker" as the SDK
4. Add your `GROQ_API_KEY` as a secret in Space settings
5. Push to your Space repository

---

## 🏗️ Architecture

This project follows a **"Direct-First" Hybrid Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Code Explainer                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐         ┌─────────────────────────┐   │
│  │   Streamlit     │         │       Backend           │   │
│  │   Frontend      │         │                         │   │
│  │                 │  Direct │  ┌─────────────────┐    │   │
│  │  components.py  │─────────│─►│   services.py   │    │   │
│  │                 │  Import │  │  (Business Logic)│   │   │
│  │  streamlit_app  │         │  └─────────────────┘    │   │
│  │     .py         │         │           │             │   │
│  │                 │   OR    │           ▼             │   │
│  │                 │         │  ┌─────────────────┐    │   │
│  │                 │   HTTP  │  │    api.py       │    │   │
│  │                 │─────────│─►│   (FastAPI)     │    │   │
│  │                 │ (--mode │  └─────────────────┘    │   │
│  │                 │   api)  │                         │   │
│  └─────────────────┘         └─────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Direct Mode** (Default) | Frontend imports backend services directly | HF Spaces, local dev |
| **API Mode** | Frontend calls FastAPI endpoints via HTTP | Microservices, scaling |

### Directory Structure

```
ai-code-explainer/
├── .env.example           # Environment template
├── Dockerfile             # Streamlit container (HF Spaces)
├── Dockerfile.api         # FastAPI container (API mode)
├── docker-compose.yml     # Multi-container orchestration
├── pyproject.toml         # uv/pip project configuration
├── requirements.txt       # pip-compatible dependencies
├── README.md              # This file
└── src/
    ├── __init__.py
    ├── streamlit_app.py   # Main entry point
    ├── frontend/
    │   ├── __init__.py
    │   └── components.py  # Reusable UI components
    └── backend/
        ├── __init__.py
        ├── config.py      # Centralized configuration
        ├── services.py    # Business logic (AI interactions)
        └── api.py         # FastAPI endpoints
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | - | Your Groq API key |
| `GROQ_BASE_URL` | No | `https://api.groq.com` | API base URL (no `/openai/v1`) |
| `GROQ_MODEL_NAME` | No | `llama-3.3-70b-versatile` | AI model to use |
| `MAX_CODE_LINES` | No | `500` | Max lines to analyze |

### Available Models

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `llama-3.3-70b-versatile` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Complex code, detailed explanations |
| `llama-3.1-8b-instant` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Quick analysis, simple code |
| `llama-3.1-70b-versatile` | ⭐⭐⭐ | ⭐⭐⭐⭐ | Alternative to 3.3 |
| `mixtral-8x7b-32768` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Large files (32K context) |
| `gemma2-9b-it` | ⭐⭐⭐⭐ | ⭐⭐⭐ | Efficient, balanced |

---

## 🎯 Use Cases

### 🎓 For Students
- Understand complex algorithms with ELI5 explanations
- Learn from flowchart visualizations
- Get analogies that make concepts click

### 👨‍💻 For Developers
- Document legacy code with auto-generated docstrings
- Review code for best practices
- Refactor for better readability

### 🎯 For Interview Prep
- Analyze time/space complexity of solutions
- Get optimization suggestions
- Understand algorithmic patterns

### 🔒 For Security Review
- Detect common vulnerabilities
- Find hardcoded secrets
- Review for injection risks

---

## 🔧 API Mode

For microservices architecture or when you need to scale the backend separately:

```bash
# Start both API and Streamlit in API mode
docker-compose --profile api-mode up --build

# Or manually:
# Terminal 1: Start FastAPI
uvicorn src.backend.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit in API mode
streamlit run src/streamlit_app.py -- --mode api --api-url http://localhost:8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/explain` | POST | Generate code explanation |
| `/analyze-complexity` | POST | Analyze time/space complexity |
| `/check-security` | POST | Security vulnerability scan |
| `/review-practices` | POST | Best practices review |
| `/generate-docstring` | POST | Generate documentation |
| `/generate-flowchart` | POST | Generate Mermaid flowchart |
| `/refactor` | POST | Suggest refactoring |
| `/chat` | POST | Interactive Q&A |

API documentation available at `/docs` when running in API mode.

---

## 🛡️ Safety

- ✅ **No Code Execution**: This tool only *analyzes* code, never runs it
- ✅ **No Storage**: Code is sent to Groq's API but not stored permanently
- ✅ **Input Limits**: Configurable limits on code size to prevent abuse
- ✅ **API Key Protection**: Keys stored securely, never exposed in UI

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Groq](https://groq.com) for blazing-fast LLM inference
- [Streamlit](https://streamlit.io) for the amazing web framework
- [FastAPI](https://fastapi.tiangolo.com) for the robust API framework
- [Hugging Face](https://huggingface.co) for hosting and deployment

---

<div align="center">
  <p>Built with ❤️ using Streamlit and Groq AI</p>
  <p>
    <a href="https://github.com/your-repo/ai-code-explainer">GitHub</a> •
    <a href="https://huggingface.co/spaces/your-space/ai-code-explainer">Demo</a> •
    <a href="https://console.groq.com">Get API Key</a>
  </p>
</div>
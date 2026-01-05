"""
AI Code Explainer - Main Streamlit Application.

This is the main entry point for the AI Code Explainer application.
It supports two modes:
1. Direct Mode (DEFAULT): Imports backend services directly as Python modules.
2. API Mode: Communicates with backend via HTTP endpoints (use --mode api).

Usage:
    # Direct Mode (Default - recommended for HF Spaces)
    streamlit run src/streamlit_app.py
    
    # API Mode (requires running FastAPI server separately)
    streamlit run src/streamlit_app.py -- --mode api
"""

import sys
import argparse
from typing import Optional, List, Dict, Any
import requests

import streamlit as st

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="AI Code Explainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/ai-code-explainer",
        "Report a bug": "https://github.com/your-repo/ai-code-explainer/issues",
        "About": "AI Code Explainer - Analyze and understand code with AI",
    },
)


# =============================================================================
# Mode Detection and Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Code Explainer")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "api"],
        default="direct",
        help="Run mode: 'direct' (default) uses Python imports, 'api' uses HTTP endpoints",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API base URL for API mode",
    )
    
    # Handle Streamlit's argument passing (after --)
    try:
        # Find the -- separator
        if "--" in sys.argv:
            idx = sys.argv.index("--")
            args = parser.parse_args(sys.argv[idx + 1:])
        else:
            args = parser.parse_args([])
    except:
        args = parser.parse_args([])
    
    return args


# Parse arguments at module load
ARGS = parse_args()
IS_API_MODE = ARGS.mode == "api"
API_BASE_URL = ARGS.api_url


# =============================================================================
# Service Layer Abstraction
# =============================================================================

class ServiceAdapter:
    """
    Adapter class that abstracts the service layer.
    
    In Direct Mode, it imports and uses the service classes directly.
    In API Mode, it makes HTTP requests to the FastAPI backend.
    """
    
    def __init__(self, api_key: str = "", base_url: str = "", model_name: str = ""):
        """Initialize the service adapter."""
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self._service = None
    
    def _get_direct_service(self):
        """Get the direct service instance (lazy loading)."""
        if self._service is None:
            from src.backend.services import CodeExplainerService
            self._service = CodeExplainerService(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
            )
        return self._service
    
    def update_credentials(self, api_key: str, base_url: str, model_name: str):
        """Update credentials and reset service."""
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self._service = None
    
    def _api_call(self, endpoint: str, payload: dict) -> dict:
        """Make an API call in API mode."""
        headers = {"X-API-Key": self.api_key}
        response = requests.post(
            f"{API_BASE_URL}/{endpoint}",
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    
    def detect_language(self, code: str, filename: Optional[str] = None) -> str:
        """Detect programming language."""
        if IS_API_MODE:
            result = self._api_call("detect-language", {
                "code": code,
                "filename": filename,
            })
            return result["language"]
        else:
            service = self._get_direct_service()
            return service.detect_language(code, filename)
    
    def explain_code(
        self,
        code: str,
        language: Optional[str] = None,
        level: str = "high-level",
        audience: str = "intermediate",
    ) -> str:
        """Generate code explanation."""
        if IS_API_MODE:
            result = self._api_call("explain", {
                "code": code,
                "language": language,
                "level": level,
                "audience": audience,
            })
            return result["explanation"]
        else:
            from src.backend.services import ExplanationLevel, AudienceLevel
            service = self._get_direct_service()
            return service.explain_code(
                code,
                language,
                ExplanationLevel(level),
                AudienceLevel(audience),
            )
    
    def analyze_complexity(self, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Analyze time and space complexity."""
        if IS_API_MODE:
            result = self._api_call("analyze-complexity", {
                "code": code,
                "language": language,
            })
            return result
        else:
            service = self._get_direct_service()
            return service.analyze_complexity(code, language)
    
    def check_security(self, code: str, language: Optional[str] = None) -> List[str]:
        """Check for security vulnerabilities."""
        if IS_API_MODE:
            result = self._api_call("check-security", {
                "code": code,
                "language": language,
            })
            return result["issues"]
        else:
            service = self._get_direct_service()
            return service.check_security(code, language)
    
    def review_best_practices(self, code: str, language: Optional[str] = None) -> List[str]:
        """Review code for best practices."""
        if IS_API_MODE:
            result = self._api_call("review-practices", {
                "code": code,
                "language": language,
            })
            return result["suggestions"]
        else:
            service = self._get_direct_service()
            return service.review_best_practices(code, language)
    
    def generate_docstring(
        self,
        code: str,
        language: Optional[str] = None,
        style: str = "google",
    ) -> str:
        """Generate documentation strings."""
        if IS_API_MODE:
            result = self._api_call("generate-docstring", {
                "code": code,
                "language": language,
                "style": style,
            })
            return result["documented_code"]
        else:
            service = self._get_direct_service()
            return service.generate_docstring(code, language, style)
    
    def generate_flowchart(self, code: str, language: Optional[str] = None) -> str:
        """Generate Mermaid flowchart."""
        if IS_API_MODE:
            result = self._api_call("generate-flowchart", {
                "code": code,
                "language": language,
            })
            return result["mermaid_diagram"]
        else:
            service = self._get_direct_service()
            return service.generate_flowchart(code, language)
    
    def generate_dependency_graph(self, code: str, language: Optional[str] = None) -> str:
        """Generate dependency graph."""
        if IS_API_MODE:
            result = self._api_call("generate-dependency-graph", {
                "code": code,
                "language": language,
            })
            return result["mermaid_diagram"]
        else:
            service = self._get_direct_service()
            return service.generate_dependency_graph(code, language)
    
    def suggest_refactoring(
        self,
        code: str,
        language: Optional[str] = None,
        focus: str = "readability",
    ) -> tuple:
        """Suggest refactoring improvements."""
        if IS_API_MODE:
            result = self._api_call("refactor", {
                "code": code,
                "language": language,
                "focus": focus,
            })
            return result["explanation"], result["refactored_code"]
        else:
            service = self._get_direct_service()
            return service.suggest_refactoring(code, language, focus)
    
    def chat_about_code(
        self,
        code: str,
        question: str,
        history: List = None,
        language: Optional[str] = None,
    ) -> str:
        """Answer questions about code."""
        history = history or []
        
        if IS_API_MODE:
            result = self._api_call("chat", {
                "code": code,
                "question": question,
                "history": history,
                "language": language,
            })
            return result["answer"]
        else:
            from src.backend.services import ChatMessage
            service = self._get_direct_service()
            chat_history = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in history
            ]
            return service.chat_about_code(code, question, chat_history, language)


# =============================================================================
# Page Components
# =============================================================================

def page_explain():
    """Explain page - Main code explanation view."""
    from src.frontend.components import (
        render_code_input,
        render_explanation_output,
        render_complexity_output,
        render_security_output,
        render_best_practices_output,
        render_action_buttons,
        render_error,
        render_loading_spinner,
    )
    
    st.markdown("## 📖 Explain Code")
    st.caption("Paste code below and click Explain")
    
    # Code input
    code_data = render_code_input()
    code = code_data["code"]

    # Action buttons always visible right after input
    action = render_action_buttons(code)

    if not code.strip():
        return

    config = st.session_state.get("sidebar_config", {})
    if not config.get("api_key"):
        return

    service = ServiceAdapter(
        api_key=config.get("api_key", ""),
        base_url=config.get("base_url", ""),
        model_name=config.get("model_name", ""),
    )

    language = code_data.get("language")
    if not language:
        try:
            language = service.detect_language(code, code_data.get("filename"))
            st.session_state.detected_language = language
        except Exception:
            language = "unknown"

    st.caption(f"Detected language: {language}")
    
    if action == "explain":
        with render_loading_spinner("🧠 Analyzing code..."):
            try:
                explanation = service.explain_code(
                    code=code,
                    language=language,
                    level=config.get("explanation_level", "high-level"),
                    audience=config.get("audience", "intermediate"),
                )
                st.markdown("---")
                render_explanation_output(explanation, language)
            except Exception as e:
                render_error(e, "explanation")
    
    elif action == "analyze":
        with render_loading_spinner("📊 Analyzing complexity..."):
            try:
                # Complexity
                complexity = service.analyze_complexity(code, language)
                st.markdown("---")
                render_complexity_output(complexity)
                
                # Security (for auditor mode)
                if config.get("analysis_mode") == "auditor":
                    st.markdown("---")
                    security = service.check_security(code, language)
                    render_security_output(security)
                    
                    st.markdown("---")
                    practices = service.review_best_practices(code, language)
                    render_best_practices_output(practices)
            except Exception as e:
                render_error(e, "analysis")


def page_visualize():
    """Visualize page - Code flow visualization."""
    from src.frontend.components import (
        render_code_input,
        render_visualization,
        render_error,
        render_loading_spinner,
        render_api_key_warning,
    )
    
    st.markdown("## 📊 Visualize Code")
    st.caption("Generate flowcharts from your code")
    
    # Code input
    code_data = render_code_input()
    code = code_data["code"]

    has_code = bool(code.strip())
    if not has_code:
        st.info("👆 Paste code above, then generate a flowchart or graph.")

    has_key = render_api_key_warning()
    
    # Get service adapter
    config = st.session_state.get("sidebar_config", {})
    service = ServiceAdapter(
        api_key=config.get("api_key", ""),
        base_url=config.get("base_url", ""),
        model_name=config.get("model_name", ""),
    )
    
    language = code_data.get("language") or "unknown"

    st.markdown("---")
    
    # Visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "📈 Generate Flowchart",
            use_container_width=True,
            type="primary",
            disabled=not (has_code and has_key),
        ):
            with render_loading_spinner("🎨 Creating flowchart..."):
                try:
                    if language == "unknown":
                        language = service.detect_language(code)
                        st.session_state.detected_language = language
                    diagram = service.generate_flowchart(code, language)
                    st.session_state.flowchart = diagram
                except Exception as e:
                    render_error(e, "flowchart generation")
    
    with col2:
        if st.button(
            "🔗 Generate Dependency Graph",
            use_container_width=True,
            disabled=not (has_code and has_key),
        ):
            with render_loading_spinner("🔗 Mapping dependencies..."):
                try:
                    if language == "unknown":
                        language = service.detect_language(code)
                        st.session_state.detected_language = language
                    diagram = service.generate_dependency_graph(code, language)
                    st.session_state.dependency_graph = diagram
                except Exception as e:
                    render_error(e, "dependency graph generation")
    
    # Display visualizations
    if st.session_state.get("flowchart"):
        st.markdown("---")
        render_visualization(st.session_state.flowchart, "Code Flowchart")
    
    if st.session_state.get("dependency_graph"):
        st.markdown("---")
        render_visualization(st.session_state.dependency_graph, "Dependency Graph")


def page_refactor():
    """Refactor page - Code improvement suggestions."""
    from src.frontend.components import (
        render_code_input,
        render_refactoring_view,
        render_error,
        render_loading_spinner,
        render_api_key_warning,
    )
    
    st.markdown("## 🔄 Refactor Code")
    st.caption("Get improvement suggestions")
    
    # Code input
    code_data = render_code_input()
    code = code_data["code"]

    has_code = bool(code.strip())
    if not has_code:
        st.info("👆 Paste code above, then run refactoring.")

    has_key = render_api_key_warning()
    
    # Get service adapter
    config = st.session_state.get("sidebar_config", {})
    service = ServiceAdapter(
        api_key=config.get("api_key", ""),
        base_url=config.get("base_url", ""),
        model_name=config.get("model_name", ""),
    )
    
    language = code_data.get("language") or "unknown"

    st.markdown("---")
    
    # Refactoring options
    focus = st.radio(
        "Refactoring Focus",
        options=["readability", "performance", "maintainability"],
        format_func=lambda x: {
            "readability": "📖 Readability - Better naming and structure",
            "performance": "⚡ Performance - Optimize for speed",
            "maintainability": "🔧 Maintainability - Reduce duplication, improve modularity",
        }[x],
        horizontal=True,
    )
    
    if st.button(
        "🔄 Suggest Refactoring",
        use_container_width=True,
        type="primary",
        disabled=not (has_code and has_key),
    ):
        with render_loading_spinner("🔄 Analyzing code for improvements..."):
            try:
                if language == "unknown":
                    language = service.detect_language(code)
                    st.session_state.detected_language = language
                explanation, refactored = service.suggest_refactoring(code, language, focus)
                st.markdown("---")
                render_refactoring_view(code, refactored, explanation, language)
            except Exception as e:
                render_error(e, "refactoring")


def page_docstrings():
    """Docstrings page - Generate documentation."""
    from src.frontend.components import (
        render_code_input,
        render_docstring_output,
        render_error,
        render_loading_spinner,
        render_api_key_warning,
    )
    
    st.markdown("## 📝 Generate Docstrings")
    st.caption("Add documentation to your code")
    
    # Code input
    code_data = render_code_input()
    code = code_data["code"]

    has_code = bool(code.strip())
    if not has_code:
        st.info("👆 Paste code above, then generate docstrings.")

    has_key = render_api_key_warning()
    
    # Get service adapter
    config = st.session_state.get("sidebar_config", {})
    service = ServiceAdapter(
        api_key=config.get("api_key", ""),
        base_url=config.get("base_url", ""),
        model_name=config.get("model_name", ""),
    )
    
    language = code_data.get("language") or "python"

    st.markdown("---")
    
    # Documentation style
    style_options = {
        "google": "📋 Google Style",
        "numpy": "🔢 NumPy Style",
        "sphinx": "📚 Sphinx/RST Style",
        "jsdoc": "📄 JSDoc Style",
    }
    
    style = st.selectbox(
        "Documentation Style",
        options=list(style_options.keys()),
        format_func=lambda x: style_options[x],
    )
    
    if st.button(
        "📝 Generate Documentation",
        use_container_width=True,
        type="primary",
        disabled=not (has_code and has_key),
    ):
        with render_loading_spinner("📝 Generating documentation..."):
            try:
                if language == "python":
                    try:
                        language = service.detect_language(code)
                        st.session_state.detected_language = language
                    except Exception:
                        pass
                documented = service.generate_docstring(code, language, style)
                st.markdown("---")
                render_docstring_output(documented, language)
            except Exception as e:
                render_error(e, "docstring generation")


def page_chat():
    """Chat page - Interactive Q&A about code."""
    from src.frontend.components import (
        render_code_input,
        render_chat_interface,
        render_error,
        render_api_key_warning,
    )
    
    st.markdown("## 💬 Chat About Code")
    st.caption("Ask questions about your code")
    
    # Code input (collapsible when chat is active)
    with st.expander("📝 Code Context", expanded=not st.session_state.get("chat_history")):
        code_data = render_code_input()
        code = code_data["code"]
    
    if not code:
        st.info("👆 Enter some code above to start chatting!")
        return
    
    if not render_api_key_warning():
        return
    
    # Get service adapter
    config = st.session_state.get("sidebar_config", {})
    service = ServiceAdapter(
        api_key=config.get("api_key", ""),
        base_url=config.get("base_url", ""),
        model_name=config.get("model_name", ""),
    )
    
    # Detect language
    language = st.session_state.get("detected_language", "unknown")
    
    st.markdown("---")
    
    # Chat handler function
    def chat_handler(code: str, question: str, history: List) -> str:
        return service.chat_about_code(code, question, history, language)
    
    # Render chat interface
    render_chat_interface(code, chat_handler)


def page_home():
    """Home page - Welcome and quick start."""
    st.markdown("# 🧠 AI Code Explainer")
    st.markdown("**Understand any code in seconds**")
    
    # Quick start
    st.info(
        "👈 **Get Started:** Select a feature from the sidebar, paste your code, and click analyze!"
    )
    
    if not st.session_state.get("groq_api_key"):
        st.warning(
            "🔑 **First time?** Get a free API key at [console.groq.com](https://console.groq.com) "
            "and add it in Settings (sidebar)"
        )
    
    st.markdown("---")
    
    # Feature cards - simple and scannable
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📖 Explain")
        st.caption("Get plain-English explanations of any code")
        
        st.markdown("### 📊 Visualize")
        st.caption("Generate flowcharts and dependency graphs")
        
        st.markdown("### 📝 Docstrings")
        st.caption("Auto-generate documentation")
    
    with col2:
        st.markdown("### 🔄 Refactor")
        st.caption("Get improvement suggestions")
        
        st.markdown("### 💬 Chat")
        st.caption("Ask questions about your code")
        
        st.markdown("### 🔍 Analyze")
        st.caption("Complexity, security & best practices")
    
    st.markdown("---")
    st.caption("⚠️ This tool analyzes code but does NOT execute it. Your code is processed by Groq AI.")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Import sidebar component
    from src.frontend.components import render_sidebar, init_session_state
    
    # Initialize session state
    init_session_state()
    
    # =================================================================
    # SIDEBAR - Navigation FIRST, then Settings
    # =================================================================
    with st.sidebar:
        st.markdown("## 🧠 AI Code Explainer")
        
        # Navigation at TOP - visible immediately
        pages = {
            "🏠 Home": page_home,
            "📖 Explain": page_explain,
            "📊 Visualize": page_visualize,
            "🔄 Refactor": page_refactor,
            "📝 Docstrings": page_docstrings,
            "💬 Chat": page_chat,
        }
        
        selected_page = st.radio(
            "Navigate",
            options=list(pages.keys()),
            label_visibility="collapsed",
        )
        
        st.markdown("---")
    
    # Render settings below navigation
    sidebar_config = render_sidebar()
    st.session_state.sidebar_config = sidebar_config
    
    # Display mode indicator (API mode only)
    if IS_API_MODE:
        st.sidebar.caption(f"🌐 API Mode: {API_BASE_URL}")
    
    # Render selected page
    pages[selected_page]()


if __name__ == "__main__":
    main()
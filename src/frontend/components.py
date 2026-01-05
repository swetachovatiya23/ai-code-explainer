"""
Streamlit UI Components for AI Code Explainer.

This module contains reusable UI components for the Streamlit frontend,
including the sidebar configuration, code input areas, and output displays.
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Callable
import os

from src.backend.config import (
    get_settings,
    get_available_models,
    get_model_descriptions,
    get_explanation_levels,
    get_audience_levels,
    get_analysis_modes,
    get_supported_languages,
    get_file_extensions,
)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    
    settings = get_settings()
    
    # API Configuration
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = settings.GROQ_API_KEY or ""
    
    if "groq_base_url" not in st.session_state:
        st.session_state.groq_base_url = settings.GROQ_BASE_URL
    
    if "groq_model_name" not in st.session_state:
        st.session_state.groq_model_name = settings.GROQ_MODEL_NAME
    
    # User Preferences
    if "audience_level" not in st.session_state:
        st.session_state.audience_level = "intermediate"
    
    if "explanation_level" not in st.session_state:
        st.session_state.explanation_level = "high-level"
    
    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = "educational"
    
    if "output_style" not in st.session_state:
        st.session_state.output_style = "separate_report"
    
    # Code State
    if "current_code" not in st.session_state:
        st.session_state.current_code = ""
    
    if "detected_language" not in st.session_state:
        st.session_state.detected_language = ""
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Credentials saved indicator
    if "credentials_saved" not in st.session_state:
        st.session_state.credentials_saved = False


# =============================================================================
# Sidebar Components
# =============================================================================

def render_sidebar() -> Dict[str, Any]:
    """
    Render the configuration sidebar.
    
    Returns:
        Dictionary with all sidebar configuration values.
    """
    init_session_state()
    
    with st.sidebar:
        # -----------------------------------------------------------------
        # API Key - Simple check with status indicator
        # -----------------------------------------------------------------
        if st.session_state.groq_api_key:
            st.success("🔐 API Key configured")
        else:
            st.warning("🔑 API Key required")
        
        # -----------------------------------------------------------------
        # Settings (Collapsible - keeps UI clean)
        # -----------------------------------------------------------------
        with st.expander("⚙️ Settings", expanded=not st.session_state.groq_api_key):

            st.text_input(
                "Groq API Key",
                value=st.session_state.groq_api_key,
                key="groq_api_key",
                type="password",
                placeholder="gsk_...",
                help="Get a key at console.groq.com",
            )

            # Model Selection
            models = get_available_models()
            model_descriptions = get_model_descriptions()
            current_model_idx = (
                models.index(st.session_state.groq_model_name)
                if st.session_state.groq_model_name in models
                else 0
            )
            st.selectbox(
                "AI Model",
                options=models,
                index=current_model_idx,
                key="groq_model_name",
                format_func=lambda x: model_descriptions.get(x, x),
            )

            # Audience Level
            audience_levels = get_audience_levels()
            audience_options = [level["id"] for level in audience_levels]
            audience_labels = {
                level["id"]: f"{level['icon']} {level['name']}" for level in audience_levels
            }
            audience_idx = (
                audience_options.index(st.session_state.audience_level)
                if st.session_state.audience_level in audience_options
                else 0
            )
            st.selectbox(
                "Audience",
                options=audience_options,
                index=audience_idx,
                key="audience_level",
                format_func=lambda x: audience_labels.get(x, x),
            )

            # Analysis Mode
            analysis_modes = get_analysis_modes()
            mode_options = [mode["id"] for mode in analysis_modes]
            mode_labels = {mode["id"]: f"{mode['icon']} {mode['name']}" for mode in analysis_modes}
            mode_idx = (
                mode_options.index(st.session_state.analysis_mode)
                if st.session_state.analysis_mode in mode_options
                else 0
            )
            st.selectbox(
                "Mode",
                options=mode_options,
                index=mode_idx,
                key="analysis_mode",
                format_func=lambda x: mode_labels.get(x, x),
            )

            # Detail Level
            exp_levels = get_explanation_levels()
            exp_options = [level["id"] for level in exp_levels]
            exp_labels = {level["id"]: f"{level['icon']} {level['name']}" for level in exp_levels}
            exp_idx = (
                exp_options.index(st.session_state.explanation_level)
                if st.session_state.explanation_level in exp_options
                else 0
            )
            st.selectbox(
                "Detail Level",
                options=exp_options,
                index=exp_idx,
                key="explanation_level",
                format_func=lambda x: exp_labels.get(x, x),
            )

            col_save, col_clear = st.columns(2)
            with col_save:
                if st.button("💾 Save", use_container_width=True, type="primary"):
                    st.session_state.credentials_saved = True
                    st.success("✅ Saved!")
                    st.rerun()
            with col_clear:
                if st.button("🧹 Clear", use_container_width=True):
                    settings = get_settings()
                    st.session_state.groq_api_key = ""
                    st.session_state.groq_base_url = settings.GROQ_BASE_URL
                    st.session_state.groq_model_name = (
                        settings.GROQ_MODEL_NAME or get_available_models()[0]
                    )
                    st.session_state.credentials_saved = False
                    st.rerun()

        # -----------------------------------------------------------------
        # Advanced (Sibling expander - Streamlit forbids nested expanders)
        # -----------------------------------------------------------------
        with st.expander("🔧 Advanced", expanded=False):
            st.text_input(
                "API Base URL",
                value=st.session_state.groq_base_url,
                key="groq_base_url",
                placeholder="https://api.groq.com",
                help="Do not include '/openai/v1' (the app adds it when needed).",
            )
        
        # -----------------------------------------------------------------
        # Footer
        # -----------------------------------------------------------------
        st.markdown("---")
        st.caption("AI Code Explainer v1.0 • Powered by Groq")
    
    return {
        "api_key": st.session_state.groq_api_key,
        "base_url": st.session_state.groq_base_url,
        "model_name": st.session_state.groq_model_name,
        "audience": st.session_state.audience_level,
        "explanation_level": st.session_state.explanation_level,
        "analysis_mode": st.session_state.analysis_mode,
        "output_style": st.session_state.get("output_style", "separate_report"),
    }


# =============================================================================
# Code Input Components
# =============================================================================

def render_code_input() -> Dict[str, Any]:
    """
    Render the code input section with multiple input methods.
    
    Returns:
        Dictionary with code and metadata.
    """
    # Input method tabs - simple and clean
    tab1, tab2, tab3 = st.tabs(["✏️ Paste", "📁 Upload", "🔗 GitHub"])
    
    code = ""
    filename = None
    language = None
    
    with tab1:
        code = st.text_area(
            "Code",
            value=st.session_state.current_code,
            height=250,
            placeholder="# Paste your code here...",
            label_visibility="collapsed",
        )
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload",
            type=list(get_file_extensions().keys()),
            label_visibility="collapsed",
        )
        
        if uploaded_file is not None:
            try:
                code = uploaded_file.read().decode("utf-8")
                filename = uploaded_file.name
                st.success(f"✅ {filename}")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    with tab3:
        github_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/user/repo/blob/main/file.py",
            label_visibility="collapsed",
        )
        
        if st.button("Fetch", disabled=not github_url):
            with st.spinner("Fetching..."):
                try:
                    from src.backend.services import fetch_github_code
                    code, filename = fetch_github_code(github_url)
                    st.success(f"✅ {filename}")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
    
    # Update session state
    if code:
        st.session_state.current_code = code
    
    return {
        "code": code,
        "filename": filename,
        "language": language,
        "line_count": len(code.split('\n')) if code else 0,
    }


# =============================================================================
# Output Display Components
# =============================================================================

def render_explanation_output(
    explanation: str,
    language: str = "unknown",
    show_copy: bool = True,
) -> None:
    """
    Render the code explanation output.
    
    Args:
        explanation: The explanation text (markdown).
        language: Detected programming language.
        show_copy: Whether to show copy button.
    """
    st.markdown("### 📖 Explanation")
    
    # Language badge
    st.markdown(f"**Detected Language:** `{language}`")
    
    # Explanation content
    st.markdown(explanation)
    
    # Copy button
    if show_copy:
        if st.button("📋 Copy Explanation", key="copy_explanation"):
            st.write(
                f'<script>navigator.clipboard.writeText(`{explanation}`)</script>',
                unsafe_allow_html=True,
            )
            st.success("Copied to clipboard!")


def render_complexity_output(complexity_data: Dict[str, Any]) -> None:
    """
    Render complexity analysis output.
    
    Args:
        complexity_data: Dictionary with complexity analysis.
    """
    st.markdown("### 📊 Complexity Analysis")
    
    if "analysis" in complexity_data:
        st.markdown(complexity_data["analysis"])
    else:
        st.info("No complexity data available.")


def render_security_output(issues: List[str]) -> None:
    """
    Render security scan results.
    
    Args:
        issues: List of security issues found.
    """
    st.markdown("### 🔒 Security Scan")
    
    if not issues or (len(issues) == 1 and "no obvious" in issues[0].lower()):
        st.success("✅ No obvious security issues detected!")
    else:
        st.warning(f"⚠️ Found {len(issues)} potential issue(s):")
        for issue in issues:
            if issue.strip():
                st.markdown(f"- {issue}")


def render_best_practices_output(suggestions: List[str]) -> None:
    """
    Render best practices review results.
    
    Args:
        suggestions: List of best practice suggestions.
    """
    st.markdown("### ✨ Best Practices Review")
    
    if not suggestions:
        st.success("✅ Code follows best practices!")
    else:
        for suggestion in suggestions:
            if suggestion.strip():
                st.markdown(f"- {suggestion}")


def render_docstring_output(documented_code: str, language: str = "python") -> None:
    """
    Render generated docstrings.
    
    Args:
        documented_code: Code with generated docstrings.
        language: Programming language for syntax highlighting.
    """
    st.markdown("### 📝 Generated Documentation")
    
    st.code(documented_code, language=language)
    
    if st.button("📋 Copy Documented Code", key="copy_docstring"):
        st.success("Use the copy button in the code block above!")


def render_visualization(
    mermaid_code: str,
    title: str = "Code Visualization",
) -> None:
    """
    Render Mermaid diagram visualization.
    
    Args:
        mermaid_code: Mermaid diagram syntax.
        title: Title for the visualization.
    """
    st.markdown(f"### 📊 {title}")
    
    # Show mermaid diagram using streamlit-mermaid or fallback
    try:
        import streamlit_mermaid as stmd
        stmd.st_mermaid(mermaid_code)
    except ImportError:
        # Fallback: show the code with instructions
        st.info("💡 Install `streamlit-mermaid` for interactive diagrams")
        st.code(mermaid_code, language="mermaid")
        
        # Provide link to Mermaid Live Editor
        st.markdown(
            "[🔗 View in Mermaid Live Editor](https://mermaid.live/edit)"
        )
    
    # Show raw code in expander
    with st.expander("📄 View Mermaid Source"):
        st.code(mermaid_code, language="mermaid")


def render_refactoring_view(
    original_code: str,
    refactored_code: str,
    explanation: str,
    language: str = "python",
) -> None:
    """
    Render before/after refactoring comparison.
    
    Args:
        original_code: Original code.
        refactored_code: Refactored code.
        explanation: Explanation of changes.
        language: Programming language.
    """
    st.markdown("### 🔄 Refactoring Suggestions")
    
    # Explanation
    if explanation:
        with st.expander("📋 What Changed", expanded=True):
            st.markdown(explanation)
    
    # Side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Before (Original)**")
        st.code(original_code, language=language)
    
    with col2:
        st.markdown("**After (Refactored)**")
        st.code(refactored_code, language=language)
    
    # Copy refactored code
    if st.button("📋 Copy Refactored Code", key="copy_refactored"):
        st.success("Use the copy button in the code block above!")


def render_chat_interface(
    code: str,
    chat_handler: Callable[[str, str, List], str],
) -> None:
    """
    Render interactive chat interface for code Q&A.
    
    Args:
        code: The code being discussed.
        chat_handler: Function to handle chat messages.
    """
    st.markdown("### 💬 Ask About This Code")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about the code..."):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chat_handler(code, question, st.session_state.chat_history[:-1])
                    st.markdown(response)
                    
                    # Add assistant response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                    })
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# =============================================================================
# Error Display Components
# =============================================================================

def render_error(error: Exception, context: str = "") -> None:
    """
    Render a user-friendly error message.
    
    Args:
        error: The exception that occurred.
        context: Additional context about where the error occurred.
    """
    error_message = str(error)
    
    # Parse common error types for better messages
    if "API key" in error_message.lower() or "authentication" in error_message.lower():
        st.error(
            "🔑 **API Key Error**\n\n"
            "Please check your Groq API key in the sidebar settings. "
            "Get a key at [console.groq.com](https://console.groq.com)"
        )
    elif "rate limit" in error_message.lower():
        st.error(
            "⏱️ **Rate Limit Exceeded**\n\n"
            "You've made too many requests. Please wait a moment and try again."
        )
    elif "model" in error_message.lower():
        st.error(
            "🤖 **Model Error**\n\n"
            "There was an issue with the selected AI model. "
            "Try selecting a different model from the sidebar."
        )
    elif "connection" in error_message.lower() or "network" in error_message.lower():
        st.error(
            "🌐 **Connection Error**\n\n"
            "Unable to connect to the AI service. "
            "Please check your internet connection and try again."
        )
    else:
        st.error(
            f"❌ **Error{f' in {context}' if context else ''}**\n\n"
            f"{error_message}\n\n"
            "Please try again or check your settings."
        )


def render_api_key_warning() -> bool:
    """
    Check and render warning if API key is missing.
    
    Returns:
        True if API key is present, False otherwise.
    """
    if not st.session_state.get("groq_api_key"):
        st.warning(
            "🔑 **API Key Required**\n\n"
            "Please enter your Groq API key in the sidebar to use the AI features. "
            "Get a free key at [console.groq.com](https://console.groq.com)"
        )
        return False
    return True


# =============================================================================
# Loading States
# =============================================================================

def render_loading_spinner(message: str = "Analyzing code..."):
    """
    Context manager for loading spinner.
    
    Args:
        message: Loading message to display.
    
    Returns:
        Spinner context manager.
    """
    return st.spinner(message)


# =============================================================================
# Page Headers
# =============================================================================

def render_page_header(
    title: str,
    description: str = "",
    emoji: str = "🤖",
) -> None:
    """
    Render a consistent page header.
    
    Args:
        title: Page title.
        description: Optional description.
        emoji: Emoji to display.
    """
    st.markdown(f"# {emoji} {title}")
    
    if description:
        st.markdown(f"*{description}*")
    
    st.markdown("---")


# =============================================================================
# Action Buttons
# =============================================================================

def render_action_buttons(
    code: str,
    on_explain: Callable = None,
    on_analyze: Callable = None,
    on_visualize: Callable = None,
    on_docstring: Callable = None,
    on_refactor: Callable = None,
) -> str:
    """
    Render action buttons for code analysis.
    
    Args:
        code: The code to analyze.
        on_*: Callback functions for each action.
    
    Returns:
        The action that was clicked, or empty string.
    """
    has_code = bool(code.strip())
    has_key = bool(st.session_state.get("groq_api_key"))

    if not has_code:
        st.info("👆 Enter code above to get started")

    if has_code and not has_key:
        render_api_key_warning()
    
    st.markdown("---")
    
    # Simple action buttons
    col1, col2, col3 = st.columns(3)
    
    action = ""
    disabled = not (has_code and has_key)
    
    with col1:
        if st.button(
            "📖 Explain",
            use_container_width=True,
            type="primary",
            disabled=disabled,
        ):
            action = "explain"
    
    with col2:
        if st.button("📊 Analyze", use_container_width=True, disabled=disabled):
            action = "analyze"
    
    with col3:
        if st.button("🔄 Refactor", use_container_width=True, disabled=disabled):
            action = "refactor"
    
    return action

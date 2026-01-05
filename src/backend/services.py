"""
Business Logic Services for AI Code Explainer.

This module contains all the core business logic for code analysis,
explanation, visualization, and refactoring using the Groq API.

All functions are designed to be used directly (Direct Mode) or through
FastAPI endpoints (API Mode).
"""

import re
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from groq import Groq

from .config import (
    get_settings,
    get_file_extensions,
    get_supported_languages,
    Settings,
)


# =============================================================================
# Data Classes and Enums
# =============================================================================

class ExplanationLevel(str, Enum):
    """Code explanation granularity levels."""
    HIGH_LEVEL = "high-level"
    LINE_BY_LINE = "line-by-line"
    ELI5 = "eli5"


class AudienceLevel(str, Enum):
    """Target audience for explanations."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class AnalysisMode(str, Enum):
    """Analysis focus mode."""
    EDUCATIONAL = "educational"
    AUDITOR = "auditor"


@dataclass
class CodeAnalysisResult:
    """Result container for code analysis."""
    language: str
    explanation: str
    summary: Optional[str] = None
    complexity: Optional[Dict[str, str]] = None
    security_issues: Optional[List[str]] = None
    best_practices: Optional[List[str]] = None
    mermaid_diagram: Optional[str] = None
    docstring: Optional[str] = None
    refactored_code: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ChatMessage:
    """Chat message for interactive Q&A."""
    role: str  # "user" or "assistant"
    content: str


# =============================================================================
# Code Explainer Service Class
# =============================================================================

class CodeExplainerService:
    """
    Main service class for code explanation and analysis.
    
    This class encapsulates all AI-powered code analysis functionality.
    It can be used directly in Streamlit or through FastAPI endpoints.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the Code Explainer Service.
        
        Args:
            api_key: Groq API key. If not provided, uses settings.
            base_url: Groq base URL. If not provided, uses settings.
            model_name: Model to use. If not provided, uses settings.
        """
        settings = get_settings()
        
        self.api_key = api_key or settings.GROQ_API_KEY
        self.base_url = base_url or settings.GROQ_BASE_URL
        self.model_name = model_name or settings.GROQ_MODEL_NAME
        
        self._client: Optional[Groq] = None

    @staticmethod
    def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
        if not base_url:
            return None
        normalized = base_url.strip().rstrip("/")
        # Users sometimes paste the full Groq OpenAI-compatible endpoint.
        # Our app expects the base host URL only; the Groq SDK handles the rest.
        if normalized.endswith("/openai/v1"):
            normalized = normalized[: -len("/openai/v1")].rstrip("/")
        return normalized
    
    @property
    def client(self) -> Groq:
        """Get or create Groq client instance."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("Groq API key is required. Please provide it in settings.")
            self._client = Groq(
                api_key=self.api_key,
                base_url=self._normalize_base_url(self.base_url),
            )
        return self._client
    
    def update_credentials(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Update service credentials and reset client.
        
        Args:
            api_key: New API key.
            base_url: New base URL.
            model_name: New model name.
        """
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = self._normalize_base_url(base_url) or ""
        if model_name:
            self.model_name = model_name
        
        # Reset client to use new credentials
        self._client = None
    
    def detect_language(self, code: str, filename: Optional[str] = None) -> str:
        """
        Detect the programming language of the provided code.
        
        Args:
            code: Source code to analyze.
            filename: Optional filename for extension-based detection.
        
        Returns:
            Detected language name.
        """
        # First try extension-based detection
        if filename:
            ext = os.path.splitext(filename)[1].lower()
            ext_map = get_file_extensions()
            if ext in ext_map:
                return ext_map[ext]
        
        # Pattern-based detection
        patterns = {
            "python": [
                r"^import\s+\w+",
                r"^from\s+\w+\s+import",
                r"def\s+\w+\s*\(",
                r"class\s+\w+\s*[:\(]",
                r"if\s+__name__\s*==\s*['\"]__main__['\"]",
                r"print\s*\(",
            ],
            "javascript": [
                r"const\s+\w+\s*=",
                r"let\s+\w+\s*=",
                r"var\s+\w+\s*=",
                r"function\s+\w+\s*\(",
                r"=>\s*\{",
                r"require\s*\(['\"]",
                r"module\.exports",
                r"console\.log\s*\(",
            ],
            "typescript": [
                r":\s*(string|number|boolean|any)\s*[;=)]",
                r"interface\s+\w+",
                r"type\s+\w+\s*=",
                r"<\w+>",
            ],
            "java": [
                r"public\s+class\s+\w+",
                r"public\s+static\s+void\s+main",
                r"System\.out\.println",
                r"private\s+\w+\s+\w+;",
                r"@Override",
            ],
            "cpp": [
                r"#include\s*<\w+>",
                r"std::",
                r"cout\s*<<",
                r"cin\s*>>",
                r"int\s+main\s*\(",
                r"nullptr",
            ],
            "c": [
                r"#include\s*<\w+\.h>",
                r"printf\s*\(",
                r"scanf\s*\(",
                r"int\s+main\s*\(",
                r"malloc\s*\(",
            ],
            "csharp": [
                r"using\s+System",
                r"namespace\s+\w+",
                r"Console\.WriteLine",
                r"public\s+class\s+\w+",
            ],
            "go": [
                r"package\s+\w+",
                r"func\s+\w+\s*\(",
                r"import\s+\(",
                r"fmt\.\w+",
            ],
            "rust": [
                r"fn\s+\w+\s*\(",
                r"let\s+mut\s+\w+",
                r"impl\s+\w+",
                r"pub\s+fn",
                r"println!\s*\(",
            ],
            "ruby": [
                r"def\s+\w+",
                r"class\s+\w+",
                r"puts\s+",
                r"require\s+['\"]",
                r"end\s*$",
            ],
            "php": [
                r"<\?php",
                r"\$\w+\s*=",
                r"echo\s+",
                r"function\s+\w+\s*\(",
            ],
            "sql": [
                r"SELECT\s+.+\s+FROM",
                r"INSERT\s+INTO",
                r"UPDATE\s+\w+\s+SET",
                r"CREATE\s+TABLE",
                r"ALTER\s+TABLE",
            ],
            "bash": [
                r"#!/bin/bash",
                r"#!/bin/sh",
                r"\$\{\w+\}",
                r"echo\s+",
                r"if\s+\[\s+",
            ],
        }
        
        scores = {lang: 0 for lang in patterns}
        
        for lang, lang_patterns in patterns.items():
            for pattern in lang_patterns:
                if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                    scores[lang] += 1
        
        # Return language with highest score, or "unknown" if no matches
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        
        return "unknown"
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """
        Make a call to the Groq LLM.
        
        Args:
            system_prompt: System message for context.
            user_prompt: User message with the request.
            temperature: Sampling temperature (0-1).
        
        Returns:
            LLM response text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {str(e)}")
    
    def explain_code(
        self,
        code: str,
        language: Optional[str] = None,
        level: ExplanationLevel = ExplanationLevel.HIGH_LEVEL,
        audience: AudienceLevel = AudienceLevel.INTERMEDIATE,
    ) -> str:
        """
        Generate an explanation of the provided code.
        
        Args:
            code: Source code to explain.
            language: Programming language (auto-detected if not provided).
            level: Explanation granularity level.
            audience: Target audience level.
        
        Returns:
            Human-readable explanation of the code.
        """
        if not language:
            language = self.detect_language(code)
        
        audience_instructions = {
            AudienceLevel.BEGINNER: "Use simple terms, avoid jargon, and include analogies. Explain concepts as if teaching a complete beginner.",
            AudienceLevel.INTERMEDIATE: "Balance technical accuracy with accessibility. Use proper terminology but explain complex concepts.",
            AudienceLevel.EXPERT: "Focus on architecture, design patterns, and advanced concepts. Assume strong programming knowledge.",
        }
        
        level_instructions = {
            ExplanationLevel.HIGH_LEVEL: "Provide a concise summary of what this code does overall. Focus on the main purpose and functionality.",
            ExplanationLevel.LINE_BY_LINE: "Walk through the code step by step, explaining what each significant line or block does.",
            ExplanationLevel.ELI5: "Explain this code using simple analogies and everyday examples. Make it understandable to a 5-year-old.",
        }
        
        system_prompt = f"""You are an expert code explainer. Your task is to explain code clearly and accurately.

Language: {language}
Target Audience: {audience.value} - {audience_instructions[audience]}
Explanation Level: {level.value} - {level_instructions[level]}

Guidelines:
1. First, identify the programming language if not specified
2. Trace the execution path logically
3. Do NOT hallucinate functions or features that don't exist in the code
4. Use proper markdown formatting for better readability
5. Include relevant emojis to make explanations engaging"""

        user_prompt = f"""Please explain the following {language} code:

```{language}
{code}
```"""

        return self._call_llm(system_prompt, user_prompt)
    
    def analyze_complexity(self, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze time and space complexity of the code.
        
        Args:
            code: Source code to analyze.
            language: Programming language.
        
        Returns:
            Dictionary with complexity analysis.
        """
        if not language:
            language = self.detect_language(code)
        
        system_prompt = """You are an expert algorithm analyst. Analyze the time and space complexity of code.

Provide your analysis in the following JSON-like format:
- Time Complexity: O(?) with explanation
- Space Complexity: O(?) with explanation
- Key Observations: List of important algorithmic observations
- Optimization Suggestions: How to improve if applicable

Be precise and accurate. Base your analysis on actual loop structures, recursion, and data structures used."""

        user_prompt = f"""Analyze the complexity of this {language} code:

```{language}
{code}
```"""

        analysis = self._call_llm(system_prompt, user_prompt, temperature=0.2)
        
        return {
            "language": language,
            "analysis": analysis,
        }
    
    def check_security(self, code: str, language: Optional[str] = None) -> List[str]:
        """
        Perform basic security scan on the code.
        
        Args:
            code: Source code to scan.
            language: Programming language.
        
        Returns:
            List of potential security issues.
        """
        if not language:
            language = self.detect_language(code)
        
        system_prompt = """You are a security analyst specializing in code review. 
Scan code for common security vulnerabilities including:

1. SQL Injection patterns
2. Hardcoded credentials/API keys
3. Cross-Site Scripting (XSS) vulnerabilities
4. Insecure data handling
5. Command injection risks
6. Path traversal vulnerabilities
7. Insecure cryptographic practices

Format each issue as a clear, actionable item with severity level (HIGH/MEDIUM/LOW)."""

        user_prompt = f"""Scan this {language} code for security vulnerabilities:

```{language}
{code}
```

List any security concerns found, or state "No obvious security issues detected" if the code appears secure."""

        response = self._call_llm(system_prompt, user_prompt, temperature=0.1)
        
        # Parse response into list
        issues = [line.strip() for line in response.split('\n') if line.strip()]
        return issues
    
    def review_best_practices(
        self,
        code: str,
        language: Optional[str] = None,
    ) -> List[str]:
        """
        Review code for best practices compliance.
        
        Args:
            code: Source code to review.
            language: Programming language.
        
        Returns:
            List of best practice suggestions.
        """
        if not language:
            language = self.detect_language(code)
        
        language_specific = {
            "python": "PEP 8 compliance, pythonic idioms, type hints",
            "javascript": "ESLint rules, modern ES6+ syntax, async patterns",
            "java": "Java conventions, SOLID principles, proper exception handling",
            "cpp": "Modern C++ guidelines, memory safety, RAII patterns",
        }
        
        specific_guidelines = language_specific.get(
            language, 
            "language-specific conventions and common best practices"
        )
        
        system_prompt = f"""You are a senior code reviewer. Review code for best practices.

Focus on:
1. {specific_guidelines}
2. Variable and function naming conventions
3. Code organization and structure
4. Documentation and comments
5. Error handling patterns
6. Code readability and maintainability

Provide actionable suggestions for improvement."""

        user_prompt = f"""Review this {language} code for best practices:

```{language}
{code}
```"""

        response = self._call_llm(system_prompt, user_prompt, temperature=0.3)
        
        suggestions = [line.strip() for line in response.split('\n') if line.strip()]
        return suggestions
    
    def generate_docstring(
        self,
        code: str,
        language: Optional[str] = None,
        style: str = "google",
    ) -> str:
        """
        Generate documentation strings for the code.
        
        Args:
            code: Source code to document.
            language: Programming language.
            style: Documentation style (google, numpy, sphinx for Python; jsdoc for JS).
        
        Returns:
            Code with generated docstrings.
        """
        if not language:
            language = self.detect_language(code)
        
        style_instructions = {
            "google": "Use Google-style docstrings with Args, Returns, Raises sections",
            "numpy": "Use NumPy-style docstrings with Parameters, Returns sections",
            "sphinx": "Use Sphinx/reStructuredText style docstrings",
            "jsdoc": "Use JSDoc style documentation with @param, @returns tags",
        }
        
        instruction = style_instructions.get(style, style_instructions["google"])
        
        system_prompt = f"""You are a documentation expert. Generate comprehensive documentation for code.

Style: {instruction}

Guidelines:
1. Add docstrings to all functions, classes, and modules
2. Describe parameters with types and descriptions
3. Document return values
4. Note any exceptions that may be raised
5. Include brief usage examples where helpful
6. Return the COMPLETE code with docstrings added"""

        user_prompt = f"""Add documentation to this {language} code:

```{language}
{code}
```

Return the complete code with docstrings added."""

        return self._call_llm(system_prompt, user_prompt, temperature=0.2)
    
    def generate_flowchart(
        self,
        code: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Generate a Mermaid flowchart diagram for the code logic.
        
        Args:
            code: Source code to visualize.
            language: Programming language.
        
        Returns:
            Mermaid diagram syntax.
        """
        if not language:
            language = self.detect_language(code)
        
        system_prompt = """You are an expert at creating flowcharts from code.
Generate Mermaid.js flowchart syntax that visualizes the code's logic.

Rules:
1. Use proper Mermaid flowchart syntax (flowchart TD or flowchart LR)
2. Include decision nodes for conditionals (if/else)
3. Show loops clearly
4. Label edges where helpful
5. Keep the diagram readable and not overly complex
6. Return ONLY the Mermaid code block, no additional explanation

Example format:
```mermaid
flowchart TD
    A[Start] --> B{Condition?}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
```"""

        user_prompt = f"""Create a Mermaid flowchart for this {language} code:

```{language}
{code}
```

Return only the Mermaid diagram code."""

        response = self._call_llm(system_prompt, user_prompt, temperature=0.2)
        
        # Extract mermaid code if wrapped in code block
        if "```mermaid" in response:
            match = re.search(r"```mermaid\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return response.strip()
    
    def generate_dependency_graph(
        self,
        code: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Generate a dependency graph showing function relationships.
        
        Args:
            code: Source code to analyze.
            language: Programming language.
        
        Returns:
            Mermaid diagram syntax for dependency graph.
        """
        if not language:
            language = self.detect_language(code)
        
        system_prompt = """You are an expert at analyzing code structure.
Generate a Mermaid.js graph showing function/class dependencies.

Rules:
1. Identify all functions, methods, and classes
2. Show which functions call which other functions
3. Use appropriate shapes (rectangles for functions, hexagons for classes)
4. Show clear directional relationships
5. Return ONLY the Mermaid code block

Example format:
```mermaid
graph TD
    A[main] --> B[helper_function]
    A --> C[process_data]
    C --> D[validate_input]
```"""

        user_prompt = f"""Analyze the dependencies in this {language} code:

```{language}
{code}
```

Return a Mermaid graph showing function/class dependencies."""

        response = self._call_llm(system_prompt, user_prompt, temperature=0.2)
        
        # Extract mermaid code if wrapped in code block
        if "```mermaid" in response:
            match = re.search(r"```mermaid\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return response.strip()
    
    def suggest_refactoring(
        self,
        code: str,
        language: Optional[str] = None,
        focus: str = "readability",
    ) -> Tuple[str, str]:
        """
        Suggest refactoring improvements for the code.
        
        Args:
            code: Source code to refactor.
            language: Programming language.
            focus: Focus area (readability, performance, maintainability).
        
        Returns:
            Tuple of (explanation, refactored_code).
        """
        if not language:
            language = self.detect_language(code)
        
        focus_instructions = {
            "readability": "Improve code readability, naming, and structure",
            "performance": "Optimize for better performance and efficiency",
            "maintainability": "Improve modularity, reduce duplication, enhance testability",
        }
        
        instruction = focus_instructions.get(focus, focus_instructions["readability"])
        
        system_prompt = f"""You are an expert code refactoring assistant.
Your goal: {instruction}

Guidelines:
1. Explain what changes you're making and why
2. Preserve the original functionality exactly
3. Apply best practices for the language
4. Format the refactored code properly

Respond in this format:
## Refactoring Explanation
[Your explanation of changes]

## Refactored Code
```{language}
[Refactored code here]
```"""

        user_prompt = f"""Refactor this {language} code with focus on {focus}:

```{language}
{code}
```"""

        response = self._call_llm(system_prompt, user_prompt, temperature=0.3)
        
        # Parse response
        explanation = ""
        refactored = ""
        
        if "## Refactoring Explanation" in response:
            parts = response.split("## Refactored Code")
            if len(parts) == 2:
                explanation = parts[0].replace("## Refactoring Explanation", "").strip()
                code_part = parts[1]
                match = re.search(r"```\w*\s*(.*?)\s*```", code_part, re.DOTALL)
                if match:
                    refactored = match.group(1).strip()
        
        if not refactored:
            # Try to extract code block from anywhere in response
            match = re.search(r"```\w*\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                refactored = match.group(1).strip()
                explanation = response.split("```")[0].strip()
        
        return explanation, refactored if refactored else code
    
    def chat_about_code(
        self,
        code: str,
        question: str,
        history: List[ChatMessage] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Answer questions about the code in a conversational manner.
        
        Args:
            code: Source code being discussed.
            question: User's question about the code.
            history: Previous conversation messages.
            language: Programming language.
        
        Returns:
            Answer to the user's question.
        """
        if not language:
            language = self.detect_language(code)
        
        history = history or []
        
        system_prompt = f"""You are a helpful coding assistant discussing a specific piece of code.
You have expertise in {language} and can answer questions about the code's logic, 
design choices, and potential improvements.

The user is asking about this {language} code:
```{language}
{code}
```

Answer questions accurately and helpfully. If something is not clear from the code,
say so rather than making assumptions."""

        # Build messages with history
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": question})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.4,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Chat failed: {str(e)}")
    
    def full_analysis(
        self,
        code: str,
        language: Optional[str] = None,
        mode: AnalysisMode = AnalysisMode.EDUCATIONAL,
        audience: AudienceLevel = AudienceLevel.INTERMEDIATE,
    ) -> CodeAnalysisResult:
        """
        Perform comprehensive code analysis.
        
        Args:
            code: Source code to analyze.
            language: Programming language.
            mode: Analysis mode (educational or auditor).
            audience: Target audience level.
        
        Returns:
            CodeAnalysisResult with all analysis components.
        """
        if not language:
            language = self.detect_language(code)
        
        try:
            # Get explanation based on mode
            level = (
                ExplanationLevel.ELI5 
                if mode == AnalysisMode.EDUCATIONAL and audience == AudienceLevel.BEGINNER
                else ExplanationLevel.HIGH_LEVEL
            )
            
            explanation = self.explain_code(code, language, level, audience)
            
            result = CodeAnalysisResult(
                language=language,
                explanation=explanation,
            )
            
            # Add complexity analysis
            complexity = self.analyze_complexity(code, language)
            result.complexity = complexity
            
            # For auditor mode, add security and best practices
            if mode == AnalysisMode.AUDITOR:
                result.security_issues = self.check_security(code, language)
                result.best_practices = self.review_best_practices(code, language)
            
            return result
            
        except Exception as e:
            return CodeAnalysisResult(
                language=language,
                explanation="",
                error=str(e),
            )


# =============================================================================
# Standalone Functions (for backwards compatibility and direct imports)
# =============================================================================

def get_service(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> CodeExplainerService:
    """
    Get a configured CodeExplainerService instance.
    
    Args:
        api_key: Groq API key.
        base_url: Groq base URL.
        model_name: Model to use.
    
    Returns:
        Configured service instance.
    """
    return CodeExplainerService(api_key, base_url, model_name)


def explain_code(
    code: str,
    language: Optional[str] = None,
    level: str = "high-level",
    audience: str = "intermediate",
    api_key: Optional[str] = None,
) -> str:
    """
    Standalone function to explain code.
    
    Args:
        code: Source code to explain.
        language: Programming language.
        level: Explanation level (high-level, line-by-line, eli5).
        audience: Target audience (beginner, intermediate, expert).
        api_key: Groq API key.
    
    Returns:
        Code explanation.
    """
    service = get_service(api_key=api_key)
    return service.explain_code(
        code,
        language,
        ExplanationLevel(level),
        AudienceLevel(audience),
    )


def detect_language(code: str, filename: Optional[str] = None) -> str:
    """
    Detect the programming language of code.
    
    Args:
        code: Source code.
        filename: Optional filename.
    
    Returns:
        Detected language name.
    """
    service = CodeExplainerService()
    return service.detect_language(code, filename)


def fetch_github_code(url: str) -> Tuple[str, str]:
    """
    Fetch code from a GitHub URL.
    
    Args:
        url: GitHub URL (raw or regular).
    
    Returns:
        Tuple of (code_content, filename).
    """
    import requests
    
    # Convert regular GitHub URL to raw URL if needed
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com")
        url = url.replace("/blob/", "/")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Extract filename from URL
        filename = url.split("/")[-1]
        
        return response.text, filename
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch code from GitHub: {str(e)}")

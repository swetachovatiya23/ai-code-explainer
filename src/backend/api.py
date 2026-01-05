"""
FastAPI Application for AI Code Explainer API Mode.

This module provides REST API endpoints for all code analysis functionality.
Use this when running in API mode (--mode api).
"""

from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings, get_available_models, Settings
from .services import (
    CodeExplainerService,
    ExplanationLevel,
    AudienceLevel,
    AnalysisMode,
    ChatMessage,
    fetch_github_code,
)


# =============================================================================
# FastAPI App Configuration
# =============================================================================

app = FastAPI(
    title="AI Code Explainer API",
    description="API for analyzing, explaining, and visualizing source code using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class ExplainRequest(BaseModel):
    """Request model for code explanation."""
    code: str = Field(..., description="Source code to explain")
    language: Optional[str] = Field(None, description="Programming language (auto-detected if not provided)")
    level: str = Field("high-level", description="Explanation level: high-level, line-by-line, eli5")
    audience: str = Field("intermediate", description="Target audience: beginner, intermediate, expert")


class ExplainResponse(BaseModel):
    """Response model for code explanation."""
    language: str
    explanation: str


class ComplexityRequest(BaseModel):
    """Request model for complexity analysis."""
    code: str = Field(..., description="Source code to analyze")
    language: Optional[str] = Field(None, description="Programming language")


class ComplexityResponse(BaseModel):
    """Response model for complexity analysis."""
    language: str
    analysis: str


class SecurityRequest(BaseModel):
    """Request model for security scan."""
    code: str = Field(..., description="Source code to scan")
    language: Optional[str] = Field(None, description="Programming language")


class SecurityResponse(BaseModel):
    """Response model for security scan."""
    issues: List[str]


class BestPracticesRequest(BaseModel):
    """Request model for best practices review."""
    code: str = Field(..., description="Source code to review")
    language: Optional[str] = Field(None, description="Programming language")


class BestPracticesResponse(BaseModel):
    """Response model for best practices review."""
    suggestions: List[str]


class DocstringRequest(BaseModel):
    """Request model for docstring generation."""
    code: str = Field(..., description="Source code to document")
    language: Optional[str] = Field(None, description="Programming language")
    style: str = Field("google", description="Documentation style: google, numpy, sphinx, jsdoc")


class DocstringResponse(BaseModel):
    """Response model for docstring generation."""
    documented_code: str


class FlowchartRequest(BaseModel):
    """Request model for flowchart generation."""
    code: str = Field(..., description="Source code to visualize")
    language: Optional[str] = Field(None, description="Programming language")


class FlowchartResponse(BaseModel):
    """Response model for flowchart generation."""
    mermaid_diagram: str


class DependencyGraphRequest(BaseModel):
    """Request model for dependency graph generation."""
    code: str = Field(..., description="Source code to analyze")
    language: Optional[str] = Field(None, description="Programming language")


class DependencyGraphResponse(BaseModel):
    """Response model for dependency graph generation."""
    mermaid_diagram: str


class RefactorRequest(BaseModel):
    """Request model for refactoring suggestions."""
    code: str = Field(..., description="Source code to refactor")
    language: Optional[str] = Field(None, description="Programming language")
    focus: str = Field("readability", description="Focus: readability, performance, maintainability")


class RefactorResponse(BaseModel):
    """Response model for refactoring suggestions."""
    explanation: str
    refactored_code: str


class ChatRequest(BaseModel):
    """Request model for chat about code."""
    code: str = Field(..., description="Source code being discussed")
    question: str = Field(..., description="User's question")
    history: Optional[List[dict]] = Field(None, description="Conversation history")
    language: Optional[str] = Field(None, description="Programming language")


class ChatResponse(BaseModel):
    """Response model for chat."""
    answer: str


class GitHubRequest(BaseModel):
    """Request model for GitHub code fetch."""
    url: str = Field(..., description="GitHub URL to fetch code from")


class GitHubResponse(BaseModel):
    """Response model for GitHub fetch."""
    code: str
    filename: str


class LanguageDetectRequest(BaseModel):
    """Request model for language detection."""
    code: str = Field(..., description="Source code")
    filename: Optional[str] = Field(None, description="Optional filename for detection")


class LanguageDetectResponse(BaseModel):
    """Response model for language detection."""
    language: str


class FullAnalysisRequest(BaseModel):
    """Request model for full code analysis."""
    code: str = Field(..., description="Source code to analyze")
    language: Optional[str] = Field(None, description="Programming language")
    mode: str = Field("educational", description="Analysis mode: educational, auditor")
    audience: str = Field("intermediate", description="Target audience: beginner, intermediate, expert")


class FullAnalysisResponse(BaseModel):
    """Response model for full analysis."""
    language: str
    explanation: str
    complexity: Optional[dict] = None
    security_issues: Optional[List[str]] = None
    best_practices: Optional[List[str]] = None
    error: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response model for available models."""
    models: List[str]
    default: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str


# =============================================================================
# Dependencies
# =============================================================================

def get_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Extract API key from header or use environment variable.
    
    Args:
        x_api_key: API key from X-API-Key header.
    
    Returns:
        API key to use.
    
    Raises:
        HTTPException: If no API key is available.
    """
    settings = get_settings()
    api_key = x_api_key or settings.GROQ_API_KEY
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header or set GROQ_API_KEY environment variable."
        )
    
    return api_key


def get_service(api_key: str = Depends(get_api_key)) -> CodeExplainerService:
    """
    Get configured service instance.
    
    Args:
        api_key: API key from dependency.
    
    Returns:
        Configured CodeExplainerService.
    """
    return CodeExplainerService(api_key=api_key)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """Get list of available AI models."""
    models = get_available_models()
    settings = get_settings()
    return ModelsResponse(
        models=models,
        default=settings.GROQ_MODEL_NAME,
    )


@app.post("/detect-language", response_model=LanguageDetectResponse)
async def detect_language(
    request: LanguageDetectRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Detect programming language of code."""
    try:
        language = service.detect_language(request.code, request.filename)
        return LanguageDetectResponse(language=language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
async def explain_code(
    request: ExplainRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Generate explanation for code."""
    try:
        explanation = service.explain_code(
            code=request.code,
            language=request.language,
            level=ExplanationLevel(request.level),
            audience=AudienceLevel(request.audience),
        )
        language = request.language or service.detect_language(request.code)
        return ExplainResponse(
            language=language,
            explanation=explanation,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-complexity", response_model=ComplexityResponse)
async def analyze_complexity(
    request: ComplexityRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Analyze time and space complexity."""
    try:
        result = service.analyze_complexity(request.code, request.language)
        return ComplexityResponse(
            language=result["language"],
            analysis=result["analysis"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check-security", response_model=SecurityResponse)
async def check_security(
    request: SecurityRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Scan code for security vulnerabilities."""
    try:
        issues = service.check_security(request.code, request.language)
        return SecurityResponse(issues=issues)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review-practices", response_model=BestPracticesResponse)
async def review_best_practices(
    request: BestPracticesRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Review code for best practices."""
    try:
        suggestions = service.review_best_practices(request.code, request.language)
        return BestPracticesResponse(suggestions=suggestions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-docstring", response_model=DocstringResponse)
async def generate_docstring(
    request: DocstringRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Generate documentation strings for code."""
    try:
        documented = service.generate_docstring(
            code=request.code,
            language=request.language,
            style=request.style,
        )
        return DocstringResponse(documented_code=documented)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-flowchart", response_model=FlowchartResponse)
async def generate_flowchart(
    request: FlowchartRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Generate Mermaid flowchart for code logic."""
    try:
        diagram = service.generate_flowchart(request.code, request.language)
        return FlowchartResponse(mermaid_diagram=diagram)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-dependency-graph", response_model=DependencyGraphResponse)
async def generate_dependency_graph(
    request: DependencyGraphRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Generate dependency graph for code."""
    try:
        diagram = service.generate_dependency_graph(request.code, request.language)
        return DependencyGraphResponse(mermaid_diagram=diagram)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refactor", response_model=RefactorResponse)
async def suggest_refactoring(
    request: RefactorRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Suggest refactoring improvements."""
    try:
        explanation, refactored = service.suggest_refactoring(
            code=request.code,
            language=request.language,
            focus=request.focus,
        )
        return RefactorResponse(
            explanation=explanation,
            refactored_code=refactored,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_about_code(
    request: ChatRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Answer questions about code."""
    try:
        # Convert history dicts to ChatMessage objects
        history = None
        if request.history:
            history = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in request.history
            ]
        
        answer = service.chat_about_code(
            code=request.code,
            question=request.question,
            history=history,
            language=request.language,
        )
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-github", response_model=GitHubResponse)
async def fetch_from_github(request: GitHubRequest):
    """Fetch code from GitHub URL."""
    try:
        code, filename = fetch_github_code(request.url)
        return GitHubResponse(code=code, filename=filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/full-analysis", response_model=FullAnalysisResponse)
async def full_analysis(
    request: FullAnalysisRequest,
    service: CodeExplainerService = Depends(get_service),
):
    """Perform comprehensive code analysis."""
    try:
        result = service.full_analysis(
            code=request.code,
            language=request.language,
            mode=AnalysisMode(request.mode),
            audience=AudienceLevel(request.audience),
        )
        
        return FullAnalysisResponse(
            language=result.language,
            explanation=result.explanation,
            complexity=result.complexity,
            security_issues=result.security_issues,
            best_practices=result.best_practices,
            error=result.error,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Run with Uvicorn (for direct execution)
# =============================================================================

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the FastAPI server with Uvicorn.
    
    Args:
        host: Server host address.
        port: Server port.
    """
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.backend.api:app",
        host=host or settings.API_HOST,
        port=port or settings.API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    run_api_server()

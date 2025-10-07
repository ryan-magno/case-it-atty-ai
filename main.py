"""
Main FastAPI application for Forensic Attorney AI Backend.
Production-ready with proper error handling, logging, and CORS configuration.
"""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from config import get_settings
from models import (
    PlayerHypothesis, 
    AttorneyResponse, 
    ErrorResponse, 
    CaseInfo, 
    AvailableCases,
    HealthCheckResponse
)
from attorney_ai import AttorneyAI
from utils import (
    load_case_knowledge, 
    get_all_cases, 
    validate_api_key,
    KnowledgeBaseError,
    check_openai_health,
    check_knowledge_base_health
)

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    Runs before the application starts receiving requests.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("Performing health checks...")
    
    # Check OpenAI configuration
    if not check_openai_health():
        logger.warning("⚠️  Azure OpenAI not properly configured - AI features may not work")
    else:
        logger.info("✓ Azure OpenAI configured")
    
    # Check knowledge base
    if not check_knowledge_base_health():
        logger.warning("⚠️  Knowledge base empty or inaccessible")
    else:
        cases = get_all_cases()
        logger.info(f"✓ Knowledge base loaded with {len(cases)} case(s)")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered legal expert for forensic case review in Roblox game",
    lifespan=lifespan
)


# Configure CORS for Roblox HttpService
# Roblox doesn't send Origin headers, so we allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests and their processing time.
    """
    start_time = time.time()
    
    # Log request
    logger.info(f"Incoming {request.method} request: {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        f"Completed {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with detailed feedback.
    """
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "VALIDATION_ERROR",
            "error_message": "Invalid request data",
            "detail": exc.errors()
        }
    )


@app.exception_handler(KnowledgeBaseError)
async def knowledge_base_exception_handler(request: Request, exc: KnowledgeBaseError):
    """
    Handle knowledge base errors (case not found, etc.).
    """
    logger.error(f"Knowledge base error: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "success": False,
            "error_type": "KNOWLEDGE_BASE_ERROR",
            "error_message": str(exc),
            "detail": "Check /attorney/available-cases for valid case IDs"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected errors.
    """
    logger.error(f"Unexpected error on {request.url.path}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "INTERNAL_SERVER_ERROR",
            "error_message": "An unexpected error occurred",
            "detail": str(exc) if settings.DEBUG else "Contact system administrator"
        }
    )


# ==================== ENDPOINTS ====================

@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint - basic service information.
    """
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "cases": "/attorney/available-cases",
            "review": "/attorney/review-hypothesis"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring.
    Azure App Service can use this for health probes.
    """
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        openai_configured=check_openai_health(),
        knowledge_base_loaded=check_knowledge_base_health()
    )


@app.get("/attorney/available-cases", response_model=AvailableCases, tags=["Attorney"])
async def get_available_cases():
    """
    Get list of all available cases.
    Roblox can call this to show players which cases are available.
    """
    try:
        cases = get_all_cases()
        
        # Transform to CaseInfo models
        case_info_list = []
        for case in cases:
            try:
                # Load full case data to get details
                case_data = load_case_knowledge(case["case_id"])
                
                case_info = CaseInfo(
                    case_id=case_data["case_id"],
                    case_name=case_data["case_name"],
                    total_critical_evidence=len(case_data["required_evidence"]["critical"]),
                    applicable_laws=list(case_data["legal_framework"]["applicable_laws"].keys())
                )
                case_info_list.append(case_info)
                
            except Exception as e:
                logger.warning(f"Could not load details for case {case['case_id']}: {e}")
                continue
        
        return AvailableCases(
            success=True,
            available_cases=case_info_list,
            total_cases=len(case_info_list)
        )
        
    except Exception as e:
        logger.error(f"Error fetching available cases: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve available cases"
        )


@app.get("/attorney/case-info/{case_id}", response_model=CaseInfo, tags=["Attorney"])
async def get_case_info(case_id: str):
    """
    Get detailed information about a specific case.
    
    Args:
        case_id: The case identifier
        
    Returns:
        CaseInfo with case details
    """
    try:
        case_knowledge = load_case_knowledge(case_id)
        
        return CaseInfo(
            case_id=case_knowledge["case_id"],
            case_name=case_knowledge["case_name"],
            total_critical_evidence=len(case_knowledge["required_evidence"]["critical"]),
            applicable_laws=list(case_knowledge["legal_framework"]["applicable_laws"].keys())
        )
        
    except KnowledgeBaseError:
        raise  # Let the exception handler deal with it
    except Exception as e:
        logger.error(f"Error getting case info for {case_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve case information"
        )


@app.post("/attorney/review-hypothesis", response_model=AttorneyResponse, tags=["Attorney"])
async def review_hypothesis(
    hypothesis: PlayerHypothesis,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Main endpoint - Review player's hypothesis and provide legal critique.
    
    This is called from Roblox when player reaches 80% progress.
    
    Args:
        hypothesis: Player's hypothesis submission
        x_api_key: Optional API key for authentication (from header)
        
    Returns:
        AttorneyResponse with complete evaluation
    """
    # Validate API key if configured
    if not validate_api_key(x_api_key):
        logger.warning(f"Invalid API key attempted for case {hypothesis.case_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    
    try:
        logger.info(f"Processing hypothesis review for case: {hypothesis.case_id}")
        
        # Load case knowledge
        case_knowledge = load_case_knowledge(hypothesis.case_id)
        
        # Initialize Attorney AI
        attorney = AttorneyAI(case_knowledge)
        
        # Evaluate hypothesis
        result = attorney.evaluate_hypothesis(hypothesis)
        
        logger.info(
            f"Hypothesis evaluation complete - "
            f"Score: {result.overall_score}/100, "
            f"Strength: {result.case_strength}"
        )
        
        return result
        
    except KnowledgeBaseError:
        raise  # Let the exception handler deal with it
    except Exception as e:
        logger.error(f"Error processing hypothesis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing hypothesis review"
        )


# ==================== ADMIN/DEBUG ENDPOINTS ====================
# These can be removed in production or protected with authentication

@app.get("/debug/config", tags=["Debug"], include_in_schema=settings.DEBUG)
async def debug_config():
    """
    Debug endpoint to check configuration.
    Only visible if DEBUG=True.
    """
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    return {
        "openai_endpoint": settings.AZURE_OPENAI_ENDPOINT[:50] + "...",
        "openai_configured": bool(settings.AZURE_OPENAI_API_KEY),
        "deployment_name": settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        "knowledge_base_path": settings.KNOWLEDGE_BASE_PATH,
        "log_level": settings.LOG_LEVEL
    }


# ==================== APPLICATION ENTRY POINT ====================

if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    # In production, this is handled by gunicorn via startup.txt
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
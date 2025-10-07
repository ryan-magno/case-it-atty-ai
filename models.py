"""
Pydantic models for request/response validation.
Ensures type safety and automatic validation of incoming/outgoing data.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PlayerHypothesis(BaseModel):
    """
    Player's hypothesis submission from Roblox.
    Validated automatically by FastAPI.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "case_id": "RW-2025-01",
                "perpetrator": "Samuel Jose",
                "motive": "Revenge for thesis rejection",
                "method": "Deployed ransomware using rejected thesis code",
                "evidence_collected": [
                    "2025-RW-UNIK_001_SERVERIMAGE_v1",
                    "2025-RW-UNIK_002_SERVERLOGS_v1"
                ],
                "forensic_accuracy_score": 85,
                "key_connections": [
                    "IP addresses match",
                    "Code similarity confirmed"
                ]
            }
        }
    )
    
    case_id: str = Field(
        ..., 
        description="Case identifier (e.g., 'RW-2025-01')",
        min_length=1,
        max_length=50
    )
    
    perpetrator: str = Field(
        ..., 
        description="Player's identification of the perpetrator",
        min_length=1,
        max_length=200
    )
    
    motive: str = Field(
        ..., 
        description="Player's explanation of motive",
        min_length=1,
        max_length=500
    )
    
    method: str = Field(
        ..., 
        description="Player's explanation of how crime was committed",
        min_length=1,
        max_length=500
    )
    
    evidence_collected: List[str] = Field(
        ..., 
        description="List of evidence IDs collected by player"
    )
    
    forensic_accuracy_score: int = Field(
        ..., 
        ge=0,  # Greater than or equal to 0
        le=100,  # Less than or equal to 100
        description="Player's forensic accuracy score (0-100)"
    )
    
    key_connections: List[str] = Field(
        default=[],
        description="Player's identified evidence connections"
    )
    
    @field_validator('evidence_collected')
    @classmethod
    def validate_evidence_list(cls, v):
        """Ensure evidence list is not empty and has no duplicates."""
        if not v:
            raise ValueError("Evidence list cannot be empty")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate evidence IDs detected")
        return v


class AttorneyResponse(BaseModel):
    """
    Attorney AI's response sent back to Roblox.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "case_strength": "strong",
                "overall_score": 88,
                "critique": "Excellent work. Your evidence chain is solid...",
                "missing_critical_evidence": [],
                "procedural_violations": [],
                "legal_issues": [],
                "recommendations": []
            }
        }
    )
    
    success: bool = Field(
        ...,
        description="Whether the evaluation was successful"
    )
    
    case_strength: str = Field(
        ..., 
        description="Overall case strength assessment",
        pattern="^(strong|moderate|weak|not_prosecutable)$"
    )
    
    overall_score: int = Field(
        ..., 
        ge=0, 
        le=100,
        description="Combined evaluation score (0-100)"
    )
    
    critique: str = Field(
        ..., 
        description="AI-generated legal critique"
    )
    
    missing_critical_evidence: List[str] = Field(
        default=[],
        description="List of missing critical evidence IDs"
    )
    
    procedural_violations: List[str] = Field(
        default=[],
        description="List of procedural violations detected"
    )
    
    legal_issues: List[str] = Field(
        default=[],
        description="List of legal issues with the case"
    )
    
    recommendations: List[str] = Field(
        default=[],
        description="Actionable recommendations for improvement"
    )


class ErrorResponse(BaseModel):
    """
    Standardized error response for failed requests.
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error_type": "CASE_NOT_FOUND",
                "error_message": "The specified case does not exist",
                "detail": "Case ID 'INVALID-01' not found in knowledge base"
            }
        }
    )
    
    success: bool = False
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class CaseInfo(BaseModel):
    """
    Basic case information response.
    """
    
    case_id: str
    case_name: str
    total_critical_evidence: int
    applicable_laws: List[str]


class AvailableCases(BaseModel):
    """
    List of available cases response.
    """
    
    success: bool = True
    available_cases: List[CaseInfo]
    total_cases: int


class HealthCheckResponse(BaseModel):
    """
    Health check endpoint response.
    """
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    openai_configured: bool = Field(..., description="Azure OpenAI connection status")
    knowledge_base_loaded: bool = Field(..., description="Knowledge base availability")
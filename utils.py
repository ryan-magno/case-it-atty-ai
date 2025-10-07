"""
Utility functions for knowledge base management and common operations.
"""

import json
import os
import glob
import logging
from typing import Dict, List, Optional
from functools import lru_cache

from config import get_settings

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()


class KnowledgeBaseError(Exception):
    """Custom exception for knowledge base errors."""
    pass


@lru_cache(maxsize=10)
def load_case_knowledge(case_id: str) -> Dict:
    """
    Load case knowledge from JSON files with caching.
    
    Args:
        case_id: The case identifier to load
        
    Returns:
        Dict containing case knowledge data
        
    Raises:
        KnowledgeBaseError: If case not found or invalid JSON
    """
    knowledge_base_path = settings.KNOWLEDGE_BASE_PATH
    
    # Check if knowledge base directory exists
    if not os.path.exists(knowledge_base_path):
        logger.error(f"Knowledge base directory not found: {knowledge_base_path}")
        raise KnowledgeBaseError(
            "Knowledge base directory not found. Please ensure 'knowledge_base' folder exists."
        )
    
    # Get all JSON files in knowledge_base folder
    json_pattern = os.path.join(knowledge_base_path, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        logger.error("No JSON files found in knowledge base")
        raise KnowledgeBaseError(
            "No knowledge base files found. Please add case JSON files to the knowledge_base folder."
        )
    
    # Search for matching case
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                case_data = json.load(f)
                
                # Check if this is the case we're looking for
                if case_data.get("case_id") == case_id:
                    logger.info(f"Successfully loaded case: {case_id} from {file_path}")
                    return case_data
                    
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in file {file_path}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            continue
    
    # Case not found
    logger.error(f"Case {case_id} not found in knowledge base")
    raise KnowledgeBaseError(
        f"Case '{case_id}' not found. Available cases can be retrieved from /attorney/available-cases"
    )


def get_all_cases() -> List[Dict]:
    """
    Get list of all available cases.
    
    Returns:
        List of dictionaries containing case metadata
    """
    knowledge_base_path = settings.KNOWLEDGE_BASE_PATH
    cases = []
    
    if not os.path.exists(knowledge_base_path):
        logger.warning(f"Knowledge base directory not found: {knowledge_base_path}")
        return cases
    
    json_pattern = os.path.join(knowledge_base_path, "*.json")
    json_files = glob.glob(json_pattern)
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                case_data = json.load(f)
                
                # Extract basic case info
                case_info = {
                    "case_id": case_data.get("case_id", "Unknown"),
                    "case_name": case_data.get("case_name", "Unknown Case"),
                    "file_name": os.path.basename(file_path)
                }
                cases.append(case_info)
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping invalid case file {file_path}: {e}")
            continue
    
    logger.info(f"Found {len(cases)} cases in knowledge base")
    return cases


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate API key from Roblox request.
    
    Args:
        api_key: API key from request header
        
    Returns:
        True if valid (or validation disabled), False otherwise
    """
    # If no API key is configured, allow all requests (for development)
    if not settings.ROBLOX_API_KEY:
        return True
    
    # Compare provided key with configured key
    return api_key == settings.ROBLOX_API_KEY


def sanitize_text_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize text input from Roblox to prevent injection attacks.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    # Trim to max length
    text = text[:max_length]
    
    # Remove any potential SQL/NoSQL injection patterns (basic sanitization)
    # Note: Pydantic validation handles most of this, but good to be safe
    dangerous_patterns = ["<script>", "javascript:", "onerror=", "onclick="]
    for pattern in dangerous_patterns:
        text = text.replace(pattern, "")
    
    return text.strip()


def check_openai_health() -> bool:
    """
    Check if Azure OpenAI is properly configured.
    
    Returns:
        True if configured, False otherwise
    """
    try:
        return all([
            settings.AZURE_OPENAI_ENDPOINT,
            settings.AZURE_OPENAI_API_KEY,
            settings.AZURE_OPENAI_DEPLOYMENT_NAME
        ])
    except Exception:
        return False


def check_knowledge_base_health() -> bool:
    """
    Check if knowledge base is accessible and has valid cases.
    
    Returns:
        True if healthy, False otherwise
    """
    try:
        cases = get_all_cases()
        return len(cases) > 0
    except Exception:
        return False
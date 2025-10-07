"""
Unit tests for the utility functions in utils.py.
These tests ensure that helper functions for loading data, validation,
and sanitization work correctly.
"""

import os
import json
import shutil
import pytest

# Assuming your project structure allows this import
from utils import (
    load_case_knowledge,
    get_all_cases,
    validate_api_key,
    sanitize_text_input,
    KnowledgeBaseError
)
from config import get_settings

# --- Fixtures for setting up a temporary knowledge base ---

@pytest.fixture(scope="module")
def temp_kb_path():
    """Create a temporary directory for the knowledge base."""
    path = "./temp_knowledge_base"
    os.makedirs(path, exist_ok=True)
    
    # Find the actual case files in knowledge_base folder
    base_dir = os.path.dirname(__file__)
    source_kb = os.path.join(base_dir, '..', 'knowledge_base')
    
    # Copy real case files into the temp directory
    case_1_src = os.path.join(source_kb, "case_1.json")
    case_2_src = os.path.join(source_kb, "case_2.json")
    
    if os.path.exists(case_1_src):
        shutil.copy(case_1_src, os.path.join(path, "case_1.json"))
    if os.path.exists(case_2_src):
        shutil.copy(case_2_src, os.path.join(path, "case_2.json"))
    
    # Create an invalid JSON file for testing error handling
    with open(os.path.join(path, "invalid.json"), "w") as f:
        f.write("{'bad_json':}")
        
    yield path
    
    # Teardown: remove the temporary directory after tests are done
    shutil.rmtree(path)


@pytest.fixture(autouse=True)
def override_settings(monkeypatch, temp_kb_path):
    """Fixture to automatically override settings for all tests in this file."""
    # Use monkeypatch to temporarily change the KNOWLEDGE_BASE_PATH setting
    monkeypatch.setattr(get_settings(), 'KNOWLEDGE_BASE_PATH', temp_kb_path)
    # Clear the lru_cache on load_case_knowledge to ensure tests are isolated
    load_case_knowledge.cache_clear()


# --- Tests for load_case_knowledge ---

def test_load_case_knowledge_success():
    """Test loading a valid case file from the knowledge base."""
    case_data = load_case_knowledge("RW-2025-01")
    assert isinstance(case_data, dict)
    assert case_data["case_id"] == "RW-2025-01"


def test_load_case_knowledge_not_found():
    """Test that KnowledgeBaseError is raised for a non-existent case."""
    with pytest.raises(KnowledgeBaseError, match="Case 'NOT-A-REAL-CASE' not found"):
        load_case_knowledge("NOT-A-REAL-CASE")


def test_load_case_knowledge_no_directory(monkeypatch):
    """Test that KnowledgeBaseError is raised if the KB directory doesn't exist."""
    # Point to a directory that we know doesn't exist
    monkeypatch.setattr(get_settings(), 'KNOWLEDGE_BASE_PATH', './non_existent_dir/')
    load_case_knowledge.cache_clear() # important to clear cache
    
    with pytest.raises(KnowledgeBaseError, match="Knowledge base directory not found"):
        load_case_knowledge("RW-2025-01")


# --- Tests for get_all_cases ---

def test_get_all_cases_success():
    """Test retrieving all valid cases from the knowledge base."""
    cases = get_all_cases()
    # Should find 2 valid cases, ignoring the invalid one
    assert len(cases) == 2
    case_ids = {case["case_id"] for case in cases}
    assert "RW-2025-01" in case_ids
    assert "2025-CISTM-0116" in case_ids


# --- Tests for validate_api_key ---

def test_validate_api_key(monkeypatch):
    """Test API key validation logic."""
    # Set a temporary API key for this test
    test_key = "my-secret-test-key"
    monkeypatch.setattr(get_settings(), 'ROBLOX_API_KEY', test_key)
    
    assert validate_api_key(test_key) is True
    assert validate_api_key("wrong-key") is False
    assert validate_api_key(None) is False
    
    # Test when no API key is configured (should always be True)
    monkeypatch.setattr(get_settings(), 'ROBLOX_API_KEY', None)
    assert validate_api_key("any-key") is True
    assert validate_api_key(None) is True


# --- Tests for sanitize_text_input ---

def test_sanitize_text_input():
    """Test the input sanitization function."""
    # Test trimming
    assert sanitize_text_input("  hello world  ") == "hello world"
    # Test max length
    long_text = "a" * 1500
    assert len(sanitize_text_input(long_text)) == 1000
    # Test removal of dangerous patterns
    malicious_input = "hello <script>alert('xss')</script> world"
    sanitized = sanitize_text_input(malicious_input)
    assert "<script>" not in sanitized
    assert "alert('xss')" in sanitized # Note: basic replace doesn't remove content inside

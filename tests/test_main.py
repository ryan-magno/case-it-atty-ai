"""
Tests for the main FastAPI application endpoints (main.py).
These are integration tests that simulate HTTP requests to the running application.
"""

import os
import sys
import json
from fastapi.testclient import TestClient
from unittest.mock import patch
from unittest.mock import patch, MagicMock

# This adds the project root directory to the Python path.
# It allows the test files to import modules from the parent directory (e.g., main, utils).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from config import get_settings

# Initialize the test client
client = TestClient(app)
settings = get_settings()

# Define a valid API key for testing if one is set
VALID_API_KEY = settings.ROBLOX_API_KEY or "test_api_key_if_none"
HEADERS = {"X-API-Key": VALID_API_KEY}


# Fixture to temporarily set the API key for tests
def setup_module(module):
    """Set up the environment for tests."""
    # If ROBLOX_API_KEY is not set in the environment, set one for testing.
    if not settings.ROBLOX_API_KEY:
        os.environ["ROBLOX_API_KEY"] = VALID_API_KEY
        settings.ROBLOX_API_KEY = VALID_API_KEY


def teardown_module(module):
    """Clean up after tests."""
    if "ROBLOX_API_KEY" in os.environ and settings.ROBLOX_API_KEY == VALID_API_KEY:
        del os.environ["ROBLOX_API_KEY"]


def test_root_endpoint():
    """Test the root endpoint to ensure the service is running."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == settings.APP_NAME
    assert data["status"] == "operational"


def test_health_check_endpoint():
    """Test the health check endpoint for monitoring status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "openai_configured" in data
    assert "knowledge_base_loaded" in data


def test_get_available_cases_endpoint():
    """Test fetching the list of all available cases."""
    response = client.get("/attorney/available-cases")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "available_cases" in data
    assert len(data["available_cases"]) > 0
    # Check if a known case is in the list
    case_ids = [case["case_id"] for case in data["available_cases"]]
    assert "RW-2025-01" in case_ids


def test_get_case_info_endpoint_success():
    """Test fetching details for a specific, valid case."""
    case_id = "RW-2025-01"
    response = client.get(f"/attorney/case-info/{case_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == case_id
    assert data["case_name"] == "Universidad ng Kanluran Ransomware Attack"


def test_get_case_info_endpoint_not_found():
    """Test fetching details for a non-existent case."""
    case_id = "INVALID-CASE-ID"
    response = client.get(f"/attorney/case-info/{case_id}")
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
    assert data["error_type"] == "KNOWLEDGE_BASE_ERROR"


@patch("attorney_ai.AzureOpenAI")
def test_review_hypothesis_strong_case(mock_azure_client):
    """
    Test the main review endpoint with a strong, correct hypothesis.
    Mocks the OpenAI call to avoid external dependencies.
    """
    # Mock the client instance and its chat.completions.create method
    mock_client_instance = MagicMock()
    mock_azure_client.return_value = mock_client_instance
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked AI Critique: Excellent work."
    mock_client_instance.chat.completions.create.return_value = mock_response

    # This payload represents a near-perfect submission for case_1.json
    payload = {
        "case_id": "RW-2025-01",
        "perpetrator": "Samuel Jose",
        "motive": "Revenge for his thesis being rejected.",
        "method": "He used the ransomware code from his rejected thesis.",
        "evidence_collected": [
            "2025-RW-UNIK_001_SERVERIMAGE_v1",
            "2025-RW-UNIK_002_SERVERLOGS_v1",
            "2025-RW-UNIK_003_RANSOMNOTE_v1",
            "2025-RW-UNIK_006_LAPTOPCOMMANDS_v1",
            "2025-RW-UNIK_007_THESESIMILARITY_v1"
        ],
        "forensic_accuracy_score": 95,
        "key_connections": [
            "IP address 103.XX.XXX.XX matches between server logs and laptop",
            "Thesis code identical to ransomware code"
        ]
    }

    response = client.post("/attorney/review-hypothesis", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["case_strength"] in ["strong", "moderate"]
    assert data["overall_score"] >= 85
    assert "Mocked AI Critique" in data["critique"]
    assert not data["missing_critical_evidence"]
    assert not data["procedural_violations"]


@patch("attorney_ai.AzureOpenAI")
def test_review_hypothesis_weak_case(mock_azure_client):
    """Test the review endpoint with a weak hypothesis missing critical evidence."""
    # Mock the client instance
    mock_client_instance = MagicMock()
    mock_azure_client.return_value = mock_client_instance
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked AI Critique: Significant issues detected."
    mock_client_instance.chat.completions.create.return_value = mock_response

    # This payload is missing most critical evidence
    payload = {
        "case_id": "RW-2025-01",
        "perpetrator": "An unknown hacker",
        "motive": "For money.",
        "method": "Hacking.",
        "evidence_collected": [
            "2025-RW-UNIK_003_RANSOMNOTE_v1" # Only one piece of evidence
        ],
        "forensic_accuracy_score": 40,
        "key_connections": []
    }

    response = client.post("/attorney/review-hypothesis", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["case_strength"] in ["weak", "not_prosecutable"]
    assert data["overall_score"] < 50
    assert len(data["missing_critical_evidence"]) == 4


def test_review_hypothesis_invalid_api_key():
    """Test that a request with an invalid API key is rejected."""
    payload = {"case_id": "RW-2025-01", "perpetrator": "test", "motive": "test", "method": "test", "evidence_collected": ["1"], "forensic_accuracy_score": 50}
    response = client.post(
        "/attorney/review-hypothesis",
        json=payload,
        headers={"X-API-Key": "this-is-a-wrong-key"}
    )
    assert response.status_code == 401


def test_review_hypothesis_missing_api_key():
    """Test that a request with a missing API key is rejected if one is configured."""
    if not settings.ROBLOX_API_KEY:
        # If no key is configured in settings, this test is not applicable
        return

    payload = {"case_id": "RW-2025-01", "perpetrator": "test", "motive": "test", "method": "test", "evidence_collected": ["1"], "forensic_accuracy_score": 50}
    response = client.post("/attorney/review-hypothesis", json=payload)
    assert response.status_code == 401


def test_review_hypothesis_validation_error():
    """Test that requests with invalid data structures are rejected with a 422 error."""
    # Payload is missing required fields like 'perpetrator'
    invalid_payload = {
        "case_id": "RW-2025-01"
    }
    response = client.post("/attorney/review-hypothesis", json=invalid_payload, headers=HEADERS)
    assert response.status_code == 422 # Unprocessable Entity
    data = response.json()
    assert data["error_type"] == "VALIDATION_ERROR"


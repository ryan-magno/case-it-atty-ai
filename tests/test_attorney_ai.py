"""
Unit tests for the AttorneyAI class in attorney_ai.py.
These tests focus on the internal logic of scoring and evaluation,
isolating it from the web framework.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# This adds the project root directory to the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from attorney_ai import AttorneyAI
from models import PlayerHypothesis

# Load sample case data from the JSON files to use in tests
@pytest.fixture
def case_1_knowledge():
    """Provides the knowledge base data for case_1.json."""
    # Look in knowledge_base folder at project root
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, '..', 'knowledge_base', 'case_1.json')
    with open(file_path, "r") as f:
        return json.load(f)

@pytest.fixture
def perfect_hypothesis_case_1(case_1_knowledge):
    """Provides a perfect hypothesis for case_1."""
    solution = case_1_knowledge["canonical_solution"]
    critical_evidence = [ev["id"] for ev in case_1_knowledge["required_evidence"]["critical"]]
    
    return PlayerHypothesis(
        case_id="RW-2025-01",
        perpetrator=solution["perpetrator"],
        motive=solution["motive"],
        method=solution["method"],
        evidence_collected=critical_evidence,
        forensic_accuracy_score=100,
        key_connections=solution["key_connections"]
    )

@pytest.fixture
def poor_hypothesis_case_1():
    """Provides a poor hypothesis with incorrect info and missing evidence."""
    return PlayerHypothesis(
        case_id="RW-2025-01",
        perpetrator="John Doe",
        motive="For fun",
        method="A simple virus",
        evidence_collected=["2025-RW-UNIK_003_RANSOMNOTE_v1"],
        forensic_accuracy_score=20,
        key_connections=[]
    )

def test_attorney_ai_initialization(case_1_knowledge):
    """Test that the AttorneyAI class initializes correctly."""
    attorney = AttorneyAI(case_1_knowledge)
    assert attorney.case_data["case_id"] == "RW-2025-01"
    assert "evaluation_criteria" in attorney.case_data

def test_score_hypothesis_perfect(case_1_knowledge, perfect_hypothesis_case_1):
    """Test hypothesis scoring with a perfect submission."""
    attorney = AttorneyAI(case_1_knowledge)
    score = attorney._score_hypothesis(perfect_hypothesis_case_1)
    # The score should be at or near 100
    assert score >= 90

def test_score_hypothesis_poor(case_1_knowledge, poor_hypothesis_case_1):
    """Test hypothesis scoring with an incorrect submission."""
    attorney = AttorneyAI(case_1_knowledge)
    score = attorney._score_hypothesis(poor_hypothesis_case_1)
    # The score should be very low
    assert score < 30

def test_score_evidence_complete(case_1_knowledge, perfect_hypothesis_case_1):
    """Test evidence scoring with all critical evidence collected."""
    attorney = AttorneyAI(case_1_knowledge)
    score = attorney._score_evidence(perfect_hypothesis_case_1.evidence_collected)
    assert score == 100

def test_score_evidence_incomplete(case_1_knowledge, poor_hypothesis_case_1):
    """Test evidence scoring with missing critical evidence."""
    attorney = AttorneyAI(case_1_knowledge)
    score = attorney._score_evidence(poor_hypothesis_case_1.evidence_collected)
    # There are 5 critical items, only 1 was collected. Score should be 1/5 = 20.
    assert score == 20

def test_find_missing_critical_evidence(case_1_knowledge):
    """Test the logic for identifying missing critical evidence."""
    attorney = AttorneyAI(case_1_knowledge)
    collected = ["2025-RW-UNIK_001_SERVERIMAGE_v1"]
    missing = attorney._find_missing_critical_evidence(collected)
    assert len(missing) == 4
    assert "2025-RW-UNIK_002_SERVERLOGS_v1" in missing

def test_determine_case_strength(case_1_knowledge):
    """Test the case strength determination logic under various conditions."""
    attorney = AttorneyAI(case_1_knowledge)
    # Strong case
    assert attorney._determine_case_strength(95, [], []) == "strong"
    # Moderate case
    assert attorney._determine_case_strength(80, ["one_missing_item"], []) == "moderate"
    # Weak case
    assert attorney._determine_case_strength(60, ["one", "two"], []) == "weak"
    # Not prosecutable
    assert attorney._determine_case_strength(40, ["many", "missing"], ["violations"]) == "not_prosecutable"

@patch("attorney_ai.AzureOpenAI")
def test_generate_critique_openai_success(mock_azure_client, case_1_knowledge, perfect_hypothesis_case_1):
    """Test that a critique is generated via a successful OpenAI API call."""
    # Mock the client instance and its chat.completions.create method
    mock_client_instance = MagicMock()
    mock_azure_client.return_value = mock_client_instance
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Successful AI Critique"
    mock_client_instance.chat.completions.create.return_value = mock_response

    attorney = AttorneyAI(case_1_knowledge)
    critique = attorney._generate_critique(perfect_hypothesis_case_1, 95)
    
    # Assert that the OpenAI API was called and the mocked content was returned
    mock_client_instance.chat.completions.create.assert_called_once()
    assert critique == "Successful AI Critique"

@patch("attorney_ai.AzureOpenAI")
def test_generate_critique_openai_failure_fallback(mock_azure_client, case_1_knowledge, poor_hypothesis_case_1):
    """Test that a fallback critique is generated if the OpenAI API call fails."""
    # Mock the client to raise an exception
    mock_client_instance = MagicMock()
    mock_azure_client.return_value = mock_client_instance
    mock_client_instance.chat.completions.create.side_effect = Exception("OpenAI is down")

    attorney = AttorneyAI(case_1_knowledge)
    critique = attorney._generate_critique(poor_hypothesis_case_1, 30)

    # Assert that the API was called
    mock_client_instance.chat.completions.create.assert_called_once()
    # Assert that a fallback message is returned instead of the AI response
    assert "substantial additional work" in critique

def test_full_evaluation_flow(case_1_knowledge, perfect_hypothesis_case_1):
    """An end-to-end test of the evaluation method, mocking the AI part."""
    with patch.object(AttorneyAI, '_generate_critique', return_value="Mocked Critique") as mock_critique:
        attorney = AttorneyAI(case_1_knowledge)
        response = attorney.evaluate_hypothesis(perfect_hypothesis_case_1)
        
        assert response.success is True
        assert response.case_strength == "strong"
        assert response.overall_score >= 95
        assert response.critique == "Mocked Critique"
        assert len(response.missing_critical_evidence) == 0
        mock_critique.assert_called_once()
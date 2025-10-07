"""
Attorney AI core logic for hypothesis evaluation.
Handles scoring, critique generation, and legal assessment.
"""

import logging
from typing import Dict, List, Tuple
from openai import AzureOpenAI, OpenAIError

from models import PlayerHypothesis, AttorneyResponse
from config import get_settings

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()

class AttorneyAI:
    """
    Attorney Magno AI - Legal expert for case review.
    Evaluates player hypotheses against canonical solutions.
    """
    
    def __init__(self, case_knowledge: Dict):
        """
        Initialize Attorney AI with case knowledge and the Azure OpenAI client.
        
        Args:
            case_knowledge: Dictionary containing case data from knowledge base
        """
        self.case_data = case_knowledge
        # Initialize the new AzureOpenAI client (v1.x syntax)
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,  
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        logger.info(f"Attorney AI initialized for case: {case_knowledge.get('case_id')}")
    
    def evaluate_hypothesis(self, hypothesis: PlayerHypothesis) -> AttorneyResponse:
        """
        Main evaluation function - orchestrates the entire assessment.
        
        Args:
            hypothesis: Player's hypothesis submission
            
        Returns:
            AttorneyResponse with complete evaluation
        """
        logger.info(f"Evaluating hypothesis for case: {hypothesis.case_id}")
        
        try:
            # Step 1: Score the hypothesis components
            hypothesis_score = self._score_hypothesis(hypothesis)
            logger.debug(f"Hypothesis score: {hypothesis_score}/100")
            
            # Step 2: Score evidence collection
            evidence_score = self._score_evidence(hypothesis.evidence_collected)
            logger.debug(f"Evidence score: {evidence_score}/100")
            
            # Step 3: Calculate weighted total score
            # Formula: (hypothesis + evidence + forensic_accuracy) / 3
            total_score = int(
                (hypothesis_score + evidence_score + hypothesis.forensic_accuracy_score) / 3
            )
            logger.info(f"Total score calculated: {total_score}/100")
            
            # Step 4: Identify issues
            missing_evidence = self._find_missing_critical_evidence(hypothesis.evidence_collected)
            violations = self._check_procedural_violations(hypothesis.evidence_collected)
            legal_issues = self._assess_legal_issues(hypothesis)
            
            # Step 5: Determine overall case strength
            case_strength = self._determine_case_strength(
                total_score, 
                missing_evidence, 
                violations
            )
            
            # Step 6: Generate AI critique using GPT-4
            critique = self._generate_critique(hypothesis, total_score)
            
            # Step 7: Generate actionable recommendations
            recommendations = self._generate_recommendations(
                missing_evidence, 
                violations, 
                legal_issues
            )
            
            # Return complete assessment
            return AttorneyResponse(
                success=True,
                case_strength=case_strength,
                overall_score=total_score,
                critique=critique,
                missing_critical_evidence=missing_evidence,
                procedural_violations=violations,
                legal_issues=legal_issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error during hypothesis evaluation: {e}", exc_info=True)
            raise
    
    def _score_hypothesis(self, hypothesis: PlayerHypothesis) -> int:
        """
        Score player's hypothesis against canonical solution.
        
        Returns:
            Score from 0-100 based on correctness
        """
        score = 0
        criteria = self.case_data["evaluation_criteria"]["hypothesis_scoring"]
        canonical = self.case_data["canonical_solution"]
        
        # 1. Perpetrator Identification (30 points)
        if hypothesis.perpetrator.lower() == canonical["perpetrator"].lower():
            score += criteria["perpetrator_identification"]["points"]
            logger.debug("Perpetrator correctly identified (+30 points)")
        else:
            logger.debug(f"Perpetrator mismatch: '{hypothesis.perpetrator}' vs '{canonical['perpetrator']}'")
        
        # 2. Motive Understanding (20 points)
        # Extract key motive keywords from canonical solution
        canonical_motive_lower = canonical["motive"].lower()
        motive_keywords = self._extract_keywords(canonical_motive_lower)
        
        # Check if player's motive contains key concepts
        player_motive_lower = hypothesis.motive.lower()
        keyword_matches = sum(1 for keyword in motive_keywords if keyword in player_motive_lower)
        
        if keyword_matches > 0:
            # Partial credit based on keyword matches
            motive_score = min(criteria["motive_understanding"]["points"], 
                             int(criteria["motive_understanding"]["points"] * (keyword_matches / len(motive_keywords))))
            score += motive_score
            logger.debug(f"Motive understanding: {keyword_matches}/{len(motive_keywords)} keywords (+{motive_score} points)")
        
        # 3. Evidence Correlation (30 points)
        canonical_connections = set(conn.lower() for conn in canonical["key_connections"])
        player_connections = set(conn.lower() for conn in hypothesis.key_connections)
        
        # Calculate overlap
        matching_connections = canonical_connections.intersection(player_connections)
        connection_match_ratio = len(matching_connections) / len(canonical_connections) if canonical_connections else 0
        
        correlation_score = int(criteria["evidence_correlation"]["points"] * connection_match_ratio)
        score += correlation_score
        logger.debug(f"Evidence correlation: {len(matching_connections)}/{len(canonical_connections)} (+{correlation_score} points)")
        
        # 4. Method Explanation (20 points)
        canonical_method_lower = canonical["method"].lower()
        method_keywords = self._extract_keywords(canonical_method_lower)
        
        player_method_lower = hypothesis.method.lower()
        method_keyword_matches = sum(1 for keyword in method_keywords if keyword in player_method_lower)
        
        if method_keyword_matches > 0:
            method_score = min(criteria["method_explanation"]["points"],
                             int(criteria["method_explanation"]["points"] * (method_keyword_matches / len(method_keywords))))
            score += method_score
            logger.debug(f"Method explanation: {method_keyword_matches}/{len(method_keywords)} keywords (+{method_score} points)")
        
        return score
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from canonical solution text.
        Filters out common words.
        """
        # Common words to ignore
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                     "of", "with", "by", "from", "as", "is", "was", "are", "were"}
        
        # Split and filter
        words = text.split()
        keywords = [word.strip(",.!?;:") for word in words 
                   if len(word) > 3 and word.lower() not in stop_words]
        
        return keywords[:10]  # Limit to 10 most important keywords
    
    def _score_evidence(self, evidence_collected: List[str]) -> int:
        """
        Score evidence collection completeness.
        
        Returns:
            Score from 0-100 based on critical evidence collected
        """
        required_critical = [item["id"] for item in self.case_data["required_evidence"]["critical"]]
        
        # Count how many critical pieces were collected
        critical_collected = sum(1 for evidence_id in evidence_collected 
                               if evidence_id in required_critical)
        
        # Calculate percentage
        critical_percentage = critical_collected / len(required_critical) if required_critical else 0
        
        score = int(critical_percentage * 100)
        logger.debug(f"Evidence collection: {critical_collected}/{len(required_critical)} critical items ({score}%)")
        
        return score
    
    def _find_missing_critical_evidence(self, evidence_collected: List[str]) -> List[str]:
        """
        Identify missing critical evidence.
        
        Returns:
            List of missing critical evidence IDs
        """
        required_critical = [item["id"] for item in self.case_data["required_evidence"]["critical"]]
        missing = [evidence_id for evidence_id in required_critical 
                  if evidence_id not in evidence_collected]
        
        if missing:
            logger.info(f"Missing {len(missing)} critical evidence items")
        
        return missing
    
    def _check_procedural_violations(self, evidence_collected: List[str]) -> List[str]:
        """
        Check for procedural violations based on evidence.
        
        Returns:
            List of detected violations
        """
        violations = []
        
        # Check for hash verification (critical for digital evidence)
        hash_required_evidence = [item["id"] for item in self.case_data["required_evidence"]["critical"]
                                 if "SERVERIMAGE" in item["id"] or "IMAGE" in item["id"]]
        
        for evidence_id in hash_required_evidence:
            if evidence_id not in evidence_collected:
                violations.append(f"Missing forensic image with hash verification: {evidence_id}")
        
        return violations
    
    def _assess_legal_issues(self, hypothesis: PlayerHypothesis) -> List[str]:
        """
        Assess legal issues with the case build.
        
        Returns:
            List of legal issues that could affect prosecution
        """
        issues = []
        
        # Check for extortion proof in cybercrime cases
        if "RW" in hypothesis.case_id or "ransomware" in hypothesis.method.lower():
            ransom_evidence = [e for e in hypothesis.evidence_collected if "RANSOM" in e]
            if not ransom_evidence:
                issues.append("Cannot prove extortion element without ransom demand evidence")
        
        # Check for connection establishment
        if not hypothesis.key_connections:
            issues.append("No evidence connections established - case relies on circumstantial evidence only")
        
        # Check for assault cases
        if "CISTM" in hypothesis.case_id:
            wound_evidence = [e for e in hypothesis.evidence_collected if "WOUND" in e]
            cctv_evidence = [e for e in hypothesis.evidence_collected if "CCTV" in e]
            
            if not wound_evidence:
                issues.append("Missing victim injury documentation")
            if not cctv_evidence:
                issues.append("No video evidence to corroborate assault sequence")
        
        return issues
    
    def _determine_case_strength(
        self, 
        total_score: int, 
        missing_evidence: List[str], 
        violations: List[str]
    ) -> str:
        """
        Determine overall prosecutability of the case.
        
        Returns:
            One of: "strong", "moderate", "weak", "not_prosecutable"
        """
        # Strong case: High score, no missing critical evidence, no violations
        if total_score >= 90 and not missing_evidence and not violations:
            return "strong"
        
        # Moderate case: Good score, minimal issues
        elif total_score >= 70 and len(missing_evidence) <= 1:
            return "moderate"
        
        # Weak case: Passing score but significant issues
        elif total_score >= 50:
            return "weak"
        
        # Not prosecutable: Too many problems
        else:
            return "not_prosecutable"
    
    def _generate_recommendations(
        self, 
        missing_evidence: List[str], 
        violations: List[str], 
        legal_issues: List[str]
    ) -> List[str]:
        """
        Generate actionable recommendations for the player.
        
        Returns:
            List of specific recommendations
        """
        recommendations = []
        
        # Address missing evidence
        if missing_evidence:
            # Show only first 2 to avoid overwhelming the player
            evidence_names = missing_evidence[:2]
            recommendations.append(
                f"Return to investigation and collect: {', '.join(evidence_names)}"
            )
        
        # Address procedural violations
        if violations:
            recommendations.append(
                "Review evidence collection procedures - procedural violations detected"
            )
        
        # Address legal issues
        if legal_issues:
            recommendations.append(
                "Strengthen evidence connections to address legal vulnerabilities"
            )
        
        # If everything is good
        if not recommendations:
            recommendations.append("Case is ready for prosecution - proceed to final report")
        
        return recommendations
    
    def _generate_critique(self, hypothesis: PlayerHypothesis, total_score: int) -> str:
        """
        Generate AI-powered critique using GPT-4.
        Scales tone and detail based on performance.
        
        Returns:
            Personalized critique text
        """
        # Select appropriate response scale
        response_scaling = self.case_data["response_scaling"]
        
        if total_score >= 90:
            tone_guide = response_scaling["score_90_plus"]
        elif total_score >= 70:
            tone_guide = response_scaling["score_70_89"]
        elif total_score >= 50:
            tone_guide = response_scaling["score_50_69"]
        else:
            tone_guide = response_scaling["score_below_50"]
        
        # Build context for GPT-4
        canonical_solution = self.case_data["canonical_solution"]
        
        # Construct prompt for Attorney Magno
        prompt = f"""You are Attorney Magno, a senior Filipino legal expert specializing in forensic case review.

CASE: {self.case_data["case_name"]}
TONE: {tone_guide["tone"]}
DETAIL LEVEL: {tone_guide["detail"]}

CANONICAL SOLUTION:
- Perpetrator: {canonical_solution["perpetrator"]}
- Motive: {canonical_solution["motive"]}
- Method: {canonical_solution["method"]}

STUDENT INVESTIGATOR'S HYPOTHESIS:
- Perpetrator Identified: {hypothesis.perpetrator}
- Stated Motive: {hypothesis.motive}
- Stated Method: {hypothesis.method}
- Evidence Collected: {len(hypothesis.evidence_collected)} items
- Forensic Accuracy Score: {hypothesis.forensic_accuracy_score}%

EVALUATION SCORE: {total_score}/100

Provide a {tone_guide["detail"]} legal critique focusing on:
1. Case prosecutability under Philippine law
2. Evidence admissibility issues
3. Strengths and weaknesses of the investigation

Reference relevant Philippine laws (RA 10175 for cybercrime, RPC for physical crimes).
Keep response under 250 words.
Maintain {tone_guide["tone"]} tone.
Be direct and educational."""

        try:
            # Call Azure OpenAI GPT-4 using the new client syntax
            response = self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are Attorney Magno, a Filipino legal expert specializing in cybercrime and forensic case prosecution. You provide educational, constructive feedback to trainee investigators."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=350,
                temperature=0.7,
                top_p=0.9
            )
            
            critique = response.choices[0].message.content.strip()
            logger.info("GPT-4 critique generated successfully")
            return critique
            
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            # Fallback critique if AI fails
            return self._generate_fallback_critique(total_score, tone_guide)
        
        except Exception as e:
            logger.error(f"Unexpected error generating critique: {e}", exc_info=True)
            return self._generate_fallback_critique(total_score, tone_guide)
    
    def _generate_fallback_critique(self, total_score: int, tone_guide: Dict) -> str:
        """
        Generate a basic critique if GPT-4 is unavailable.
        
        Returns:
            Fallback critique text
        """
        if total_score >= 90:
            return "Excellent investigative work. Your hypothesis aligns with the evidence, and procedural compliance is strong. This case is ready for prosecution."
        elif total_score >= 70:
            return "Solid investigation with good evidence collection. Address the identified gaps to strengthen prosecutability."
        elif total_score >= 50:
            return "Investigation shows promise but has significant weaknesses. Review missing evidence and procedural violations before proceeding."
        else:
            return "Investigation requires substantial additional work. Critical evidence is missing, and procedural compliance issues must be addressed."

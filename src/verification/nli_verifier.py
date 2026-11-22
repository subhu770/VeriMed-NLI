"""Modular NLI verification engine combining deterministic numeric guards, semantic matching, and clinical entailment."""

import os
import re
from typing import List, Optional, Tuple, Dict, Set, Any
from src.api.schemas import (
    AtomicClaim,
    ClinicalTokenType,
    EntailmentDecision,
    HallucinationAlert,
    HallucinationType,
    AlertSeverity,
    VerifiedClaim,
    AuditReport,
    SafetyStatus,
)
from src.verification.numeric_guard import NumericGuard, NumericGuardCheckResult


class NLIVerifier:
    """Production-grade NLI verification engine with CPU simulation mode and strict attribution."""

    STOPWORDS = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "up", "about", "into", "over", "after", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "patient", "showed", "noted", "reports", "daily",
        "will", "would", "shall", "should", "may", "might", "must", "can", "could",
        "left", "right", "upper", "lower", "within"
    }

    def __init__(
        self,
        simulation_mode: Optional[bool] = None,
        min_entailment_confidence: float = 0.60,
        contradiction_confidence: float = 0.95
    ) -> None:
        if simulation_mode is None:
            env_mode = os.getenv("SIMULATION_MODE", "true").lower()
            self.simulation_mode = env_mode in ("true", "1", "yes")
        else:
            self.simulation_mode = simulation_mode

        self.min_entailment_confidence = min_entailment_confidence
        self.contradiction_confidence = contradiction_confidence
        self.numeric_guard = NumericGuard()

    def verify_claim(self, claim: AtomicClaim, source_sentences: List[str]) -> VerifiedClaim:
        """Verify an individual atomic claim against indexed source document sentences."""
        if not source_sentences:
            return VerifiedClaim(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                decision=EntailmentDecision.NEUTRAL,
                confidence=0.5,
                best_source_sent_id=None,
                best_source_sentence=None,
                matched_citations=[],
                alerts=[
                    HallucinationAlert(
                        claim_id=claim.claim_id,
                        alert_type=HallucinationType.UNGROUNDED_CLAIM,
                        severity=AlertSeverity.WARNING,
                        reason="Source document is empty; cannot ground claim.",
                        claim_span=claim.claim_text,
                        source_span=None,
                        source_sent_id=None
                    )
                ],
                is_redlined=True,
                redline_reason="Ungrounded: No reference source available."
            )

        claim_tokens = self._tokenize(claim.claim_text)

        # Evaluate candidate sentences
        candidates: List[Dict[str, Any]] = []
        for sent_idx, sentence in enumerate(source_sentences):
            sentence_tokens = self._tokenize(sentence)
            overlap = self._compute_token_overlap(claim_tokens, sentence_tokens)

            # Check entity sharing
            shared_entities = []
            if claim.extracted_entities:
                shared_entities = [
                    e for e in claim.extracted_entities
                    if re.search(r'\b' + re.escape(e.lower()) + r'\b', sentence.lower())
                ]
                entity_score = len(shared_entities) / len(claim.extracted_entities)
            else:
                entity_score = overlap

            guard_res = self.numeric_guard.verify_claim_against_sentence(
                claim=claim,
                source_sentence=sentence,
                source_sent_id=sent_idx
            )

            # Compute alignment score
            alignment_score = self._compute_alignment_score(claim, sentence, claim_tokens, guard_res)

            candidates.append({
                "sent_idx": sent_idx,
                "sentence": sentence,
                "overlap": overlap,
                "entity_score": entity_score,
                "shared_entities": shared_entities,
                "guard_res": guard_res,
                "alignment_score": alignment_score,
                "is_relevant": (len(shared_entities) > 0) or (overlap >= 0.35)
            })

        # 1. Check for Entailment match
        # Find candidates that are not contradictions and satisfy entailment threshold
        entailed_candidates = [
            c for c in candidates
            if not c["guard_res"].is_contradiction and c["alignment_score"] >= self.min_entailment_confidence
        ]

        if entailed_candidates:
            # Sort by highest alignment score
            entailed_candidates.sort(key=lambda x: x["alignment_score"], reverse=True)
            top_entailed = entailed_candidates[0]

            matched_indices = [
                c["sent_idx"] for c in entailed_candidates
                if c["alignment_score"] >= self.min_entailment_confidence
            ]

            return VerifiedClaim(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                decision=EntailmentDecision.ENTAILED,
                confidence=round(min(0.99, top_entailed["alignment_score"]), 4),
                best_source_sent_id=top_entailed["sent_idx"],
                best_source_sentence=top_entailed["sentence"],
                matched_citations=matched_indices,
                alerts=[],
                is_redlined=False,
                redline_reason=None
            )

        # 2. Check for Contradiction on relevant sentences
        # A contradiction is valid if the sentence is relevant to the claim (shares entity or key terms)
        contradiction_candidates = [
            c for c in candidates
            if c["guard_res"].is_contradiction and c["is_relevant"]
        ]

        if contradiction_candidates:
            # Sort by highest relevance/overlap
            contradiction_candidates.sort(key=lambda x: (x["entity_score"], x["overlap"]), reverse=True)
            top_contra = contradiction_candidates[0]
            guard_res: NumericGuardCheckResult = top_contra["guard_res"]

            alert = guard_res.alert or HallucinationAlert(
                claim_id=claim.claim_id,
                alert_type=guard_res.hallucination_type,
                severity=AlertSeverity.CRITICAL,
                reason=guard_res.details,
                claim_span=claim.claim_text,
                source_span=top_contra["sentence"],
                source_sent_id=top_contra["sent_idx"]
            )

            return VerifiedClaim(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                decision=EntailmentDecision.CONTRADICTION,
                confidence=self.contradiction_confidence,
                best_source_sent_id=top_contra["sent_idx"],
                best_source_sentence=top_contra["sentence"],
                matched_citations=[],
                alerts=[alert],
                is_redlined=True,
                redline_reason=alert.reason
            )

        # 3. Otherwise: Ungrounded Neutral claim
        # Find closest sentence for diagnostic attribution
        candidates.sort(key=lambda x: x["overlap"], reverse=True)
        closest = candidates[0] if candidates else None

        alert = HallucinationAlert(
            claim_id=claim.claim_id,
            alert_type=HallucinationType.UNGROUNDED_CLAIM,
            severity=AlertSeverity.WARNING,
            reason=f"Claim could not be grounded in source text (best token overlap: {closest['overlap'] if closest else 0.0:.2f}).",
            claim_span=claim.claim_text,
            source_span=closest["sentence"] if closest else None,
            source_sent_id=closest["sent_idx"] if closest else None
        )

        return VerifiedClaim(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            decision=EntailmentDecision.NEUTRAL,
            confidence=0.55,
            best_source_sent_id=closest["sent_idx"] if closest else None,
            best_source_sentence=closest["sentence"] if closest else None,
            matched_citations=[],
            alerts=[alert],
            is_redlined=True,
            redline_reason="Ungrounded: Insufficient source evidence."
        )

    def verify_all_claims(
        self,
        claims: List[AtomicClaim],
        source_sentences: List[str]
    ) -> List[VerifiedClaim]:
        """Verify an entire collection of decomposed atomic claims."""
        return [self.verify_claim(c, source_sentences) for c in claims]

    def build_audit_report(
        self,
        verified_claims: List[VerifiedClaim],
        processing_time_ms: float,
        document_id: Optional[str] = None
    ) -> AuditReport:
        """Aggregate verified claims into a production clinical audit report."""
        total = len(verified_claims)
        entailed = sum(1 for c in verified_claims if c.decision == EntailmentDecision.ENTAILED)
        contradictions = sum(1 for c in verified_claims if c.decision == EntailmentDecision.CONTRADICTION)
        neutrals = sum(1 for c in verified_claims if c.decision == EntailmentDecision.NEUTRAL)
        redlined = sum(1 for c in verified_claims if c.is_redlined)

        all_alerts: List[HallucinationAlert] = []
        for c in verified_claims:
            all_alerts.extend(c.alerts)

        fcs = (entailed / total) if total > 0 else 1.0

        if contradictions > 0:
            status = SafetyStatus.FAILED
        elif neutrals > 0:
            status = SafetyStatus.FLAGGED
        else:
            status = SafetyStatus.PASSED

        return AuditReport(
            document_id=document_id,
            total_claims=total,
            entailed_claims_count=entailed,
            contradiction_claims_count=contradictions,
            neutral_claims_count=neutrals,
            redlined_claims_count=redlined,
            factual_consistency_score=round(fcs, 4),
            hallucination_catch_rate=None,
            safety_status=status,
            alerts=all_alerts,
            processing_time_ms=round(processing_time_ms, 2)
        )

    def _compute_alignment_score(
        self,
        claim: AtomicClaim,
        sentence: str,
        claim_tokens: Set[str],
        guard_res: NumericGuardCheckResult
    ) -> float:
        """Compute holistic semantic alignment score between claim and source sentence."""
        sentence_tokens = self._tokenize(sentence)
        token_overlap = self._compute_token_overlap(claim_tokens, sentence_tokens)

        # Entity coverage
        if claim.extracted_entities:
            matched_entities = sum(
                1 for e in claim.extracted_entities
                if re.search(r'\b' + re.escape(e.lower()) + r'\b', sentence.lower())
            )
            entity_score = matched_entities / len(claim.extracted_entities)
        else:
            entity_score = token_overlap

        # Numeric bonus if explicitly consistent
        numeric_score = 1.0 if guard_res.is_consistent and guard_res.source_token is not None else 0.5

        if claim.extracted_entities:
            combined = (0.50 * entity_score) + (0.35 * token_overlap) + (0.15 * numeric_score)
        else:
            combined = (0.75 * token_overlap) + (0.25 * numeric_score)

        return combined

    def _tokenize(self, text: str) -> Set[str]:
        """Normalize and tokenize text to a set of content words."""
        words = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return {w for w in words if w not in self.STOPWORDS and len(w) > 1}

    def _compute_token_overlap(self, tokens1: Set[str], tokens2: Set[str]) -> float:
        """Compute token containment ratio of tokens1 in tokens2."""
        if not tokens1:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        return len(intersection) / len(tokens1)

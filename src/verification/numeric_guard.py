"""Deterministic numerical, dosage, lab value, and negation consistency checker."""

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from src.api.schemas import (
    AtomicClaim,
    ClinicalToken,
    ClinicalTokenType,
    HallucinationAlert,
    HallucinationType,
    AlertSeverity,
)
from src.pipeline.claim_decomposer import ClinicalUnitExtractor, NegationDetector


@dataclass
class NumericGuardCheckResult:
    """Result of evaluating numerical and negation consistency between claim and source."""
    is_contradiction: bool
    is_consistent: bool
    hallucination_type: HallucinationType
    alert: Optional[HallucinationAlert]
    matched_entity: Optional[str]
    claim_token: Optional[ClinicalToken]
    source_token: Optional[ClinicalToken]
    details: str


class NumericGuard:
    """Deterministic guardrail enforcing absolute consistency for numbers, dosages, labs, and negations."""

    def __init__(self, numeric_tolerance: float = 1e-4) -> None:
        self.extractor = ClinicalUnitExtractor()
        self.negation_detector = NegationDetector()
        self.tolerance = numeric_tolerance

    def verify_claim_against_sentence(
        self,
        claim: AtomicClaim,
        source_sentence: str,
        source_sent_id: Optional[int] = None
    ) -> NumericGuardCheckResult:
        """Evaluate claim against a candidate source sentence for numerical or negation conflicts."""
        source_tokens = self.extractor.extract_tokens(source_sentence)
        source_is_neg, source_neg_cues = self.negation_detector.analyze_negation(source_sentence)

        # 1. Check Clinical Tokens (Dosages, Lab Values, BP, Percentages) first
        token_result = self._check_token_conflicts(
            claim=claim,
            source_tokens=source_tokens,
            source_sentence=source_sentence,
            source_sent_id=source_sent_id
        )
        if token_result.is_contradiction:
            return token_result

        # 2. Check Negation Polarity Conflict on Shared Entities or Conditions
        negation_result = self._check_negation_conflict(
            claim=claim,
            source_sentence=source_sentence,
            source_is_neg=source_is_neg,
            source_neg_cues=source_neg_cues,
            source_sent_id=source_sent_id
        )
        if negation_result.is_contradiction:
            return negation_result

        # If token result found consistent matching numbers
        if token_result.is_consistent:
            return token_result

        # If neither contradiction nor explicit numeric match was found
        return NumericGuardCheckResult(
            is_contradiction=False,
            is_consistent=True,
            hallucination_type=HallucinationType.NONE,
            alert=None,
            matched_entity=None,
            claim_token=None,
            source_token=None,
            details="No numerical or polarity conflict detected."
        )

    def _check_negation_conflict(
        self,
        claim: AtomicClaim,
        source_sentence: str,
        source_is_neg: bool,
        source_neg_cues: List[str],
        source_sent_id: Optional[int]
    ) -> NumericGuardCheckResult:
        """Detect inverted clinical findings (e.g. denies chest pain vs confirmed chest pain)."""
        shared_entities = self._find_shared_concepts(claim.claim_text, source_sentence)
        if not shared_entities:
            return NumericGuardCheckResult(
                is_contradiction=False,
                is_consistent=True,
                hallucination_type=HallucinationType.NONE,
                alert=None,
                matched_entity=None,
                claim_token=None,
                source_token=None,
                details="No shared clinical concept for negation evaluation."
            )

        if claim.is_negated != source_is_neg:
            matched_concept = ", ".join(shared_entities)
            claim_state = "NEGATED" if claim.is_negated else "AFFIRMATIVE"
            source_state = "NEGATED" if source_is_neg else "AFFIRMATIVE"

            alert = HallucinationAlert(
                claim_id=claim.claim_id,
                alert_type=HallucinationType.NEGATION_FLIP,
                severity=AlertSeverity.CRITICAL,
                reason=(
                    f"Flipped clinical negation polarity for concept '{matched_concept}': "
                    f"Claim is {claim_state} (cues: {claim.negation_cues}) but Source is {source_state} "
                    f"(cues: {source_neg_cues})."
                ),
                claim_span=claim.claim_text,
                source_span=source_sentence,
                source_sent_id=source_sent_id
            )

            return NumericGuardCheckResult(
                is_contradiction=True,
                is_consistent=False,
                hallucination_type=HallucinationType.NEGATION_FLIP,
                alert=alert,
                matched_entity=matched_concept,
                claim_token=None,
                source_token=None,
                details=alert.reason
            )

        return NumericGuardCheckResult(
            is_contradiction=False,
            is_consistent=True,
            hallucination_type=HallucinationType.NONE,
            alert=None,
            matched_entity=None,
            claim_token=None,
            source_token=None,
            details="Negation polarities align."
        )

    def _check_token_conflicts(
        self,
        claim: AtomicClaim,
        source_tokens: List[ClinicalToken],
        source_sentence: str,
        source_sent_id: Optional[int]
    ) -> NumericGuardCheckResult:
        """Verify that dosages, labs, and numbers in the claim strictly match the source."""
        if not claim.extracted_quantities:
            return NumericGuardCheckResult(
                is_contradiction=False,
                is_consistent=True,
                hallucination_type=HallucinationType.NONE,
                alert=None,
                matched_entity=None,
                claim_token=None,
                source_token=None,
                details="No quantities in claim to verify."
            )

        for c_tok in claim.extracted_quantities:
            # 1. Blood Pressure check
            if c_tok.token_type == ClinicalTokenType.BLOOD_PRESSURE:
                bp_match = self._verify_bp(c_tok, source_tokens, claim.claim_id, source_sentence, source_sent_id)
                if bp_match.is_contradiction:
                    return bp_match

            # 2. Lab Value check (prioritize labs over general dosages)
            elif c_tok.token_type == ClinicalTokenType.LAB_VALUE:
                lab_match = self._verify_lab(c_tok, source_tokens, claim.claim_id, source_sentence, source_sent_id)
                if lab_match.is_contradiction:
                    return lab_match

            # 3. Dosage check
            elif c_tok.token_type == ClinicalTokenType.DOSAGE:
                dosage_match = self._verify_dosage(c_tok, source_tokens, claim.claim_id, source_sentence, source_sent_id)
                if dosage_match.is_contradiction:
                    return dosage_match

            # 4. Frequency check
            elif c_tok.token_type == ClinicalTokenType.FREQUENCY:
                freq_match = self._verify_frequency(c_tok, source_tokens, claim.claim_id, source_sentence, source_sent_id)
                if freq_match.is_contradiction:
                    return freq_match

            # 5. General / Percentage / Technical
            elif c_tok.token_type in (ClinicalTokenType.PERCENTAGE, ClinicalTokenType.GENERAL_NUMERIC):
                gen_match = self._verify_general_numeric(c_tok, source_tokens, claim.claim_id, source_sentence, source_sent_id, claim.claim_text)
                if gen_match.is_contradiction:
                    return gen_match

        return NumericGuardCheckResult(
            is_contradiction=False,
            is_consistent=True,
            hallucination_type=HallucinationType.NONE,
            alert=None,
            matched_entity=None,
            claim_token=None,
            source_token=None,
            details="All numeric quantities verified consistent."
        )

    def _verify_bp(
        self,
        c_tok: ClinicalToken,
        source_tokens: List[ClinicalToken],
        claim_id: str,
        source_sentence: str,
        source_sent_id: Optional[int]
    ) -> NumericGuardCheckResult:
        source_bps = [t for t in source_tokens if t.token_type == ClinicalTokenType.BLOOD_PRESSURE]
        if not source_bps:
            return NumericGuardCheckResult(
                is_contradiction=False,
                is_consistent=False,
                hallucination_type=HallucinationType.NONE,
                alert=None,
                matched_entity="blood pressure",
                claim_token=c_tok,
                source_token=None,
                details="No BP in candidate source."
            )

        for s_bp in source_bps:
            sys_diff = abs((c_tok.numeric_val or 0.0) - (s_bp.numeric_val or 0.0))
            dia_diff = abs((c_tok.secondary_val or 0.0) - (s_bp.secondary_val or 0.0))
            if sys_diff < self.tolerance and dia_diff < self.tolerance:
                return NumericGuardCheckResult(
                    is_contradiction=False,
                    is_consistent=True,
                    hallucination_type=HallucinationType.NONE,
                    alert=None,
                    matched_entity="blood pressure",
                    claim_token=c_tok,
                    source_token=s_bp,
                    details=f"Exact match on BP: {c_tok.raw_text}"
                )

        top_s_bp = source_bps[0]
        alert = HallucinationAlert(
            claim_id=claim_id,
            alert_type=HallucinationType.NUMERIC_MISMATCH,
            severity=AlertSeverity.CRITICAL,
            reason=f"Blood pressure mismatch: Claim states '{c_tok.raw_text}' but source states '{top_s_bp.raw_text}'.",
            claim_span=c_tok.raw_text,
            source_span=top_s_bp.raw_text,
            source_sent_id=source_sent_id
        )
        return NumericGuardCheckResult(
            is_contradiction=True,
            is_consistent=False,
            hallucination_type=HallucinationType.NUMERIC_MISMATCH,
            alert=alert,
            matched_entity="blood pressure",
            claim_token=c_tok,
            source_token=top_s_bp,
            details=alert.reason
        )

    def _verify_dosage(
        self,
        c_tok: ClinicalToken,
        source_tokens: List[ClinicalToken],
        claim_id: str,
        source_sentence: str,
        source_sent_id: Optional[int]
    ) -> NumericGuardCheckResult:
        entity = c_tok.context_entity
        candidate_dosages = [t for t in source_tokens if t.token_type == ClinicalTokenType.DOSAGE]

        if entity:
            matched_dosages = [t for t in candidate_dosages if t.context_entity and t.context_entity.lower() == entity.lower()]
            if not matched_dosages and entity.lower() in source_sentence.lower():
                matched_dosages = candidate_dosages
        else:
            matched_dosages = candidate_dosages

        if not matched_dosages:
            return NumericGuardCheckResult(
                is_contradiction=False,
                is_consistent=False,
                hallucination_type=HallucinationType.NONE,
                alert=None,
                matched_entity=entity,
                claim_token=c_tok,
                source_token=None,
                details=f"No matching dosage candidate for entity '{entity}'."
            )

        for s_dos in matched_dosages:
            if s_dos.normalized_val is not None and c_tok.normalized_val is not None:
                if abs(s_dos.normalized_val - c_tok.normalized_val) < self.tolerance:
                    return NumericGuardCheckResult(
                        is_contradiction=False,
                        is_consistent=True,
                        hallucination_type=HallucinationType.NONE,
                        alert=None,
                        matched_entity=entity,
                        claim_token=c_tok,
                        source_token=s_dos,
                        details=f"Dosage matches for {entity}: {c_tok.raw_text}"
                    )

        top_s_dos = matched_dosages[0]
        alert = HallucinationAlert(
            claim_id=claim_id,
            alert_type=HallucinationType.DOSAGE_ALTERATION,
            severity=AlertSeverity.CRITICAL,
            reason=(
                f"Altered dosage for medication '{entity or 'unknown'}': "
                f"Claim asserts '{c_tok.raw_text}' ({c_tok.normalized_val} {c_tok.normalized_unit}), "
                f"but Source records '{top_s_dos.raw_text}' ({top_s_dos.normalized_val} {top_s_dos.normalized_unit})."
            ),
            claim_span=c_tok.raw_text,
            source_span=top_s_dos.raw_text,
            source_sent_id=source_sent_id
        )
        return NumericGuardCheckResult(
            is_contradiction=True,
            is_consistent=False,
            hallucination_type=HallucinationType.DOSAGE_ALTERATION,
            alert=alert,
            matched_entity=entity,
            claim_token=c_tok,
            source_token=top_s_dos,
            details=alert.reason
        )

    def _verify_lab(
        self,
        c_tok: ClinicalToken,
        source_tokens: List[ClinicalToken],
        claim_id: str,
        source_sentence: str,
        source_sent_id: Optional[int]
    ) -> NumericGuardCheckResult:
        lab_entity = c_tok.context_entity
        source_labs = [t for t in source_tokens if t.token_type == ClinicalTokenType.LAB_VALUE]

        matched_labs = []
        if lab_entity:
            matched_labs = [t for t in source_labs if t.context_entity and t.context_entity.lower() == lab_entity.lower()]
            if not matched_labs and lab_entity.lower() in source_sentence.lower():
                matched_labs = source_labs
        else:
            matched_labs = source_labs

        if not matched_labs:
            return NumericGuardCheckResult(
                is_contradiction=False,
                is_consistent=False,
                hallucination_type=HallucinationType.NONE,
                alert=None,
                matched_entity=lab_entity,
                claim_token=c_tok,
                source_token=None,
                details=f"No matching lab value found for '{lab_entity}'."
            )

        for s_lab in matched_labs:
            if s_lab.normalized_val is not None and c_tok.normalized_val is not None:
                rel_diff = abs(s_lab.normalized_val - c_tok.normalized_val) / max(1.0, abs(s_lab.normalized_val))
                if rel_diff < 0.005:
                    return NumericGuardCheckResult(
                        is_contradiction=False,
                        is_consistent=True,
                        hallucination_type=HallucinationType.NONE,
                        alert=None,
                        matched_entity=lab_entity,
                        claim_token=c_tok,
                        source_token=s_lab,
                        details=f"Lab value matches for {lab_entity}: {c_tok.raw_text}"
                    )

        top_s_lab = matched_labs[0]
        alert = HallucinationAlert(
            claim_id=claim_id,
            alert_type=HallucinationType.FABRICATED_LAB,
            severity=AlertSeverity.CRITICAL,
            reason=(
                f"Contradictory or fabricated lab value for '{lab_entity}': "
                f"Claim reports '{c_tok.raw_text}' ({c_tok.normalized_val} {c_tok.normalized_unit}), "
                f"whereas Source records '{top_s_lab.raw_text}' ({top_s_lab.normalized_val} {top_s_lab.normalized_unit})."
            ),
            claim_span=c_tok.raw_text,
            source_span=top_s_lab.raw_text,
            source_sent_id=source_sent_id
        )
        return NumericGuardCheckResult(
            is_contradiction=True,
            is_consistent=False,
            hallucination_type=HallucinationType.FABRICATED_LAB,
            alert=alert,
            matched_entity=lab_entity,
            claim_token=c_tok,
            source_token=top_s_lab,
            details=alert.reason
        )

    def _verify_frequency(
        self,
        c_tok: ClinicalToken,
        source_tokens: List[ClinicalToken],
        claim_id: str,
        source_sentence: str,
        source_sent_id: Optional[int]
    ) -> NumericGuardCheckResult:
        entity = c_tok.context_entity
        source_freqs = [t for t in source_tokens if t.token_type == ClinicalTokenType.FREQUENCY]
        if not source_freqs:
            return NumericGuardCheckResult(
                is_contradiction=False,
                is_consistent=True,
                hallucination_type=HallucinationType.NONE,
                alert=None,
                matched_entity=entity,
                claim_token=c_tok,
                source_token=None,
                details="No frequency in source."
            )

        norm_claim_freq = self._normalize_freq_str(c_tok.raw_text)
        for s_freq in source_freqs:
            norm_source_freq = self._normalize_freq_str(s_freq.raw_text)
            if norm_claim_freq == norm_source_freq:
                return NumericGuardCheckResult(
                    is_contradiction=False,
                    is_consistent=True,
                    hallucination_type=HallucinationType.NONE,
                    alert=None,
                    matched_entity=entity,
                    claim_token=c_tok,
                    source_token=s_freq,
                    details="Frequency aligns."
                )

        if entity and entity.lower() in source_sentence.lower():
            top_s_freq = source_freqs[0]
            alert = HallucinationAlert(
                claim_id=claim_id,
                alert_type=HallucinationType.DOSAGE_ALTERATION,
                severity=AlertSeverity.CRITICAL,
                reason=(
                    f"Conflicting dosing frequency for '{entity}': "
                    f"Claim prescribes '{c_tok.raw_text}', but Source specifies '{top_s_freq.raw_text}'."
                ),
                claim_span=c_tok.raw_text,
                source_span=top_s_freq.raw_text,
                source_sent_id=source_sent_id
            )
            return NumericGuardCheckResult(
                is_contradiction=True,
                is_consistent=False,
                hallucination_type=HallucinationType.DOSAGE_ALTERATION,
                alert=alert,
                matched_entity=entity,
                claim_token=c_tok,
                source_token=top_s_freq,
                details=alert.reason
            )

        return NumericGuardCheckResult(
            is_contradiction=False,
            is_consistent=True,
            hallucination_type=HallucinationType.NONE,
            alert=None,
            matched_entity=entity,
            claim_token=c_tok,
            source_token=None,
            details="Frequency check passed."
        )

    def _verify_general_numeric(
        self,
        c_tok: ClinicalToken,
        source_tokens: List[ClinicalToken],
        claim_id: str,
        source_sentence: str,
        source_sent_id: Optional[int],
        claim_text: str = ""
    ) -> NumericGuardCheckResult:
        # Check if source sentence shares context or concepts with this general numeric token
        shared = self._find_shared_concepts(claim_text, source_sentence)
        has_entity_mention = bool(c_tok.context_entity and c_tok.context_entity.lower() in source_sentence.lower())
        if not shared and not has_entity_mention:
            return NumericGuardCheckResult(
                is_contradiction=False,
                is_consistent=False,
                hallucination_type=HallucinationType.NONE,
                alert=None,
                matched_entity=c_tok.context_entity,
                claim_token=c_tok,
                source_token=None,
                details="No shared concept for general numeric comparison."
            )

        for s_tok in source_tokens:
            if s_tok.numeric_val is not None and c_tok.numeric_val is not None:
                if abs(s_tok.numeric_val - c_tok.numeric_val) < self.tolerance:
                    return NumericGuardCheckResult(
                        is_contradiction=False,
                        is_consistent=True,
                        hallucination_type=HallucinationType.NONE,
                        alert=None,
                        matched_entity=c_tok.context_entity,
                        claim_token=c_tok,
                        source_token=s_tok,
                        details="General numeric value matches."
                    )

        conflicting_tokens = [
            t for t in source_tokens
            if t.unit == c_tok.unit and t.numeric_val is not None and c_tok.numeric_val is not None
            and abs(t.numeric_val - c_tok.numeric_val) >= self.tolerance
        ]
        if conflicting_tokens:
            top_conflict = conflicting_tokens[0]
            alert = HallucinationAlert(
                claim_id=claim_id,
                alert_type=HallucinationType.NUMERIC_MISMATCH,
                severity=AlertSeverity.WARNING,
                reason=(
                    f"Numeric value mismatch for unit '{c_tok.unit}': "
                    f"Claim has {c_tok.numeric_val}, Source has {top_conflict.numeric_val}."
                ),
                claim_span=c_tok.raw_text,
                source_span=top_conflict.raw_text,
                source_sent_id=source_sent_id
            )
            return NumericGuardCheckResult(
                is_contradiction=True,
                is_consistent=False,
                hallucination_type=HallucinationType.NUMERIC_MISMATCH,
                alert=alert,
                matched_entity=c_tok.context_entity,
                claim_token=c_tok,
                source_token=top_conflict,
                details=alert.reason
            )

        return NumericGuardCheckResult(
            is_contradiction=False,
            is_consistent=True,
            hallucination_type=HallucinationType.NONE,
            alert=None,
            matched_entity=c_tok.context_entity,
            claim_token=c_tok,
            source_token=None,
            details="No conflicting general numeric token found."
        )

    def _normalize_freq_str(self, text: str) -> str:
        s = text.lower().strip()
        if s in ("bid", "b.i.d.", "twice daily", "2 times daily"):
            return "twice_daily"
        if s in ("tid", "t.i.d.", "three times daily"):
            return "three_times_daily"
        if s in ("qid", "q.i.d.", "four times daily"):
            return "four_times_daily"
        if s in ("daily", "once daily", "qd", "q.d.", "q24h"):
            return "once_daily"
        if s in ("prn", "p.r.n.", "as needed"):
            return "as_needed"
        return s

    def _find_shared_concepts(self, claim_text: str, source_text: str) -> List[str]:
        words_claim = set(re.findall(r'[a-zA-Z]{3,}', claim_text.lower()))
        words_source = set(re.findall(r'[a-zA-Z]{3,}', source_text.lower()))

        stopwords = {
            "the", "and", "was", "for", "with", "that", "this", "from",
            "have", "had", "has", "were", "been", "patient", "showed",
            "noted", "reports", "reported", "history", "clinical", "discharge",
            "admission", "hospital", "date", "findings", "daily", "given", "within"
        }
        claim_content = words_claim - stopwords
        source_content = words_source - stopwords
        return list(claim_content.intersection(source_content))

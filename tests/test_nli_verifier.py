"""Tests verifying NLI entailment, contradiction classification, and source sentence attribution."""

import pytest
from src.api.schemas import (
    EntailmentDecision,
    HallucinationType,
    SafetyStatus,
)
from src.pipeline.claim_decomposer import ClaimDecomposer
from src.verification.nli_verifier import NLIVerifier


@pytest.fixture
def verifier() -> NLIVerifier:
    return NLIVerifier(simulation_mode=True)


@pytest.fixture
def decomposer() -> ClaimDecomposer:
    return ClaimDecomposer()


@pytest.fixture
def sample_source_sentences():
    return [
        "Patient is a 68-year-old male admitted with non-ST elevation myocardial infarction.",  # src_0
        "Cardiac catheterization revealed 80% stenosis of the left anterior descending artery.",  # src_1
        "Echocardiogram demonstrated preserved left ventricular ejection fraction of 55%.",     # src_2
        "Discharge lab evaluation showed serum Potassium of 4.3 mEq/L and Creatinine 1.0 mg/dL.", # src_3
        "Serial electrocardiograms showed no evidence of acute ST elevations or arrhythmias.",  # src_4
        "Patient was discharged in stable condition on Atorvastatin 80mg daily and Metoprolol 25mg BID.", # src_5
        "Follow-up is scheduled with cardiology in three weeks."  # src_6
    ]


class TestNLIVerifierEntailment:
    """Verifies grounding, entailment classification, and precise source attribution citations."""

    def test_entailed_claim_attributed_to_correct_source_sentence(
        self,
        verifier: NLIVerifier,
        decomposer: ClaimDecomposer,
        sample_source_sentences
    ):
        claim_text = "Echocardiogram showed left ventricular ejection fraction of 55%."
        claim = decomposer.decompose(claim_text)[0]

        verified = verifier.verify_claim(claim, sample_source_sentences)

        assert verified.decision == EntailmentDecision.ENTAILED
        assert verified.best_source_sent_id == 2
        assert verified.best_source_sentence == sample_source_sentences[2]
        assert verified.is_redlined is False
        assert verified.confidence >= 0.65

    def test_medication_claim_entailed_to_discharge_source(
        self,
        verifier: NLIVerifier,
        decomposer: ClaimDecomposer,
        sample_source_sentences
    ):
        claim_text = "Patient was discharged on Atorvastatin 80mg daily."
        claim = decomposer.decompose(claim_text)[0]

        verified = verifier.verify_claim(claim, sample_source_sentences)

        assert verified.decision == EntailmentDecision.ENTAILED
        assert verified.best_source_sent_id == 5
        assert verified.is_redlined is False


class TestNLIVerifierContradiction:
    """Verifies that contradictions are detected and flagged for redlining."""

    def test_dosage_alteration_classified_as_contradiction(
        self,
        verifier: NLIVerifier,
        decomposer: ClaimDecomposer,
        sample_source_sentences
    ):
        # Source has Atorvastatin 80mg daily (src_5)
        claim_text = "Patient was discharged on Atorvastatin 20mg daily."
        claim = decomposer.decompose(claim_text)[0]

        verified = verifier.verify_claim(claim, sample_source_sentences)

        assert verified.decision == EntailmentDecision.CONTRADICTION
        assert verified.best_source_sent_id == 5
        assert verified.is_redlined is True
        assert len(verified.alerts) >= 1
        assert verified.alerts[0].alert_type == HallucinationType.DOSAGE_ALTERATION

    def test_negation_flip_classified_as_contradiction(
        self,
        verifier: NLIVerifier,
        decomposer: ClaimDecomposer,
        sample_source_sentences
    ):
        # Source has: "Serial electrocardiograms showed no evidence of acute ST elevations or arrhythmias." (src_4)
        claim_text = "Electrocardiograms confirmed active evidence of acute ST elevations and arrhythmias."
        claim = decomposer.decompose(claim_text)[0]

        verified = verifier.verify_claim(claim, sample_source_sentences)

        assert verified.decision == EntailmentDecision.CONTRADICTION
        assert verified.best_source_sent_id == 4
        assert verified.is_redlined is True
        assert verified.alerts[0].alert_type == HallucinationType.NEGATION_FLIP


class TestNLIVerifierNeutral:
    """Verifies that ungrounded / unsupported assertions are labeled NEUTRAL."""

    def test_unsupported_claim_classified_as_neutral(
        self,
        verifier: NLIVerifier,
        decomposer: ClaimDecomposer,
        sample_source_sentences
    ):
        claim_text = "Patient underwent emergent coronary artery bypass graft surgery."
        claim = decomposer.decompose(claim_text)[0]

        verified = verifier.verify_claim(claim, sample_source_sentences)

        assert verified.decision == EntailmentDecision.NEUTRAL
        assert verified.is_redlined is True
        assert len(verified.alerts) >= 1
        assert verified.alerts[0].alert_type == HallucinationType.UNGROUNDED_CLAIM


class TestNLIVerifierMultiClaimAudit:
    """Verifies multi-claim pipeline aggregation and audit report generation."""

    def test_audit_report_generation(
        self,
        verifier: NLIVerifier,
        decomposer: ClaimDecomposer,
        sample_source_sentences
    ):
        claims_text = (
            "Echocardiogram demonstrated left ventricular ejection fraction of 55%. "
            "Patient was discharged on Atorvastatin 20mg daily. "
            "Patient developed deep vein thrombosis."
        )
        claims = decomposer.decompose(claims_text)
        assert len(claims) == 3

        verified_claims = verifier.verify_all_claims(claims, sample_source_sentences)
        report = verifier.build_audit_report(verified_claims, processing_time_ms=12.5, document_id="TEST_001")

        assert report.total_claims == 3
        assert report.entailed_claims_count == 1
        assert report.contradiction_claims_count == 1
        assert report.neutral_claims_count == 1
        assert report.redlined_claims_count == 2
        assert report.safety_status == SafetyStatus.FAILED
        assert report.factual_consistency_score == round(1 / 3, 4)

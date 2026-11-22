"""Unit tests verifying 100% catch rate on altered dosages, lab values, and flipped negations."""

import pytest
from src.api.schemas import (
    AtomicClaim,
    ClinicalTokenType,
    HallucinationType,
    AlertSeverity,
)
from src.pipeline.claim_decomposer import ClaimDecomposer, ClinicalUnitExtractor
from src.verification.numeric_guard import NumericGuard


@pytest.fixture
def guard() -> NumericGuard:
    return NumericGuard()


@pytest.fixture
def decomposer() -> ClaimDecomposer:
    return ClaimDecomposer()


class TestDosageGuard:
    """Verifies that any altered dosage is strictly caught as a contradiction."""

    @pytest.mark.parametrize("source_sent,claim_text,drug,src_val,claim_val", [
        (
            "Patient was discharged on Metformin 500mg orally twice daily.",
            "Patient was discharged on Metformin 1000mg orally twice daily.",
            "Metformin", 500.0, 1000.0
        ),
        (
            "Initiated Lisinopril 10mg daily for hypertension control.",
            "Initiated Lisinopril 20mg daily for hypertension control.",
            "Lisinopril", 10.0, 20.0
        ),
        (
            "Prescribed Atorvastatin 40mg at bedtime.",
            "Prescribed Atorvastatin 80mg at bedtime.",
            "Atorvastatin", 40.0, 80.0
        ),
        (
            "Administer Enoxaparin 40mg subcutaneously daily.",
            "Administer Enoxaparin 120mg subcutaneously daily.",
            "Enoxaparin", 40.0, 120.0
        ),
        (
            "Patient takes Levothyroxine 50mcg every morning on an empty stomach.",
            "Patient takes Levothyroxine 100mcg every morning.",
            "Levothyroxine", 0.05, 0.10
        ),
    ])
    def test_altered_dosages_100_percent_caught(
        self,
        guard: NumericGuard,
        decomposer: ClaimDecomposer,
        source_sent: str,
        claim_text: str,
        drug: str,
        src_val: float,
        claim_val: float
    ):
        claims = decomposer.decompose(claim_text)
        assert len(claims) >= 1
        claim = claims[0]

        result = guard.verify_claim_against_sentence(claim, source_sent, source_sent_id=0)

        assert result.is_contradiction is True, f"Failed to catch altered dosage for {drug}"
        assert result.hallucination_type == HallucinationType.DOSAGE_ALTERATION
        assert result.alert is not None
        assert result.alert.severity == AlertSeverity.CRITICAL

    def test_unit_normalization_equivalence(self, guard: NumericGuard, decomposer: ClaimDecomposer):
        """0.05mg and 50mcg of Levothyroxine should be recognized as equivalent, not a contradiction."""
        source = "Patient takes Levothyroxine 50mcg daily."
        claim_text = "Patient takes Levothyroxine 0.05mg daily."

        claim = decomposer.decompose(claim_text)[0]
        result = guard.verify_claim_against_sentence(claim, source, source_sent_id=0)

        assert result.is_contradiction is False
        assert result.is_consistent is True


class TestLabValuesGuard:
    """Verifies 100% detection of altered, fabricated, or shifted laboratory values."""

    @pytest.mark.parametrize("source_sent,claim_text,lab_name", [
        (
            "Discharge lab demonstrated Potassium of 4.2 mEq/L within normal limits.",
            "Discharge lab demonstrated Potassium of 6.8 mEq/L.",
            "Potassium"
        ),
        (
            "Admission Creatinine was 1.1 mg/dL, showing baseline renal function.",
            "Admission Creatinine was 3.5 mg/dL with acute elevation.",
            "Creatinine"
        ),
        (
            "Outpatient HbA1c was measured at 6.5%.",
            "Outpatient HbA1c was measured at 10.2%.",
            "HbA1c"
        ),
        (
            "Hematology confirmed WBC of 11.5 k/uL.",
            "Hematology confirmed WBC of 24.5 k/uL.",
            "WBC"
        ),
        (
            "Cardiac troponin I was 0.03 ng/mL at 6 hours post-presentation.",
            "Cardiac troponin I was 2.85 ng/mL at 6 hours post-presentation.",
            "Troponin"
        ),
    ])
    def test_altered_labs_100_percent_caught(
        self,
        guard: NumericGuard,
        decomposer: ClaimDecomposer,
        source_sent: str,
        claim_text: str,
        lab_name: str
    ):
        claims = decomposer.decompose(claim_text)
        assert len(claims) >= 1
        claim = claims[0]

        result = guard.verify_claim_against_sentence(claim, source_sent, source_sent_id=0)

        assert result.is_contradiction is True, f"Failed to catch corrupted lab for {lab_name}"
        assert result.hallucination_type in (HallucinationType.FABRICATED_LAB, HallucinationType.NUMERIC_MISMATCH)
        assert result.alert is not None
        assert result.alert.severity == AlertSeverity.CRITICAL

    def test_wbc_unit_normalization(self, guard: NumericGuard, decomposer: ClaimDecomposer):
        """11.5 k/uL and 11,500 /uL represent identical WBC counts and must pass."""
        source = "WBC count was 11.5 k/uL."
        claim_text = "WBC count was 11,500 /uL."

        claim = decomposer.decompose(claim_text)[0]
        result = guard.verify_claim_against_sentence(claim, source, source_sent_id=0)

        assert result.is_contradiction is False
        assert result.is_consistent is True


class TestNegationGuard:
    """Verifies 100% catch rate on inverted clinical and technical assertions."""

    @pytest.mark.parametrize("source_sent,claim_text,concept", [
        (
            "Chest radiograph showed no evidence of pneumonia or acute consolidation.",
            "Chest radiograph showed confirmed evidence of acute pneumonia.",
            "pneumonia"
        ),
        (
            "Patient denies chest pain and shortness of breath.",
            "Patient reports severe acute chest pain.",
            "chest pain"
        ),
        (
            "Blood cultures remained negative for bacteremia after 48 hours.",
            "Blood cultures were positive for bacteremia after 48 hours.",
            "bacteremia"
        ),
        (
            "Lower extremity ultrasound was negative for deep vein thrombosis.",
            "Ultrasound confirmed acute deep vein thrombosis in lower extremity.",
            "deep vein thrombosis"
        ),
        (
            "The cluster operates without single points of failure.",
            "The cluster operates with critical single points of failure.",
            "single points of failure"
        ),
    ])
    def test_flipped_negations_100_percent_caught(
        self,
        guard: NumericGuard,
        decomposer: ClaimDecomposer,
        source_sent: str,
        claim_text: str,
        concept: str
    ):
        claims = decomposer.decompose(claim_text)
        assert len(claims) >= 1
        claim = claims[0]

        result = guard.verify_claim_against_sentence(claim, source_sent, source_sent_id=0)

        assert result.is_contradiction is True, f"Failed to catch flipped negation for '{concept}'"
        assert result.hallucination_type == HallucinationType.NEGATION_FLIP
        assert result.alert is not None
        assert result.alert.severity == AlertSeverity.CRITICAL


class TestBloodPressureAndFrequency:
    """Verifies blood pressure and dosing frequency consistency."""

    def test_bp_contradiction_caught(self, guard: NumericGuard, decomposer: ClaimDecomposer):
        source = "Patient's blood pressure was stable at 120/80 mmHg."
        claim_text = "Patient's blood pressure was 165/105 mmHg."

        claim = decomposer.decompose(claim_text)[0]
        result = guard.verify_claim_against_sentence(claim, source, source_sent_id=0)

        assert result.is_contradiction is True
        assert result.hallucination_type == HallucinationType.NUMERIC_MISMATCH

    def test_frequency_conflict_caught(self, guard: NumericGuard, decomposer: ClaimDecomposer):
        source = "Take Metformin 500mg once daily with breakfast."
        claim_text = "Take Metformin 500mg three times daily."

        claim = decomposer.decompose(claim_text)[0]
        result = guard.verify_claim_against_sentence(claim, source, source_sent_id=0)

        assert result.is_contradiction is True
        assert result.hallucination_type == HallucinationType.DOSAGE_ALTERATION

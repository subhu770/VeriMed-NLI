"""Pydantic v2 schemas for VeriMed-NLI audit, claims, and verification pipelines."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EntailmentDecision(str, Enum):
    """NLI entailment classification decisions."""
    ENTAILED = "ENTAILED"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


class HallucinationType(str, Enum):
    """Taxonomy of clinical and technical hallucination categories."""
    NONE = "NONE"
    NUMERIC_MISMATCH = "NUMERIC_MISMATCH"
    DOSAGE_ALTERATION = "DOSAGE_ALTERATION"
    NEGATION_FLIP = "NEGATION_FLIP"
    FABRICATED_LAB = "FABRICATED_LAB"
    UNGROUNDED_CLAIM = "UNGROUNDED_CLAIM"
    ENTITY_MISMATCH = "ENTITY_MISMATCH"


class AlertSeverity(str, Enum):
    """Clinical safety alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class SafetyStatus(str, Enum):
    """Overall document safety audit verdict."""
    PASSED = "PASSED"
    FLAGGED = "FLAGGED"
    FAILED = "FAILED"


class ClinicalTokenType(str, Enum):
    """Categories of clinical units and metrics."""
    DOSAGE = "dosage"
    LAB_VALUE = "lab_value"
    PERCENTAGE = "percentage"
    BLOOD_PRESSURE = "blood_pressure"
    FREQUENCY = "frequency"
    GENERAL_NUMERIC = "general_numeric"


class ClinicalToken(BaseModel):
    """Structured extraction of a numerical, dosage, or lab token."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_text: str = Field(..., description="Raw text span of the clinical quantity")
    token_type: ClinicalTokenType = Field(..., description="Classification of the token")
    numeric_val: Optional[float] = Field(None, description="Primary parsed numeric value")
    secondary_val: Optional[float] = Field(None, description="Secondary numeric value (e.g., diastolic BP)")
    unit: Optional[str] = Field(None, description="Extracted medical or engineering unit")
    normalized_val: Optional[float] = Field(None, description="Normalized standard unit value (e.g. mg, /uL)")
    normalized_unit: Optional[str] = Field(None, description="Standard canonical unit string")
    context_entity: Optional[str] = Field(None, description="Clinical entity or lab name bound to this token")


class AtomicClaim(BaseModel):
    """Atomic verifiable proposition decomposed from summary text."""
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., description="Unique claim identifier, e.g., claim_0")
    claim_text: str = Field(..., description="Clean single-sentence verifiable assertion")
    extracted_entities: List[str] = Field(default_factory=list, description="Medical/technical entities identified")
    extracted_quantities: List[ClinicalToken] = Field(default_factory=list, description="Extracted numerical & dosage tokens")
    is_negated: bool = Field(False, description="Whether this claim expresses a negative finding/instruction")
    negation_cues: List[str] = Field(default_factory=list, description="Trigger tokens causing negation classification")


class HallucinationAlert(BaseModel):
    """Structured safety alert for an ungrounded or contradictory claim."""
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., description="Associated atomic claim ID")
    alert_type: HallucinationType = Field(..., description="Detected hallucination subtype")
    severity: AlertSeverity = Field(..., description="Safety severity (CRITICAL, WARNING, INFO)")
    reason: str = Field(..., description="Clinical rationale explaining why the claim violates safety constraints")
    claim_span: str = Field(..., description="Offending span in the generated claim")
    source_span: Optional[str] = Field(None, description="Contradicting or reference source span if available")
    source_sent_id: Optional[int] = Field(None, description="Zero-based index of the candidate source sentence")


class VerifiedClaim(BaseModel):
    """Final verification status and provenance attribution for an atomic claim."""
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., description="Identifier of the verified claim")
    claim_text: str = Field(..., description="Atomic claim text")
    decision: EntailmentDecision = Field(..., description="Entailment verdict")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model or heuristic confidence score")
    best_source_sent_id: Optional[int] = Field(None, description="Top grounding source sentence index")
    best_source_sentence: Optional[str] = Field(None, description="Text of the top grounding source sentence")
    matched_citations: List[int] = Field(default_factory=list, description="List of source sentence indices supporting this claim")
    alerts: List[HallucinationAlert] = Field(default_factory=list, description="List of detected anomalies")
    is_redlined: bool = Field(False, description="Whether this claim must be suppressed or redlined")
    redline_reason: Optional[str] = Field(None, description="Explanation for redline/exclusion action")


class AuditReport(BaseModel):
    """Clinical AI safety audit report detailing verification outcomes."""
    model_config = ConfigDict(extra="forbid")

    document_id: Optional[str] = Field(None, description="Identifier of the evaluated document")
    total_claims: int = Field(..., ge=0, description="Total atomic claims extracted")
    entailed_claims_count: int = Field(..., ge=0, description="Number of fully grounded entailed claims")
    contradiction_claims_count: int = Field(..., ge=0, description="Number of contradicting claims")
    neutral_claims_count: int = Field(..., ge=0, description="Number of ungrounded neutral claims")
    redlined_claims_count: int = Field(..., ge=0, description="Total claims stripped or redlined")
    factual_consistency_score: float = Field(..., ge=0.0, le=1.0, description="Factual consistency ratio: entailed / total")
    hallucination_catch_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Catch rate in benchmark/synthetic settings")
    safety_status: SafetyStatus = Field(..., description="PASSED, FLAGGED, or FAILED based on safety thresholds")
    alerts: List[HallucinationAlert] = Field(default_factory=list, description="Aggregated list of all safety alerts")
    processing_time_ms: float = Field(..., ge=0.0, description="Total pipeline latency in milliseconds")


class SummaryRequest(BaseModel):
    """Input payload for clinical or technical document summarization."""
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=10, description="Full source document text (clinical note or technical spec)")
    document_id: Optional[str] = Field(None, description="Optional external document reference ID")
    doc_type: str = Field("clinical", pattern="^(clinical|technical)$", description="Document domain")
    strict_redlining: bool = Field(True, description="If True, removes/strikes out ungrounded neutral and contradiction claims")
    min_entailment_confidence: float = Field(0.65, ge=0.0, le=1.0, description="Confidence threshold for ENTAILMENT acceptance")
    simulation_hallucination_type: Optional[str] = Field(
        None,
        description="Optional stress-test injection: 'dosage_mutation', 'negation_flip', 'lab_fabrication', 'entity_swap', 'unsupported_claim'"
    )


class SummaryResponse(BaseModel):
    """Output payload delivering vetted summary, provenance citations, and audit report."""
    model_config = ConfigDict(extra="forbid")

    document_id: Optional[str] = Field(None, description="Document ID reference")
    raw_summary: str = Field(..., description="Original draft summary generated prior to guardrail filtering")
    vetted_summary: str = Field(..., description="Safety-filtered summary with contradictions/hallucinations excised or redlined")
    attributed_summary: str = Field(..., description="Summary annotated with exact source sentence citations, e.g. [src_0]")
    verified_claims: List[VerifiedClaim] = Field(..., description="Audit breakdown for each atomic claim")
    audit_report: AuditReport = Field(..., description="Document-level safety audit summary")


class VerifyClaimsRequest(BaseModel):
    """Direct claims verification request."""
    model_config = ConfigDict(extra="forbid")

    source_text: str = Field(..., min_length=10, description="Reference ground-truth source text")
    claims: List[str] = Field(..., min_length=1, description="List of candidate claims to verify")
    strict_numeric: bool = Field(True, description="Enforce strict numeric and unit equality checks")


class VerifyClaimsResponse(BaseModel):
    """Response containing individual claim decisions and document-level audit."""
    model_config = ConfigDict(extra="forbid")

    verified_claims: List[VerifiedClaim] = Field(..., description="Verification details for each submitted claim")
    audit_report: AuditReport = Field(..., description="Consolidated audit report")


class HealthResponse(BaseModel):
    """System health and runtime status."""
    model_config = ConfigDict(extra="forbid")

    status: str = Field("healthy", description="Service health state")
    simulation_mode: bool = Field(True, description="Whether CPU simulation mode is active")
    version: str = Field("1.0.0", description="VeriMed-NLI software version")
    timestamp: str = Field(..., description="Current ISO 8601 UTC timestamp")


class MetricsResponse(BaseModel):
    """Prometheus-style runtime engine metrics."""
    model_config = ConfigDict(extra="forbid")

    requests_processed: int = Field(..., ge=0, description="Total requests processed")
    claims_verified: int = Field(..., ge=0, description="Total individual claims verified")
    hallucinations_detected: int = Field(..., ge=0, description="Total hallucinations caught and prevented")
    contradictions_caught: int = Field(..., ge=0, description="Count of hard factual contradictions")
    negations_flipped_caught: int = Field(..., ge=0, description="Count of flipped clinical negations detected")
    dosages_altered_caught: int = Field(..., ge=0, description="Count of altered dosages caught")
    fabricated_labs_caught: int = Field(..., ge=0, description="Count of fabricated lab values caught")
    avg_latency_ms: float = Field(..., ge=0.0, description="Average request latency in milliseconds")

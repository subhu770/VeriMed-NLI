"""FastAPI application exposing /summarize, /verify-claims, /health, and /metrics endpoints."""

import os
import time
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import (
    SummaryRequest,
    SummaryResponse,
    VerifyClaimsRequest,
    VerifyClaimsResponse,
    HealthResponse,
    MetricsResponse,
    AtomicClaim,
    EntailmentDecision,
    HallucinationType,
)
from src.pipeline.summarizer import ClinicalSummarizerPipeline


app = FastAPI(
    title="VeriMed-NLI",
    description="Production-grade, hallucination-guarded clinical and technical document summarization engine.",
    version="1.0.0",
)

# Enable CORS for web and microservice integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline engine singleton
pipeline = ClinicalSummarizerPipeline()

# In-memory Prometheus-style telemetry metrics
engine_metrics: Dict[str, Any] = {
    "requests_processed": 0,
    "claims_verified": 0,
    "hallucinations_detected": 0,
    "contradictions_caught": 0,
    "negations_flipped_caught": 0,
    "dosages_altered_caught": 0,
    "fabricated_labs_caught": 0,
    "total_latency_ms": 0.0,
}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return operational health status and simulation mode state."""
    return HealthResponse(
        status="healthy",
        simulation_mode=pipeline.verifier.simulation_mode,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Telemetry"])
async def get_metrics() -> MetricsResponse:
    """Return runtime performance and hallucination prevention counters."""
    reqs = engine_metrics["requests_processed"]
    avg_lat = (engine_metrics["total_latency_ms"] / reqs) if reqs > 0 else 0.0

    return MetricsResponse(
        requests_processed=reqs,
        claims_verified=engine_metrics["claims_verified"],
        hallucinations_detected=engine_metrics["hallucinations_detected"],
        contradictions_caught=engine_metrics["contradictions_caught"],
        negations_flipped_caught=engine_metrics["negations_flipped_caught"],
        dosages_altered_caught=engine_metrics["dosages_altered_caught"],
        fabricated_labs_caught=engine_metrics["fabricated_labs_caught"],
        avg_latency_ms=round(avg_lat, 2)
    )


@app.post("/summarize", response_model=SummaryResponse, tags=["Summarization"])
async def summarize_document(request: SummaryRequest) -> SummaryResponse:
    """Generate structured summary with atomic claim decomposition, NLI verification, and redlining."""
    try:
        t0 = time.perf_counter()
        response = pipeline.summarize_and_audit(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Update telemetry
        engine_metrics["requests_processed"] += 1
        engine_metrics["total_latency_ms"] += elapsed_ms
        engine_metrics["claims_verified"] += len(response.verified_claims)

        for claim in response.verified_claims:
            if claim.decision == EntailmentDecision.CONTRADICTION or claim.decision == EntailmentDecision.NEUTRAL:
                engine_metrics["hallucinations_detected"] += 1
            if claim.decision == EntailmentDecision.CONTRADICTION:
                engine_metrics["contradictions_caught"] += 1
            for alert in claim.alerts:
                if alert.alert_type == HallucinationType.NEGATION_FLIP:
                    engine_metrics["negations_flipped_caught"] += 1
                elif alert.alert_type == HallucinationType.DOSAGE_ALTERATION:
                    engine_metrics["dosages_altered_caught"] += 1
                elif alert.alert_type == HallucinationType.FABRICATED_LAB:
                    engine_metrics["fabricated_labs_caught"] += 1

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical summarization engine error: {str(e)}"
        )


@app.post("/verify-claims", response_model=VerifyClaimsResponse, tags=["Verification"])
async def verify_claims_direct(request: VerifyClaimsRequest) -> VerifyClaimsResponse:
    """Directly verify a list of atomic claims against source reference text."""
    try:
        t0 = time.perf_counter()
        source_sentences = pipeline.splitter.split(request.source_text)
        if not source_sentences:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Source text cannot be empty or contain zero parseable sentences."
            )

        # Decompose input claims into structured atomic claim representations
        atomic_claims = []
        for i, c_str in enumerate(request.claims):
            sub_claims = pipeline.decomposer.decompose(c_str)
            if sub_claims:
                for sc in sub_claims:
                    atomic_claims.append(sc)
            else:
                atomic_claims.append(AtomicClaim(
                    claim_id=f"claim_{i}",
                    claim_text=c_str,
                    extracted_entities=[],
                    extracted_quantities=[],
                    is_negated=False,
                    negation_cues=[]
                ))

        verified = pipeline.verifier.verify_all_claims(atomic_claims, source_sentences)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        audit_report = pipeline.verifier.build_audit_report(
            verified_claims=verified,
            processing_time_ms=elapsed_ms
        )

        # Update metrics
        engine_metrics["requests_processed"] += 1
        engine_metrics["total_latency_ms"] += elapsed_ms
        engine_metrics["claims_verified"] += len(verified)

        for claim in verified:
            if claim.decision != EntailmentDecision.ENTAILED:
                engine_metrics["hallucinations_detected"] += 1
            if claim.decision == EntailmentDecision.CONTRADICTION:
                engine_metrics["contradictions_caught"] += 1

        return VerifyClaimsResponse(
            verified_claims=verified,
            audit_report=audit_report
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Claims verification error: {str(e)}"
        )

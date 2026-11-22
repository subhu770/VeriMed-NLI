"""End-to-end FastAPI tests for summarization, direct claim verification, and redlining audits."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.schemas import SafetyStatus, EntailmentDecision


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


SAMPLE_CLINICAL_NOTE = (
    "Patient: Maria Garcia, 62 years old. "
    "Admission Diagnosis: Type 2 diabetes mellitus with hyperglycemic decompensation. "
    "Vitals on presentation showed blood pressure 128/82 mmHg and heart rate 74 bpm. "
    "Laboratory evaluation on discharge revealed HbA1c of 7.1%. "
    "Urine analysis showed no evidence of diabetic ketoacidosis or ketones. "
    "Patient was stabilized and discharged on Metformin 500mg orally daily. "
    "Follow-up scheduled with primary care clinic in two weeks."
)


class TestHealthAndMetricsEndpoints:
    """Tests /health and /metrics endpoints."""

    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["simulation_mode"] is True
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    def test_metrics_endpoint(self, client: TestClient):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_processed" in data
        assert "claims_verified" in data
        assert "hallucinations_detected" in data
        assert "avg_latency_ms" in data


class TestSummarizeEndpoint:
    """Tests the /summarize pipeline endpoint including automated redlining and citation attribution."""

    def test_clean_summarize_success(self, client: TestClient):
        payload = {
            "text": SAMPLE_CLINICAL_NOTE,
            "document_id": "TEST_DOC_001",
            "doc_type": "clinical",
            "strict_redlining": True,
            "min_entailment_confidence": 0.65
        }
        response = client.post("/summarize", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["document_id"] == "TEST_DOC_001"
        assert "vetted_summary" in data
        assert "attributed_summary" in data
        assert "[src_" in data["attributed_summary"]
        assert len(data["verified_claims"]) >= 1

        audit = data["audit_report"]
        assert audit["total_claims"] >= 1
        assert audit["entailed_claims_count"] >= 1
        assert audit["contradiction_claims_count"] == 0
        assert audit["safety_status"] == SafetyStatus.PASSED.value

    def test_summarize_with_dosage_mutation_injection(self, client: TestClient):
        payload = {
            "text": SAMPLE_CLINICAL_NOTE,
            "document_id": "TEST_DOC_002",
            "doc_type": "clinical",
            "strict_redlining": True,
            "simulation_hallucination_type": "dosage_mutation"
        }
        response = client.post("/summarize", json=payload)
        assert response.status_code == 200
        data = response.json()

        audit = data["audit_report"]
        assert audit["contradiction_claims_count"] >= 1
        assert audit["safety_status"] == SafetyStatus.FAILED.value

        # Check redline formatting
        vetted = data["vetted_summary"]
        assert "~~" in vetted or "[REJECTED: CONTRADICTION" in vetted

    def test_summarize_with_negation_flip_injection(self, client: TestClient):
        payload = {
            "text": SAMPLE_CLINICAL_NOTE,
            "document_id": "TEST_DOC_003",
            "doc_type": "clinical",
            "strict_redlining": True,
            "simulation_hallucination_type": "negation_flip"
        }
        response = client.post("/summarize", json=payload)
        assert response.status_code == 200
        data = response.json()

        audit = data["audit_report"]
        assert audit["contradiction_claims_count"] >= 1
        assert audit["safety_status"] == SafetyStatus.FAILED.value
        assert any(a["alert_type"] == "NEGATION_FLIP" for a in audit["alerts"])


class TestVerifyClaimsEndpoint:
    """Tests the /verify-claims direct validation endpoint."""

    def test_verify_claims_endpoint(self, client: TestClient):
        payload = {
            "source_text": SAMPLE_CLINICAL_NOTE,
            "claims": [
                "Laboratory evaluation on discharge revealed HbA1c of 7.1%.",
                "Patient was discharged on Metformin 1000mg orally daily.",  # Contradiction
                "Patient had emergency abdominal surgery."  # Ungrounded neutral
            ],
            "strict_numeric": True
        }
        response = client.post("/verify-claims", json=payload)
        assert response.status_code == 200
        data = response.json()

        claims = data["verified_claims"]
        assert len(claims) == 3

        assert claims[0]["decision"] == EntailmentDecision.ENTAILED.value
        assert claims[1]["decision"] == EntailmentDecision.CONTRADICTION.value
        assert claims[2]["decision"] == EntailmentDecision.NEUTRAL.value

        report = data["audit_report"]
        assert report["entailed_claims_count"] == 1
        assert report["contradiction_claims_count"] == 1
        assert report["neutral_claims_count"] == 1


class TestValidationErrors:
    """Verifies schema validation and boundary checks."""

    def test_short_text_rejected(self, client: TestClient):
        payload = {"text": "Short"}
        response = client.post("/summarize", json=payload)
        assert response.status_code == 422  # Pydantic v2 validation error

    def test_invalid_doc_type_rejected(self, client: TestClient):
        payload = {
            "text": SAMPLE_CLINICAL_NOTE,
            "doc_type": "financial"
        }
        response = client.post("/summarize", json=payload)
        assert response.status_code == 422

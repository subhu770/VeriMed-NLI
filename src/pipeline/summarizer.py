"""Clinical and technical document summarization pipeline with controlled hallucination injection and automated redlining."""

import re
import time
from typing import List, Optional, Tuple, Dict, Any
from src.api.schemas import (
    SummaryRequest,
    SummaryResponse,
    VerifiedClaim,
    AuditReport,
    EntailmentDecision,
    HallucinationType,
    AlertSeverity,
    HallucinationAlert,
)
from src.pipeline.claim_decomposer import ClaimDecomposer, SentenceSplitter, ClinicalUnitExtractor
from src.verification.nli_verifier import NLIVerifier


class ClinicalSummarizerPipeline:
    """End-to-end clinical and technical document summarizer with audit guardrails."""

    def __init__(
        self,
        simulation_mode: Optional[bool] = None,
        min_entailment_confidence: float = 0.65
    ) -> None:
        self.decomposer = ClaimDecomposer()
        self.splitter = SentenceSplitter()
        self.extractor = ClinicalUnitExtractor()
        self.verifier = NLIVerifier(
            simulation_mode=simulation_mode,
            min_entailment_confidence=min_entailment_confidence
        )

    def generate_draft_summary(self, source_text: str, doc_type: str = "clinical") -> str:
        """Synthesize an initial factual structured draft summary from source text."""
        sentences = self.splitter.split(source_text)
        if not sentences:
            return ""

        if doc_type == "clinical":
            # Extract key sections: Admission/Diagnosis, Labs/Vitals, Medications, Disposition
            findings: List[str] = []
            meds: List[str] = []
            labs: List[str] = []

            for s in sentences:
                s_lower = s.lower()
                # Medications
                if any(drug in s_lower for drug in ClinicalUnitExtractor.KNOWN_DRUGS):
                    meds.append(s)
                # Labs
                elif any(lab in s_lower for lab in ClinicalUnitExtractor.KNOWN_LABS):
                    labs.append(s)
                # Key clinical observations
                elif any(term in s_lower for term in ["admitted", "diagnosed", "denies", "no evidence", "presenting", "improved", "discharge"]):
                    findings.append(s)

            summary_parts: List[str] = []
            if findings:
                summary_parts.append(findings[0])
                if len(findings) > 1:
                    summary_parts.append(findings[1])
            if labs:
                summary_parts.append(labs[0])
            if meds:
                summary_parts.append(meds[0])
                if len(meds) > 1:
                    summary_parts.append(meds[1])

            # Fallback if no specific tags matched
            if not summary_parts:
                summary_parts = sentences[:min(4, len(sentences))]

            return " ".join(summary_parts)

        else:
            # Technical document summary
            tech_parts: List[str] = []
            for s in sentences:
                s_lower = s.lower()
                if any(w in s_lower for w in ["system", "architecture", "database", "latency", "throughput", "cluster", "spec"]):
                    tech_parts.append(s)

            if not tech_parts:
                tech_parts = sentences[:min(4, len(sentences))]

            return " ".join(tech_parts[:4])

    def inject_hallucination(
        self,
        summary: str,
        injection_type: str,
        source_text: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Inject controlled adversarial corruptions for stress testing."""
        if not injection_type or injection_type.lower() == "none":
            return summary, None

        inj = injection_type.lower()
        sentences = self.splitter.split(summary)
        if not sentences:
            return summary, None

        target_idx = 0
        target_sent = sentences[0]
        mutation_meta: Dict[str, Any] = {"injection_type": inj}

        if inj == "dosage_mutation":
            # Find a dosage in the summary and alter it
            for i, s in enumerate(sentences):
                tokens = self.extractor.extract_tokens(s)
                dosages = [t for t in tokens if t.token_type.value == "dosage"]
                if dosages:
                    t = dosages[0]
                    # Double or halve dosage
                    original_val = t.numeric_val or 100.0
                    mutated_val = int(original_val * 2) if original_val < 500 else int(original_val / 2)
                    mutated_token = f"{mutated_val}{t.unit or 'mg'}"
                    mutated_sent = s.replace(t.raw_text, mutated_token, 1)
                    sentences[i] = mutated_sent
                    mutation_meta.update({
                        "original_span": t.raw_text,
                        "mutated_span": mutated_token,
                        "target_sentence": mutated_sent
                    })
                    return " ".join(sentences), mutation_meta

            # Fallback dosage injection
            fallback = "Patient was prescribed Metformin 1000mg orally twice daily."
            sentences.append(fallback)
            mutation_meta["injected_sentence"] = fallback
            return " ".join(sentences), mutation_meta

        elif inj == "negation_flip":
            # Invert negation polarity (e.g. 'no evidence of pneumonia' -> 'confirmed acute pneumonia')
            for i, s in enumerate(sentences):
                s_lower = s.lower()
                if "no evidence of" in s_lower:
                    mutated = re.sub(r'\bno evidence of\b', 'confirmed acute evidence of', s, flags=re.IGNORECASE)
                    sentences[i] = mutated
                    mutation_meta["mutated_sentence"] = mutated
                    return " ".join(sentences), mutation_meta
                elif "denies" in s_lower:
                    mutated = re.sub(r'\bdenies\b', 'reports severe acute', s, flags=re.IGNORECASE)
                    sentences[i] = mutated
                    mutation_meta["mutated_sentence"] = mutated
                    return " ".join(sentences), mutation_meta
                elif "negative for" in s_lower:
                    mutated = re.sub(r'\bnegative for\b', 'positive for acute', s, flags=re.IGNORECASE)
                    sentences[i] = mutated
                    mutation_meta["mutated_sentence"] = mutated
                    return " ".join(sentences), mutation_meta
                elif "without" in s_lower:
                    mutated = re.sub(r'\bwithout\b', 'with active', s, flags=re.IGNORECASE)
                    sentences[i] = mutated
                    mutation_meta["mutated_sentence"] = mutated
                    return " ".join(sentences), mutation_meta

            # Invert an affirmative sentence to negative
            mutated = f"Patient denies any history or evidence of chest pain."
            sentences.append(mutated)
            mutation_meta["injected_sentence"] = mutated
            return " ".join(sentences), mutation_meta

        elif inj == "lab_fabrication":
            # Fabricate critical potassium or troponin
            for i, s in enumerate(sentences):
                tokens = self.extractor.extract_tokens(s)
                labs = [t for t in tokens if t.token_type.value == "lab_value"]
                if labs:
                    t = labs[0]
                    mutated_val = 6.9 if "potassium" in (t.context_entity or "").lower() else 9.8
                    mutated_token = f"{t.context_entity or 'Lab'} of {mutated_val} {t.unit or ''}".strip()
                    mutated_sent = s.replace(t.raw_text, mutated_token, 1)
                    sentences[i] = mutated_sent
                    mutation_meta["mutated_sentence"] = mutated_sent
                    return " ".join(sentences), mutation_meta

            fabricated_sent = "Discharge lab showed critical potassium of 6.8 mEq/L."
            sentences.append(fabricated_sent)
            mutation_meta["injected_sentence"] = fabricated_sent
            return " ".join(sentences), mutation_meta

        elif inj == "entity_swap":
            # Swap drug or entity
            for i, s in enumerate(sentences):
                s_lower = s.lower()
                if "lisinopril" in s_lower:
                    sentences[i] = re.sub(r'\blisinopril\b', 'Metoprolol', s, flags=re.IGNORECASE)
                    mutation_meta["swapped"] = "Lisinopril -> Metoprolol"
                    return " ".join(sentences), mutation_meta
                elif "metformin" in s_lower:
                    sentences[i] = re.sub(r'\bmetformin\b', 'Glipizide', s, flags=re.IGNORECASE)
                    mutation_meta["swapped"] = "Metformin -> Glipizide"
                    return " ".join(sentences), mutation_meta

            # Swap technical spec
            fabricated_tech = "Database architecture relies on Apache Cassandra with multi-master replication."
            sentences.append(fabricated_tech)
            mutation_meta["injected_sentence"] = fabricated_tech
            return " ".join(sentences), mutation_meta

        elif inj == "unsupported_claim":
            fabricated = "Patient developed acute bilateral pulmonary embolism requiring emergent catheter-directed thrombolysis."
            sentences.append(fabricated)
            mutation_meta["injected_sentence"] = fabricated
            return " ".join(sentences), mutation_meta

        return " ".join(sentences), mutation_meta

    def summarize_and_audit(self, request: SummaryRequest) -> SummaryResponse:
        """Run complete summarization, claim decomposition, NLI verification, and redlining."""
        start_time = time.perf_counter()

        # Step 1: Split source text into indexed sentences [src_0, src_1, ...]
        source_sentences = self.splitter.split(request.text)

        # Step 2: Generate draft summary
        draft_summary = self.generate_draft_summary(request.text, request.doc_type)

        # Step 3: Inject simulated hallucination if requested
        if request.simulation_hallucination_type:
            raw_summary, _ = self.inject_hallucination(
                draft_summary,
                request.simulation_hallucination_type,
                request.text
            )
        else:
            raw_summary = draft_summary

        # Step 4: Decompose summary into atomic claims
        atomic_claims = self.decomposer.decompose(raw_summary)

        # Step 5: Verify all claims against indexed source sentences
        verified_claims = self.verifier.verify_all_claims(atomic_claims, source_sentences)

        # Step 6: Construct Attributed Summary and Vetted / Redlined Summary
        attributed_parts: List[str] = []
        vetted_parts: List[str] = []

        for vc in verified_claims:
            if vc.decision == EntailmentDecision.ENTAILED:
                cite_tag = f"[src_{vc.best_source_sent_id}]" if vc.best_source_sent_id is not None else "[src]"
                attributed_parts.append(f"{vc.claim_text} {cite_tag}")
                vetted_parts.append(f"{vc.claim_text} {cite_tag}")
            elif vc.decision == EntailmentDecision.CONTRADICTION:
                attributed_parts.append(f"{vc.claim_text} [CONTRADICTION: {vc.redline_reason or 'Factual Conflict'}]")
                if request.strict_redlining:
                    vetted_parts.append(f"~~{vc.claim_text}~~ [REJECTED: CONTRADICTION - {vc.redline_reason}]")
                else:
                    vetted_parts.append(f"[WARNING: CONTRADICTION] {vc.claim_text}")
            else:  # NEUTRAL
                attributed_parts.append(f"{vc.claim_text} [UNGROUNDED]")
                if request.strict_redlining:
                    vetted_parts.append(f"~~{vc.claim_text}~~ [REJECTED: UNGROUNDED - Lacks source citation]")
                else:
                    vetted_parts.append(f"[UNGROUNDED] {vc.claim_text}")

        attributed_summary = " ".join(attributed_parts)
        vetted_summary = " ".join(vetted_parts)

        # Step 7: Build comprehensive audit report
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        audit_report = self.verifier.build_audit_report(
            verified_claims=verified_claims,
            processing_time_ms=elapsed_ms,
            document_id=request.document_id
        )

        return SummaryResponse(
            document_id=request.document_id,
            raw_summary=raw_summary,
            vetted_summary=vetted_summary,
            attributed_summary=attributed_summary,
            verified_claims=verified_claims,
            audit_report=audit_report
        )

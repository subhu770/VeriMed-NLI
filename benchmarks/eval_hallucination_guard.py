"""Evaluation benchmark testing Hallucination Catch Rate, FCR, False Positive Redline Rate, and latency on 150 clinical/technical documents."""

import math
import os
import sys
import time
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.schemas import EntailmentDecision
from src.pipeline.claim_decomposer import ClaimDecomposer, SentenceSplitter
from src.simulation.clinical_corpus import ClinicalCorpusGenerator, SyntheticDocument
from src.verification.nli_verifier import NLIVerifier


def run_benchmark(corpus_size: int = 150):
    """Execute rigorous clinical safety and latency evaluations on 150 synthetic documents."""
    print("=" * 78)
    print("  VERIMED-NLI: CLINICAL & TECHNICAL AUDIT GUARDRAIL BENCHMARK SUITE")
    print("=" * 78)
    print(f"Generating synthetic clinical & technical corpus (N = {corpus_size})...")

    generator = ClinicalCorpusGenerator(seed=42)
    corpus: List[SyntheticDocument] = generator.generate_corpus(total_docs=corpus_size)

    decomposer = ClaimDecomposer()
    splitter = SentenceSplitter()
    verifier = NLIVerifier(simulation_mode=True)

    latencies_ms: List[float] = []

    # Metrics accumulators
    total_injected = 0
    caught_hallucinations = 0

    category_stats: Dict[str, Dict[str, int]] = {
        "dosage_mutation": {"total": 0, "caught": 0},
        "negation_flip": {"total": 0, "caught": 0},
        "lab_fabrication": {"total": 0, "caught": 0},
        "ungrounded_claim": {"total": 0, "caught": 0},
    }

    total_clean_claims = 0
    false_positive_redlines = 0

    post_guard_surviving_total = 0
    post_guard_surviving_entailed = 0

    print(f"Running deterministic NLI & numeric verification across {len(corpus)} documents...\n")

    for doc in corpus:
        source_sentences = splitter.split(doc.source_text)

        # 1. Clean Summary Evaluation (Ground truth consistency & False Positive check)
        clean_claims = decomposer.decompose(doc.clean_summary)
        for c in clean_claims:
            total_clean_claims += 1
            res = verifier.verify_claim(c, source_sentences)
            if res.decision != EntailmentDecision.ENTAILED:
                false_positive_redlines += 1

        # 2. Corrupted Summary / Injected Hallucination Evaluation
        t0 = time.perf_counter()
        corrupted_claims = decomposer.decompose(doc.corrupted_summary)
        verified_corrupted = verifier.verify_all_claims(corrupted_claims, source_sentences)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        # Track post-guardrail surviving claims
        for vc in verified_corrupted:
            if not vc.is_redlined:
                post_guard_surviving_total += 1
                if vc.decision == EntailmentDecision.ENTAILED:
                    post_guard_surviving_entailed += 1

        # Check injected hallucinations caught
        inj_type = doc.injected_hallucination_type
        category_stats[inj_type]["total"] += 1
        total_injected += 1

        # Verify that the corrupted summary triggered at least one rejection (contradiction or ungrounded)
        caught = any(
            vc.is_redlined and (vc.decision in (EntailmentDecision.CONTRADICTION, EntailmentDecision.NEUTRAL))
            for vc in verified_corrupted
        )
        if caught:
            caught_hallucinations += 1
            category_stats[inj_type]["caught"] += 1

    # Statistical Computations
    latencies_ms.sort()
    p50 = latencies_ms[int(0.50 * len(latencies_ms))]
    p90 = latencies_ms[int(0.90 * len(latencies_ms))]
    p95 = latencies_ms[int(0.95 * len(latencies_ms))]
    p99 = latencies_ms[int(0.99 * len(latencies_ms))]
    avg_latency = sum(latencies_ms) / len(latencies_ms)

    catch_rate = (caught_hallucinations / total_injected) * 100.0 if total_injected > 0 else 0.0
    fpr_rate = (false_positive_redlines / total_clean_claims) * 100.0 if total_clean_claims > 0 else 0.0
    fcr_post = (post_guard_surviving_entailed / post_guard_surviving_total) * 100.0 if post_guard_surviving_total > 0 else 100.0

    # Presentation Table
    print("+" + "-" * 76 + "+")
    print(f"| {'METRIC':<46} | {'MEASURED VALUE':<25} |")
    print("+" + "-" * 76 + "+")
    print(f"| {'Documents Evaluated':<46} | {len(corpus):<25} |")
    print(f"| {'Total Injected Adversarial Hallucinations':<46} | {total_injected:<25} |")
    print(f"| {'Hallucinations Caught & Redlined':<46} | {caught_hallucinations:<25} |")
    print(f"| {'Hallucination Catch Rate (Recall)':<46} | {catch_rate:>23.2f}% |")
    print(f"| {'Total Clean Ground-Truth Claims Tested':<46} | {total_clean_claims:<25} |")
    print(f"| {'False Positive Redlines (Clean Claims)':<46} | {false_positive_redlines:<25} |")
    print(f"| {'False Positive Redline Rate (Target < 2%)':<46} | {fpr_rate:>23.2f}% |")
    print(f"| {'Factual Consistency Rate Post-Guardrail':<46} | {fcr_post:>23.2f}% |")
    print("+" + "-" * 76 + "+")
    print(f"| {'Average CPU Latency per Document':<46} | {avg_latency:>21.2f} ms |")
    print(f"| {'P50 (Median) Latency':<46} | {p50:>21.2f} ms |")
    print(f"| {'P90 Latency':<46} | {p90:>21.2f} ms |")
    print(f"| {'P95 Latency':<46} | {p95:>21.2f} ms |")
    print(f"| {'P99 Latency':<46} | {p99:>21.2f} ms |")
    print("+" + "-" * 76 + "+")

    print("\nHALLUCINATION CATCH RATE BY TAXONOMY SUBTYPE:")
    print("+" + "-" * 76 + "+")
    print(f"| {'Hallucination Category':<30} | {'Total Injected':<18} | {'Catch Rate':<22} |")
    print("+" + "-" * 76 + "+")
    for cat_name, stats in category_stats.items():
        tot = stats["total"]
        cght = stats["caught"]
        pct = (cght / tot * 100.0) if tot > 0 else 0.0
        print(f"| {cat_name:<30} | {tot:<18} | {cght}/{tot} ({pct:.1f}%)" + " " * (10 - len(f"{cght}/{tot} ({pct:.1f}%)")) + " |")
    print("+" + "-" * 76 + "+")

    # Safety Assertions
    assert catch_rate >= 98.0, f"Safety Failure: Catch rate {catch_rate:.2f}% below 98.0% requirement"
    assert fpr_rate <= 2.0, f"Precision Failure: False positive rate {fpr_rate:.2f}% above 2.0% threshold"
    assert fcr_post >= 99.0, f"Post-guardrail factual consistency {fcr_post:.2f}% below 99.0%"

    print("\n>>> ALL CLINICAL SAFETY CONSTRAINTS & AUDIT THRESHOLDS SATISFIED <<<")
    return {
        "catch_rate": catch_rate,
        "false_positive_rate": fpr_rate,
        "post_guard_fcr": fcr_post,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }


if __name__ == "__main__":
    run_benchmark(corpus_size=150)

"""Synthetic clinical discharge note and technical specification corpus generator (150 documents) with controlled hallucinations."""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SyntheticDocument:
    """Synthetic clinical or technical document with ground-truth and corrupted summaries."""
    doc_id: str
    doc_type: str
    specialty_or_domain: str
    source_text: str
    clean_summary: str
    corrupted_summary: str
    injected_hallucination_type: str
    ground_truth_clean_claims: List[str] = field(default_factory=list)
    injected_hallucination_claims: List[Dict[str, Any]] = field(default_factory=list)


class ClinicalCorpusGenerator:
    """Procedural generator for 150 clinical discharge summaries and technical specifications."""

    CLINICAL_SPECIALTIES = [
        "Cardiology", "Endocrinology", "Pulmonology", "Nephrology",
        "Neurology", "Infectious Disease", "General Surgery", "Emergency Medicine"
    ]

    FIRST_NAMES = ["James", "Maria", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth"]
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate_corpus(self, total_docs: int = 150) -> List[SyntheticDocument]:
        """Generate exactly total_docs (default 150) balanced between clinical and technical documents."""
        documents: List[SyntheticDocument] = []

        # 100 Clinical documents + 50 Technical specifications
        clinical_count = min(100, int(total_docs * 2 / 3))
        technical_count = total_docs - clinical_count

        for i in range(1, clinical_count + 1):
            doc_id = f"CLIN_{i:03d}"
            documents.append(self._generate_clinical_doc(doc_id, i))

        for j in range(1, technical_count + 1):
            doc_id = f"TECH_{j:03d}"
            documents.append(self._generate_technical_doc(doc_id, j))

        return documents

    def _generate_clinical_doc(self, doc_id: str, index: int) -> SyntheticDocument:
        """Generate a single structured clinical discharge summary."""
        specialty = self.CLINICAL_SPECIALTIES[(index - 1) % len(self.CLINICAL_SPECIALTIES)]
        patient_name = f"{self.rng.choice(self.FIRST_NAMES)} {self.rng.choice(self.LAST_NAMES)}"
        age = self.rng.randint(42, 84)

        # Varied parameters per specialty
        if specialty == "Cardiology":
            drug = "Metoprolol"
            true_dosage = 50
            mutated_dosage = 150
            unit = "mg"
            lab_name = "Troponin"
            true_lab = 0.03
            mutated_lab = 2.45
            lab_unit = "ng/mL"
            condition = "acute chest pain"
            negation_phrase = "Patient denies chest pain and shortness of breath upon discharge."
            flipped_negation = "Patient reports persistent chest pain and severe shortness of breath upon discharge."
            diagnosis = "Non-ST elevation myocardial infarction, stabilized medically"

        elif specialty == "Endocrinology":
            drug = "Metformin"
            true_dosage = 500
            mutated_dosage = 1000
            unit = "mg"
            lab_name = "HbA1c"
            true_lab = 7.1
            mutated_lab = 11.8
            lab_unit = "%"
            condition = "diabetic ketoacidosis"
            negation_phrase = "Urine analysis showed no evidence of diabetic ketoacidosis or ketones."
            flipped_negation = "Urine analysis confirmed acute evidence of diabetic ketoacidosis and ketones."
            diagnosis = "Type 2 diabetes mellitus with hyperglycemic decompensation"

        elif specialty == "Pulmonology":
            drug = "Azithromycin"
            true_dosage = 250
            mutated_dosage = 750
            unit = "mg"
            lab_name = "WBC"
            true_lab = 11.2
            mutated_lab = 24.5
            lab_unit = "k/uL"
            condition = "pleural effusion"
            negation_phrase = "Repeat chest radiograph showed no evidence of pneumonia or pleural effusion."
            flipped_negation = "Repeat chest radiograph confirmed evidence of severe pneumonia and pleural effusion."
            diagnosis = "Community-acquired bacterial pneumonia, clinical resolution"

        elif specialty == "Nephrology":
            drug = "Lisinopril"
            true_dosage = 10
            mutated_dosage = 40
            unit = "mg"
            lab_name = "Potassium"
            true_lab = 4.3
            mutated_lab = 6.9
            lab_unit = "mEq/L"
            condition = "hyperkalemia"
            negation_phrase = "Patient showed no evidence of hyperkalemia or peripheral edema."
            flipped_negation = "Patient showed confirmed evidence of critical hyperkalemia and peripheral edema."
            diagnosis = "Acute kidney injury stage 2, resolved to baseline creatinine"

        elif specialty == "Neurology":
            drug = "Aspirin"
            true_dosage = 81
            mutated_dosage = 325
            unit = "mg"
            lab_name = "Platelets"
            true_lab = 210
            mutated_lab = 45
            lab_unit = "k/uL"
            condition = "acute intracranial hemorrhage"
            negation_phrase = "Brain MRI demonstrated no evidence of intracranial hemorrhage or mass effect."
            flipped_negation = "Brain MRI demonstrated active evidence of acute intracranial hemorrhage."
            diagnosis = "Transient ischemic attack, carotid ultrasound unremarkable"

        elif specialty == "Infectious Disease":
            drug = "Ciprofloxacin"
            true_dosage = 500
            mutated_dosage = 100
            unit = "mg"
            lab_name = "Lactate"
            true_lab = 1.2
            mutated_lab = 4.8
            lab_unit = "mmol/L"
            condition = "bacteremia"
            negation_phrase = "Blood cultures remained negative for bacteremia after 48 hours."
            flipped_negation = "Blood cultures tested positive for bacteremia after 48 hours."
            diagnosis = "Complicated urinary tract infection with resolved sepsis"

        elif specialty == "General Surgery":
            drug = "Enoxaparin"
            true_dosage = 40
            mutated_dosage = 120
            unit = "mg"
            lab_name = "Hemoglobin"
            true_lab = 12.8
            mutated_lab = 6.2
            lab_unit = "g/dL"
            condition = "deep vein thrombosis"
            negation_phrase = "Lower extremity Doppler was negative for deep vein thrombosis."
            flipped_negation = "Lower extremity Doppler was positive for acute deep vein thrombosis."
            diagnosis = "Status post laparoscopic cholecystectomy, uncomplicated course"

        else:  # Emergency Medicine
            drug = "Atorvastatin"
            true_dosage = 40
            mutated_dosage = 10
            unit = "mg"
            lab_name = "Creatinine"
            true_lab = 1.1
            mutated_lab = 3.6
            lab_unit = "mg/dL"
            condition = "active ischemia"
            negation_phrase = "Serial electrocardiograms were unremarkable for acute ischemic changes."
            flipped_negation = "Serial electrocardiograms showed active acute ischemic changes."
            diagnosis = "Hypertensive urgency with rapid response to oral therapy"

        # Build Source Document Sentences
        source_sentences = [
            f"Patient: {patient_name}, {age} years old.",
            f"Admission Diagnosis: {diagnosis}.",
            f"Vitals on presentation showed blood pressure 128/82 mmHg and heart rate 74 bpm.",
            f"Laboratory evaluation on discharge revealed {lab_name} of {true_lab} {lab_unit}.",
            negation_phrase,
            f"Patient was stabilized and discharged on {drug} {true_dosage}{unit} orally daily.",
            "Follow-up scheduled with primary care clinic in two weeks."
        ]
        source_text = " ".join(source_sentences)

        # Ground truth clean summary
        clean_summary = (
            f"Patient was admitted with {diagnosis}. "
            f"Discharge lab confirmed {lab_name} of {true_lab} {lab_unit}. "
            f"{negation_phrase} "
            f"Discharged on {drug} {true_dosage}{unit} daily."
        )

        # Injected Hallucination Category (Cyclic rotation across hallucination taxonomy)
        hallucination_cycle = ["dosage_mutation", "negation_flip", "lab_fabrication", "ungrounded_claim"]
        chosen_hallucination = hallucination_cycle[(index - 1) % len(hallucination_cycle)]

        corrupted_summary = clean_summary
        injected_claim_info: Dict[str, Any] = {"hallucination_type": chosen_hallucination}

        if chosen_hallucination == "dosage_mutation":
            corrupted_summary = clean_summary.replace(
                f"{drug} {true_dosage}{unit}",
                f"{drug} {mutated_dosage}{unit}"
            )
            injected_claim_info.update({
                "expected_decision": "CONTRADICTION",
                "hallucination_type": "DOSAGE_ALTERATION",
                "injected_claim": f"Discharged on {drug} {mutated_dosage}{unit} daily."
            })

        elif chosen_hallucination == "negation_flip":
            corrupted_summary = clean_summary.replace(negation_phrase, flipped_negation)
            injected_claim_info.update({
                "expected_decision": "CONTRADICTION",
                "hallucination_type": "NEGATION_FLIP",
                "injected_claim": flipped_negation
            })

        elif chosen_hallucination == "lab_fabrication":
            corrupted_summary = clean_summary.replace(
                f"{lab_name} of {true_lab} {lab_unit}",
                f"{lab_name} of {mutated_lab} {lab_unit}"
            )
            injected_claim_info.update({
                "expected_decision": "CONTRADICTION",
                "hallucination_type": "FABRICATED_LAB",
                "injected_claim": f"Discharge lab confirmed {lab_name} of {mutated_lab} {lab_unit}."
            })

        elif chosen_hallucination == "ungrounded_claim":
            fabrication = "Patient suffered acute cardiac arrest requiring cardiopulmonary resuscitation."
            corrupted_summary = clean_summary + " " + fabrication
            injected_claim_info.update({
                "expected_decision": "NEUTRAL",
                "hallucination_type": "UNGROUNDED_CLAIM",
                "injected_claim": fabrication
            })

        ground_truth_claims = [
            f"Patient was admitted with {diagnosis}.",
            f"Discharge lab confirmed {lab_name} of {true_lab} {lab_unit}.",
            negation_phrase,
            f"Discharged on {drug} {true_dosage}{unit} daily."
        ]

        return SyntheticDocument(
            doc_id=doc_id,
            doc_type="clinical",
            specialty_or_domain=specialty,
            source_text=source_text,
            clean_summary=clean_summary,
            corrupted_summary=corrupted_summary,
            injected_hallucination_type=chosen_hallucination,
            ground_truth_clean_claims=ground_truth_claims,
            injected_hallucination_claims=[injected_claim_info]
        )

    def _generate_technical_doc(self, doc_id: str, index: int) -> SyntheticDocument:
        """Generate a single structured technical engineering document."""
        domains = [
            ("Distributed Storage", "PostgreSQL", "Raft consensus", "15ms", "45ms", "99.99%", "95.0%"),
            ("Telemetry Pipeline", "Apache Kafka", "Snappy compression", "25ms", "120ms", "10000 req/sec", "500 req/sec"),
            ("DICOM PACS Server", "MinIO S3", "AES-256 encryption", "12ms", "90ms", "50 Gbps", "10 Gbps"),
            ("Microservice Mesh", "Envoy Proxy", "Mutual TLS", "5ms", "85ms", "99.95%", "89.0%"),
            ("Clinical ML Engine", "ONNX Runtime", "FP16 quantization", "8ms", "60ms", "500 req/sec", "50 req/sec")
        ]

        domain, comp, feature, true_lat, mut_lat, true_tp, mut_tp = domains[(index - 1) % len(domains)]

        source_sentences = [
            f"System Architecture: {domain} infrastructure deployment.",
            f"Primary persistence is backed by {comp} with {feature}.",
            f"Benchmark evaluations confirmed P95 query latency of {true_lat}.",
            f"System throughput was measured at {true_tp} under peak load.",
            f"The architecture operates without single points of failure across all availability zones.",
            "Disaster recovery targets an RTO of 5 minutes."
        ]
        source_text = " ".join(source_sentences)

        clean_summary = (
            f"System utilizes {comp} with {feature}. "
            f"Benchmark confirmed P95 query latency of {true_lat}. "
            f"System operates without single points of failure. "
            f"Measured throughput reached {true_tp}."
        )

        hallucination_cycle = ["dosage_mutation", "negation_flip", "lab_fabrication", "ungrounded_claim"]
        chosen_hallucination = hallucination_cycle[(index - 1) % len(hallucination_cycle)]

        corrupted_summary = clean_summary
        injected_claim_info: Dict[str, Any] = {"hallucination_type": chosen_hallucination}

        if chosen_hallucination == "dosage_mutation":
            # Metric mutation
            corrupted_summary = clean_summary.replace(true_lat, mut_lat)
            injected_claim_info.update({
                "expected_decision": "CONTRADICTION",
                "hallucination_type": "NUMERIC_MISMATCH",
                "injected_claim": f"Benchmark confirmed P95 query latency of {mut_lat}."
            })

        elif chosen_hallucination == "negation_flip":
            flipped = "System operates with critical single points of failure."
            corrupted_summary = clean_summary.replace("System operates without single points of failure.", flipped)
            injected_claim_info.update({
                "expected_decision": "CONTRADICTION",
                "hallucination_type": "NEGATION_FLIP",
                "injected_claim": flipped
            })

        elif chosen_hallucination == "lab_fabrication":
            corrupted_summary = clean_summary.replace(true_tp, mut_tp)
            injected_claim_info.update({
                "expected_decision": "CONTRADICTION",
                "hallucination_type": "NUMERIC_MISMATCH",
                "injected_claim": f"Measured throughput reached {mut_tp}."
            })

        elif chosen_hallucination == "ungrounded_claim":
            fabrication = "System storage backend was migrated to MongoDB replica set in production."
            corrupted_summary = clean_summary + " " + fabrication
            injected_claim_info.update({
                "expected_decision": "NEUTRAL",
                "hallucination_type": "UNGROUNDED_CLAIM",
                "injected_claim": fabrication
            })

        ground_truth_claims = [
            f"System utilizes {comp} with {feature}.",
            f"Benchmark confirmed P95 query latency of {true_lat}.",
            "System operates without single points of failure.",
            f"Measured throughput reached {true_tp}."
        ]

        return SyntheticDocument(
            doc_id=doc_id,
            doc_type="technical",
            specialty_or_domain=domain,
            source_text=source_text,
            clean_summary=clean_summary,
            corrupted_summary=corrupted_summary,
            injected_hallucination_type=chosen_hallucination,
            ground_truth_clean_claims=ground_truth_claims,
            injected_hallucination_claims=[injected_claim_info]
        )


def generate_benchmark_corpus(count: int = 150) -> List[SyntheticDocument]:
    """Convenience factory function to instantiate 150 benchmark documents."""
    generator = ClinicalCorpusGenerator(seed=42)
    return generator.generate_corpus(total_docs=count)

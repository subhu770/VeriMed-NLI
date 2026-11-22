"""Deterministic NLP sentence splitter, clinical entity/dosage extractor, and atomic claim decomposer."""

import re
from typing import List, Optional, Tuple, Set, Dict, Any
from src.api.schemas import (
    AtomicClaim,
    ClinicalToken,
    ClinicalTokenType,
)


class SentenceSplitter:
    """High-precision sentence tokenizer handling clinical/technical abbreviations."""

    ABBREVIATIONS = {
        "dr", "mr", "ms", "mrs", "vs", "approx", "tab", "cap", "i.e", "e.g",
        "mg", "ml", "pt", "yo", "y.o", "b.i.d", "q.i.d", "p.r.n", "h.s",
        "stat", "no", "vol", "dept", "fig", "ref", "inc", "corp", "co",
        "hr", "min", "sec", "temp", "wt", "ht", "bp", "rr"
    }

    def __init__(self) -> None:
        self._boundary_pattern = re.compile(r'([.!?])(?:\s+|\n+)(?=[A-Z0-9"\'\(\[])')

    def split(self, text: str) -> List[str]:
        """Split text into clean, non-empty sentences respecting clinical abbreviations."""
        if not text or not text.strip():
            return []

        cleaned = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        candidate_sentences: List[str] = []

        for line in lines:
            clean_line = re.sub(r'^(?:[-*•]|\d+\.|\(\d+\)|\[\d+\])\s*', '', line).strip()
            if not clean_line:
                continue

            parts = []
            last_idx = 0
            for match in self._boundary_pattern.finditer(clean_line):
                split_pos = match.start() + 1
                prefix = clean_line[last_idx:split_pos].strip()
                preceding_word = re.findall(r'([a-zA-Z0-9.]+)[.!?]$', prefix)
                if preceding_word:
                    base_word = preceding_word[0].rstrip('.').lower()
                    if base_word in self.ABBREVIATIONS:
                        continue
                parts.append(prefix)
                last_idx = match.end()

            remainder = clean_line[last_idx:].strip()
            if remainder:
                parts.append(remainder)

            if parts:
                candidate_sentences.extend(parts)
            else:
                candidate_sentences.append(clean_line)

        final_sentences = []
        for s in candidate_sentences:
            s_clean = s.strip()
            if len(s_clean) > 3 and not s_clean.startswith(('##', '==', '--')):
                final_sentences.append(s_clean)

        return final_sentences


class ClinicalUnitExtractor:
    """Deterministic extractor and normalizer for dosages, lab values, vitals, and technical units."""

    MASS_NORMALIZATION = {
        "mcg": 0.001,
        "µg": 0.001,
        "ug": 0.001,
        "mg": 1.0,
        "g": 1000.0,
        "kg": 1000000.0,
    }

    VOLUME_NORMALIZATION = {
        "ml": 1.0,
        "l": 1000.0,
    }

    KNOWN_DRUGS = {
        "metformin", "lisinopril", "atorvastatin", "amiodarone", "aspirin",
        "levothyroxine", "apixaban", "amoxicillin", "furosemide", "metoprolol",
        "ciprofloxacin", "vancomycin", "insulin", "heparin", "omeprazole",
        "losartan", "gabapentin", "prednisone", "warfarin", "clopidogrel",
        "hydrochlorothiazide", "amlodipine", "azithromycin", "pantoprazole",
        "carvedilol", "albuterol", "ceftriaxone", "enoxaparin", "potassium chloride"
    }

    KNOWN_LABS = {
        "hba1c": "HbA1c",
        "a1c": "HbA1c",
        "potassium": "Potassium",
        "k+": "Potassium",
        "sodium": "Sodium",
        "na+": "Sodium",
        "creatinine": "Creatinine",
        "cr": "Creatinine",
        "bun": "BUN",
        "wbc": "WBC",
        "white blood cell": "WBC",
        "white blood count": "WBC",
        "hemoglobin": "Hemoglobin",
        "hgb": "Hemoglobin",
        "platelets": "Platelets",
        "troponin": "Troponin",
        "troponin i": "Troponin",
        "troponin t": "Troponin",
        "glucose": "Glucose",
        "ast": "AST",
        "alt": "ALT",
        "egfr": "eGFR",
        "inr": "INR",
        "bnp": "BNP",
        "tsh": "TSH",
        "bilirubin": "Bilirubin",
        "calcium": "Calcium",
        "lactate": "Lactate",
    }

    def __init__(self) -> None:
        # Blood pressure e.g. 120/80 mmHg or 130/85
        self._bp_regex = re.compile(
            r'\b(?:bp|blood pressure)?\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*hg|mmHg)?\b',
            re.IGNORECASE
        )
        # Explicit lab units e.g. 1.1 mg/dL, 4.2 mEq/L, 11.5 k/uL, 2.85 ng/mL, 11,500 /uL
        self._lab_unit_regex = re.compile(
            r'\b(\d+(?:,\d+)*(?:\.\d+)?)\s*'
            r'(mg/dl|g/dl|mmol/l|meq/l|ng/ml|pg/ml|u/l|k/ul|/\s*ul|/\s*µl|mm3|ml/min(?:/1\.73m2)?|uiu/ml)\b',
            re.IGNORECASE
        )
        # Dosage e.g. 500mg, 10 mg, 2.5 mcg, 0.5 g, 20 mEq, 100 units, 2 tablets (not followed by /)
        self._dosage_regex = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|g|kg|ml|meq|mEq|units?|iu|tablets?|tabs?|capsules?|caps?|puffs?)\b(?!\s*[/])',
            re.IGNORECASE
        )
        # Temperature e.g. 98.6 F, 37.2 C, 101 F, 38 °C
        self._temp_regex = re.compile(
            r'\b(\d{2,3}(?:\.\d+)?)\s*(?:°|deg|degrees)?\s*([CF])\b',
            re.IGNORECASE
        )
        # Pulse / Heart rate e.g. HR 72 bpm, pulse 80
        self._hr_regex = re.compile(
            r'\b(?:hr|heart rate|pulse)\s*[:=]?\s*(\d{2,3})\s*(?:bpm)?\b',
            re.IGNORECASE
        )
        # Frequency e.g. twice daily, once daily, q8h, bid, tid, qid, prn
        self._freq_regex = re.compile(
            r'\b(once daily|twice daily|three times daily|four times daily|q\d+h|q\d+-\d+h|bid|b\.i\.d\.|tid|t\.i\.d\.|qid|q\.i\.d\.|prn|p\.r\.n\.|daily|nightly|qhs)\b',
            re.IGNORECASE
        )
        # Percentages e.g. 95%, 40% (no \b after % since % is non-word)
        self._percentage_regex = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*%(?!\w)'
        )
        # Technical values e.g. 500 ms, 10 Gbps, 100 req/sec, 16 vCPU, 256 GB
        self._technical_regex = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*(ms|seconds?|s|gbps|mbps|kbps|gb|mb|tb|req/sec|rps|vcpu|cores?)\b',
            re.IGNORECASE
        )

    def extract_tokens(self, text: str) -> List[ClinicalToken]:
        """Extract all typed clinical and technical tokens from text."""
        tokens: List[ClinicalToken] = []
        covered_spans: Set[Tuple[int, int]] = set()

        def is_overlapping(start: int, end: int) -> bool:
            for s, e in covered_spans:
                if not (end <= s or start >= e):
                    return True
            return False

        # 1. Extract Blood Pressure
        for m in self._bp_regex.finditer(text):
            if is_overlapping(m.start(), m.end()):
                continue
            systolic = float(m.group(1))
            diastolic = float(m.group(2))
            if 50 <= systolic <= 260 and 30 <= diastolic <= 160:
                tokens.append(ClinicalToken(
                    raw_text=m.group(0).strip(),
                    token_type=ClinicalTokenType.BLOOD_PRESSURE,
                    numeric_val=systolic,
                    secondary_val=diastolic,
                    unit="mmHg",
                    normalized_val=systolic,
                    normalized_unit="mmHg",
                    context_entity="blood pressure"
                ))
                covered_spans.add((m.start(), m.end()))

        # 2. Extract Explicit Lab Values with Lab Units (mg/dL, mEq/L, k/uL, ng/mL, etc.)
        for m in self._lab_unit_regex.finditer(text):
            if is_overlapping(m.start(), m.end()):
                continue
            num_str = m.group(1).replace(',', '')
            try:
                num_val = float(num_str)
            except ValueError:
                continue
            unit = m.group(2).lower()
            context_entity = self._find_nearest_lab(text, m.start(), m.end())
            norm_val, norm_unit = self._normalize_lab(num_val, unit)

            tokens.append(ClinicalToken(
                raw_text=m.group(0).strip(),
                token_type=ClinicalTokenType.LAB_VALUE,
                numeric_val=num_val,
                unit=unit,
                normalized_val=norm_val,
                normalized_unit=norm_unit,
                context_entity=context_entity
            ))
            covered_spans.add((m.start(), m.end()))

        # 3. Extract Percentages (Check if bound to HbA1c lab or general)
        for m in self._percentage_regex.finditer(text):
            if is_overlapping(m.start(), m.end()):
                continue
            num_val = float(m.group(1))
            nearest_lab = self._find_nearest_lab(text, m.start(), m.end())
            if nearest_lab and nearest_lab == "HbA1c":
                tokens.append(ClinicalToken(
                    raw_text=m.group(0).strip(),
                    token_type=ClinicalTokenType.LAB_VALUE,
                    numeric_val=num_val,
                    unit="%",
                    normalized_val=num_val,
                    normalized_unit="%",
                    context_entity="HbA1c"
                ))
            else:
                tokens.append(ClinicalToken(
                    raw_text=m.group(0).strip(),
                    token_type=ClinicalTokenType.PERCENTAGE,
                    numeric_val=num_val,
                    unit="%",
                    normalized_val=num_val,
                    normalized_unit="%",
                    context_entity=self._find_nearest_entity(text, m.start(), m.end())
                ))
            covered_spans.add((m.start(), m.end()))

        # 4. Extract Dosages and Medications (excluding units followed by /)
        for m in self._dosage_regex.finditer(text):
            if is_overlapping(m.start(), m.end()):
                continue
            num_val = float(m.group(1))
            unit = m.group(2).lower()
            context_entity = self._find_nearest_drug(text, m.start(), m.end())

            norm_val = num_val
            norm_unit = unit
            if unit in self.MASS_NORMALIZATION:
                norm_val = num_val * self.MASS_NORMALIZATION[unit]
                norm_unit = "mg"
            elif unit in self.VOLUME_NORMALIZATION:
                norm_val = num_val * self.VOLUME_NORMALIZATION[unit]
                norm_unit = "ml"

            tokens.append(ClinicalToken(
                raw_text=m.group(0).strip(),
                token_type=ClinicalTokenType.DOSAGE,
                numeric_val=num_val,
                unit=unit,
                normalized_val=norm_val,
                normalized_unit=norm_unit,
                context_entity=context_entity
            ))
            covered_spans.add((m.start(), m.end()))

        # 5. Extract Frequencies
        for m in self._freq_regex.finditer(text):
            if is_overlapping(m.start(), m.end()):
                continue
            freq_raw = m.group(0).strip().lower()
            context_entity = self._find_nearest_drug(text, m.start(), m.end())
            tokens.append(ClinicalToken(
                raw_text=freq_raw,
                token_type=ClinicalTokenType.FREQUENCY,
                numeric_val=None,
                unit="frequency",
                normalized_val=None,
                normalized_unit="frequency",
                context_entity=context_entity
            ))
            covered_spans.add((m.start(), m.end()))

        # 6. Extract Heart Rate
        for m in self._hr_regex.finditer(text):
            if is_overlapping(m.start(), m.end()):
                continue
            hr_val = float(m.group(1))
            if 30 <= hr_val <= 240:
                tokens.append(ClinicalToken(
                    raw_text=m.group(0).strip(),
                    token_type=ClinicalTokenType.GENERAL_NUMERIC,
                    numeric_val=hr_val,
                    unit="bpm",
                    normalized_val=hr_val,
                    normalized_unit="bpm",
                    context_entity="heart rate"
                ))
                covered_spans.add((m.start(), m.end()))

        # 7. Extract Technical Metrics (latency, throughput, vcpu)
        for m in self._technical_regex.finditer(text):
            if is_overlapping(m.start(), m.end()):
                continue
            num_val = float(m.group(1))
            unit = m.group(2).lower()
            tokens.append(ClinicalToken(
                raw_text=m.group(0).strip(),
                token_type=ClinicalTokenType.GENERAL_NUMERIC,
                numeric_val=num_val,
                unit=unit,
                normalized_val=num_val,
                normalized_unit=unit,
                context_entity=self._find_nearest_entity(text, m.start(), m.end())
            ))
            covered_spans.add((m.start(), m.end()))

        return tokens

    def _normalize_lab(self, val: float, unit: Optional[str]) -> Tuple[float, Optional[str]]:
        if not unit:
            return val, None
        unit_clean = unit.lower().replace(' ', '')
        if unit_clean in ("k/ul", "k/µl"):
            return val * 1000.0, "/uL"
        if unit_clean in ("/ul", "/µl", "mm3"):
            return val, "/uL"
        return val, unit

    def _find_nearest_drug(self, text: str, start: int, end: int) -> Optional[str]:
        text_lower = text.lower()
        closest_drug: Optional[str] = None
        min_dist = float("inf")

        for drug in self.KNOWN_DRUGS:
            for match in re.finditer(r'\b' + re.escape(drug) + r'\b', text_lower):
                d_start, d_end = match.start(), match.end()
                dist = abs(start - d_end) if start >= d_end else abs(d_start - end)
                if dist < min_dist and dist <= 75:
                    min_dist = dist
                    closest_drug = drug.title()

        return closest_drug

    def _find_nearest_lab(self, text: str, start: int, end: int) -> Optional[str]:
        text_lower = text.lower()
        closest_lab: Optional[str] = None
        min_dist = float("inf")

        for lab_key, lab_name in self.KNOWN_LABS.items():
            for match in re.finditer(r'\b' + re.escape(lab_key) + r'\b', text_lower):
                l_start, l_end = match.start(), match.end()
                dist = abs(start - l_end) if start >= l_end else abs(l_start - end)
                if dist < min_dist and dist <= 75:
                    min_dist = dist
                    closest_lab = lab_name

        return closest_lab

    def _find_nearest_entity(self, text: str, start: int, end: int) -> Optional[str]:
        drug = self._find_nearest_drug(text, start, end)
        if drug:
            return drug
        lab = self._find_nearest_lab(text, start, end)
        if lab:
            return lab
        return None


class NegationDetector:
    """Clinical negation and assertion detector based on clinical NLP heuristics."""

    NEGATION_TRIGGERS = [
        r'\bno\b', r'\bnot\b', r'\bdenies\b', r'\bdenied\b', r'\bnegative for\b',
        r'\bwithout\b', r'\bfree of\b', r'\brule out\b', r'\bruled out\b',
        r'\bunremarkable for\b', r'\babsent\b', r'\bnon-\b', r'\bnever\b',
        r'\bdiscontinued\b', r'\bstopped\b', r'\bnone\b', r'\bwith no\b', r'\bfree from\b'
    ]

    AFFIRMATION_TRIGGERS = [
        r'\bpositive for\b', r'\bconfirmed\b', r'\breports\b', r'\breported\b',
        r'\bpresents with\b', r'\bdiagnosed with\b', r'\bshows\b', r'\bshowed\b',
        r'\brevealed\b', r'\bconsistent with\b', r'\bevidence of\b',
        r'\bstarted on\b', r'\bcontinued on\b', r'\bhistory of\b',
        r'\bacute\b', r'\bactive\b', r'\bpresent\b'
    ]

    def __init__(self) -> None:
        self._neg_pattern = re.compile('|'.join(self.NEGATION_TRIGGERS), re.IGNORECASE)
        self._aff_pattern = re.compile('|'.join(self.AFFIRMATION_TRIGGERS), re.IGNORECASE)

    def analyze_negation(self, text: str) -> Tuple[bool, List[str]]:
        """Determine if text contains clinical negation and return active cues."""
        neg_matches = [m.group(0).lower().strip() for m in self._neg_pattern.finditer(text)]
        is_negated = len(neg_matches) > 0
        return is_negated, neg_matches


class ClaimDecomposer:
    """Decomposes complex clinical and technical summaries into verifiable atomic propositions."""

    def __init__(self) -> None:
        self.splitter = SentenceSplitter()
        self.extractor = ClinicalUnitExtractor()
        self.negation = NegationDetector()

    def decompose(self, text: str) -> List[AtomicClaim]:
        """Split text into sentences and unpack compound clauses into atomic claims."""
        sentences = self.splitter.split(text)
        atomic_claims: List[AtomicClaim] = []
        claim_idx = 0

        for sentence in sentences:
            clean_s = sentence.strip()
            clean_s = re.sub(r'^(?:[A-Za-z\s]+:)\s*', '', clean_s).strip()
            if not clean_s:
                continue

            sub_clauses = self._split_compound_sentence(clean_s)

            for clause in sub_clauses:
                clause_text = clause.strip()
                if len(clause_text) < 4:
                    continue

                quantities = self.extractor.extract_tokens(clause_text)
                is_neg, cues = self.negation.analyze_negation(clause_text)
                entities = self._extract_entities(clause_text)

                claim = AtomicClaim(
                    claim_id=f"claim_{claim_idx}",
                    claim_text=clause_text,
                    extracted_entities=entities,
                    extracted_quantities=quantities,
                    is_negated=is_neg,
                    negation_cues=cues
                )
                atomic_claims.append(claim)
                claim_idx += 1

        return atomic_claims

    def _split_compound_sentence(self, sentence: str) -> List[str]:
        if ';' in sentence:
            parts = [p.strip() for p in sentence.split(';') if p.strip()]
            res = []
            for p in parts:
                res.extend(self._split_compound_sentence(p))
            return res

        if ' and ' in sentence.lower():
            tokens = self.extractor.extract_tokens(sentence)
            dosages = [t for t in tokens if t.token_type in (ClinicalTokenType.DOSAGE, ClinicalTokenType.LAB_VALUE)]
            if len(dosages) >= 2:
                parts = re.split(r'\band\b', sentence, flags=re.IGNORECASE)
                if len(parts) == 2 and len(parts[0].strip()) > 15 and len(parts[1].strip()) > 15:
                    p0 = parts[0].strip()
                    p1 = parts[1].strip()
                    if not any(drug in p1.lower() for drug in ClinicalUnitExtractor.KNOWN_DRUGS) and \
                       not any(lab in p1.lower() for lab in ClinicalUnitExtractor.KNOWN_LABS):
                        return [sentence]
                    return [p0, p1]

        return [sentence]

    def _extract_entities(self, text: str) -> List[str]:
        entities: List[str] = []
        text_lower = text.lower()

        for drug in ClinicalUnitExtractor.KNOWN_DRUGS:
            if re.search(r'\b' + re.escape(drug) + r'\b', text_lower):
                entities.append(drug.title())

        for lab_key, lab_name in ClinicalUnitExtractor.KNOWN_LABS.items():
            if re.search(r'\b' + re.escape(lab_key) + r'\b', text_lower):
                if lab_name not in entities:
                    entities.append(lab_name)

        conditions = [
            "pneumonia", "chest pain", "shortness of breath", "myocardial infarction",
            "stroke", "hypertension", "diabetes", "fever", "edema", "sepsis",
            "pulmonary embolism", "deep vein thrombosis", "dvt", "acute kidney injury",
            "heart failure", "atrial fibrillation", "infection", "headache", "asthma",
            "hyperkalemia", "hypokalemia", "tachycardia", "bradycardia"
        ]
        for cond in conditions:
            if re.search(r'\b' + re.escape(cond) + r'\b', text_lower):
                entities.append(cond.title())

        return list(dict.fromkeys(entities))

from __future__ import annotations

REASON_TYPES = [
    "definition",
    "attribute",
    "comparison",
    "temporal",
    "numeric",
    "derivation",
    "multi-hop",
]


def qa_generation_prompt(passage: str, *, num_questions: int = 2) -> str:
    single_types = [t for t in REASON_TYPES if t != "multi-hop"]
    return (
        "Given the passage below, write "
        f"{num_questions} questions that can be answered only using the passage. "
        "For each question, output a JSON object with:\n"
        '  - "question"\n'
        '  - "answer" (short, unambiguous)\n'
        '  - "evidence_span" (minimal supporting sentences copied from the passage)\n'
        f'  - "reason_type" (one of: {", ".join(single_types)})\n'
        "Avoid yes/no questions. Avoid copying long phrases from the evidence_span into the question.\n"
        "Prefer questions that require combining at least two pieces of information from the passage "
        "(e.g., compare/derive/compute/identify a relationship), not a verbatim definition lookup.\n\n"
        f"Passage:\n{passage}"
    )


def qa_generation_multihop_prompt(passage_a: str, passage_b: str, *, num_questions: int = 2) -> str:
    return (
        "You are given TWO passages (A and B). Write "
        f"{num_questions} questions that require combining information from BOTH passages to answer.\n"
        "For each question, output a JSON object with:\n"
        '  - "question"\n'
        '  - "answer" (short, unambiguous)\n'
        '  - "evidence_span_a" (minimal supporting sentences copied from Passage A)\n'
        '  - "evidence_span_b" (minimal supporting sentences copied from Passage B)\n'
        f'  - "reason_type" (use "multi-hop")\n'
        "Avoid yes/no questions. Avoid copying long phrases from the evidence spans into the question.\n"
        "The question MUST NOT be answerable from only Passage A or only Passage B.\n"
        "Make the question naturally coherent by using a shared entity/term/concept that appears in both passages.\n\n"
        f"Passage A:\n{passage_a}\n\nPassage B:\n{passage_b}"
    )


def paraphrase_prompt(question: str, *, k: int = 2) -> str:
    return (
        "Rewrite the question in "
        f"{k} meaning-preserving ways. Keep all constraints (numbers, quantifiers, conditions) unchanged. "
        "Do not add new information.\n"
        "Return a JSON list of paraphrase strings.\n\n"
        f"Question:\n{question}"
    )


def verifier_prompt(question: str, answer: str, evidence: str) -> str:
    return (
        "Decide whether the proposed answer is fully supported by the evidence span.\n"
        'Return exactly one of: "ENTAILED", "CONTRADICTED", "NOT_ENOUGH_INFO".\n\n'
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Evidence span: {evidence}"
    )


def equiv_prompt(a: str, b: str) -> str:
    return (
        "You are a strict judge. Determine whether Answer A and Answer B are semantically equivalent.\n"
        "Return JSON only: {\"equivalent\": true/false}.\n\n"
        f"Answer A: {a}\n"
        f"Answer B: {b}"
    )


def near_miss_prompt(question: str) -> str:
    return (
        "Modify the canonical question by changing exactly one minimal constraint "
        "(e.g., flip a quantifier, tweak a single numeric threshold, remove one condition) "
        "such that the modified question becomes unanswerable from the same evidence or contradicted by it.\n"
        "Return JSON only: {\"modified_question\": \"...\", \"label\": \"UNANSWERABLE\"|\"CONTRADICTED\"}.\n\n"
        f"Canonical question:\n{question}"
    )

"""The agreed repeated measurements are a required deliverable, not an assertion."""
from profiling.verify_factorization import verify


def test_factorization_evidence_matches_current_engine_and_inputs():
    verify()

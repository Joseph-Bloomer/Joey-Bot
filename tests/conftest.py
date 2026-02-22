"""
Shared pytest fixtures and configuration for all Joey-Bot tests.

This file is auto-loaded by pytest before running any test. Add fixtures here
when they are needed by more than one test module (e.g. a Flask app context,
a fake vector store, or mock LLM stubs). For now it is intentionally empty
so each test module can be understood on its own.

Usage:
    Run all tests from the project root:
        python -m pytest tests/ -v

    Run a single test file:
        python -m pytest tests/unit/test_reranker.py -v

    The `python -m` invocation ensures the project root is on sys.path so
    imports like `from app.services.reranker import HeuristicReranker` resolve
    correctly without any extra path manipulation.
"""

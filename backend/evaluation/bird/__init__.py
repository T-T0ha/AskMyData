"""BIRD benchmark harness — the product's NL2SQL pipeline against a raw-LLM baseline.

Deliberately separate from ``app``: nothing here is imported by the product, and
nothing in ``app`` is modified to support it.  The semantic arm *is* the product
(it calls ``app.api.services`` — the functions the API routes call — on a session
built from each BIRD database); the raw arm is the same model, prompt and client
shown only table and column names.  See ``README.md``.
"""

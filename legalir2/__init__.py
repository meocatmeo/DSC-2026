"""LegalIR v2: article aware retrieval for the UIT DSC 2026 Task 1 corpus.

Stages: structure aware chunking, BM25F plus a hashed bigram phrase channel, a
locally run bi-encoder, rank fusion, and a cross encoder reranker fine tuned on the
organiser's own training labels. Only checkpoints from the approved model sheet are
used, all inference is local, and the deployed system is audited to stay strictly
under the four billion parameter budget.
"""
__version__ = "2.0.0"

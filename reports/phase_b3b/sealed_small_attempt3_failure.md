# Phase B3B sealed Small Attempt 3 failure

Attempt 3 stopped after the sealed Small and all frozen artifacts passed their
identity checks. It was not retried.

- Modal run:
  `https://modal.com/apps/lzf1416082617/main/ap-0J7JeMn5g1A1ejLNI16EYx`
- Main merge commit:
  `4e27387a5925bd2f3839c1df4779a0d041785682`
- Runner commit:
  `90eced9e062004b5954fab257989b96f2a43339c`
- Wrapper pin commit:
  `6c12e7fb400bb33a05c11648be000a46690d8e79`
- Failure stage: `formal_report_serialization_audit_counts`
- Exception: `IndexError`
- Location: `scripts/run_phase_b3b_sealed_small.py:295`

The Small CSV was read and queries were constructed. Global Popularity, BPR,
Two-Tower, Hybrid, and the formal metrics were computed in remote memory.
The process then failed while serializing the audit-count section because
`observed_normal` was a NumPy array but was indexed as though it were a pandas
DataFrame.

No metric value, ranking, JSON report, or Markdown report was returned, written,
or observed. There was no automatic retry, post-failure Small/model inspection,
code change, retraining, tuning, or temporal-final access. Further action
requires independent review.

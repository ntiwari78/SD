# Risk register

| ID | Risk | Impact | Likelihood | Response |
|---|---|---|---|---|
| RK-01 | Assessment item banks arrive late or on unusable terms | R2 cannot start; UC-06 becomes an authoring system (A-06) | Medium | Start the partner conversation before R1 code; agree a JSON package contract early so the import path can be built against a schema, not a vendor |
| RK-02 | A guard failure exposes a student to harmful content | Child safety; programme credibility; regulatory exposure | Low, severe | Guard on input and output of every student-facing turn; RAG-only, no open web; templated safe reply on trip; escalation to coordinator and mentor within one minute; opt-in schools only in R3 |
| RK-03 | Consent records prove legally insufficient under DPDP | Processing must halt; enrolled cohort at risk | Low, severe | Legal review of the consent notice and `consent_record` fields before R1 pilot; notice versioned and shown in the guardian's language; WORM storage of forms |
| RK-04 | Indic LLM provider cannot contractually guarantee residency or no-training | R3 slips or falls back to Azure OpenAI Central India | Medium | Provider abstraction in the AI Gateway from day one; evaluate two providers against the golden set; Azure OpenAI as the named fallback |
| RK-05 | Cross-tenant data leak between NGOs | Partner trust; DPDP breach | Low, severe | Enforcement twice — FastAPI capability check and PostgreSQL RLS; RLS tested as a first-class test suite, not an assumption |
| RK-06 | PWA offline behaviour fails in real school conditions | Coordinators lose form data; adoption collapses | Medium | Field-test the enrollment form on a real school connection during R1, before the 500-student target |
| RK-07 | Small volunteer-augmented team loses continuity | Knowledge loss; velocity collapse | Medium | ADRs and this workspace are the mitigation; module boundaries enforced in CI so a new contributor cannot accidentally couple things |
| RK-08 | AI cost overruns at scale | Programme budget | Medium | Daily cap per workload, soft alert at 80%, hard stop to templated replies; cost logged per call and reviewed weekly |
| RK-09 | Reviewers defer to DSS summaries, creating de facto automation | The human-in-the-loop guarantee becomes nominal (ADR-0005) | Medium | Blocked recommendation verbs in the output guard; measure reviewer feedback rather than agreement; periodic audit of decisions against summaries |

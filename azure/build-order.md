# Build order

Four releases. Each release exists to prove something, not to ship a feature count —
the exit criterion is the proof.

| Release | Theme | Use cases | Proves |
|---|---|---|---|
| R1 | Foundation | UC-03, UC-01, UC-04, UC-05, UC-08 + IAM, DOC, NTF, AUD | An NGO can run the programme end to end for real students, with legally valid consent |
| R2 | Assess and Award | UC-06, UC-02 | A cohort can be assessed and a scholarship decided, with Finance closing the loop |
| R3 | Intelligence | UC-07, UC-09, UC-10 | AI helps students and reviewers without touching a decision, and is safe enough to expose to minors |
| R4 | Continuity | UC-11 + analytics hardening | A student survives the transition out of the programme, with consent re-affirmed in their own name |

## Exit criteria

**R1** — 1 NGO, 10 schools, 500 students enrolled with valid consent records; mentors logging
sessions; the volunteer-hours export delivered to and accepted by the CSR/HR team.

**R2** — First cohort assessed with versioned instruments from the psychometrics partner; the
scholarship pipeline runs end to end including a Finance disbursement status upload.

**R3** — Golden evaluation set passed in all three R1 languages; guard classifier live;
reviewers using DSS summaries; the assistant limited to opt-in schools.

**R4** — First alumni cohort transitioned; the consent re-affirmation flow proven on students
who have turned 18.

## Critical-path dependencies

| Dependency | Needed by | Owner | Status |
|---|---|---|---|
| Psychometrics item banks (signed JSON, scoring keys, norms) | R2 start | CSR Program Office | Open — OI-03, D-06 |
| Indic LLM provider contract, residency and no-training clauses | R3 start | Architecture + Legal | Open — OI-02 |
| DLT registration of SMS templates and sender ID | R1 start | Program Office + provider | Open — OI-04 |
| Azure landing zone and subscription model | R1 start | Brillio IT | Open — OI-07 |
| Safeguarding policy for mentor–minor interaction | R1, UC-08 rules | CSR + NGO partners | Open — OI-05 |
| Golden evaluation set, three languages | R3 start | Data science | Open — OI-08 |

Four of the six block a release start. They are procurement and policy items with long lead
times, not engineering work — they should be moving in parallel with R1 construction, and R1
should not be treated as slack.

## Sequencing note

R1 builds tenancy (UC-03) before enrollment (UC-01) because every domain row carries a
`tenant_id` and RLS is enforced from the first migration. Retrofitting isolation is not an
option once real student data exists.

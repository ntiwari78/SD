# Assumptions register

Twelve assumptions from HLD §4, unanswered at the close of the clarification round.
Each carries the consequence of being wrong. Confirming one is a dated entry here;
overturning one is an ADR.

| ID | Assumption | If wrong, what changes | Owner | Status |
|---|---|---|---|---|
| A-01 | All 11 use cases are designed to LLD depth; release sequencing follows §15 | Depth of Part B for later releases | CSR Program Office | Open |
| A-02 | Brillio staff use Entra ID (workforce) SSO; NGO staff, school coordinators, students and guardians use Entra External ID (phone/email OTP); students sign in on a guardian's or school-issued mobile number | Identity design, HLD §8 and §B.1 | Brillio IT | Open |
| A-03 | Consent is collected on the ground by the NGO field coordinator, recorded digitally, with guardian OTP where a phone exists and a scanned signed form otherwise | UC-01 consent sub-flow | CSR + NGO partners | Open |
| A-04 | Pilot 10,000 students / 20 NGOs / 300 schools / 500 mentors; steady state 100,000; peak ~2,000 concurrent during assessment windows | Sizing §14; pgvector vs. Azure AI Search | CSR Program Office | Open |
| A-05 | Student languages en, hi, kn in R1; ta, te, mr later. Mentor/NGO/admin surfaces English-only | i18n scope; Indic model evaluation set | CSR Program Office | Open |
| A-06 | Assessment instruments are licensed from a psychometrics partner as structured JSON item banks; the platform scores and stores, it does not author | UC-06 becomes a content-authoring system too | CSR Program Office | Open — blocks R2 |
| A-07 | Volunteer hours export to Brillio CSR/HR as a monthly file; no live integration in R1 | Integration adapter in UC-08 | Brillio CSR/HR | Open |
| A-08 | Schools are mapped against the UDISE+ school master | School entity in UC-03 | CSR Program Office | Open |
| A-09 | The student record persists Grade 9 → Grade 12 → higher education; "alumni" begins at cohort exit; consent re-affirmed at 18 | Retention policy; UC-11 data model | CSR + Legal | Open |
| A-10 | Back end is Python (FastAPI); .NET 8 is an equally valid substitute for the core API | Library choices in the LLD only | Brillio IT | Open — ADR-0010 provisional |
| A-11 | Learning and certification integrations are outbound tracked links in R1; no enrolment or completion callbacks | UC-07 scope | CSR Program Office | Open |
| A-12 | Finance updates disbursement status via admin portal or CSV upload; no ERP integration | UC-02 disbursement sub-flow | Finance | Open — see OI-06 |

## How to close one

Add the confirmation date and the person who confirmed it, set Status to `Confirmed`, and
apply the consequence. If the answer differs from the assumption, raise an ADR instead —
the register records what was assumed, the ADR records what was decided.

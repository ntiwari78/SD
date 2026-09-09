# Open items

From HLD Appendix 4, plus items raised during workspace setup. Each needs a named owner and
a date, not just an owning team.

| ID | Item | Owner | Blocks | Status |
|---|---|---|---|---|
| OI-01 | Confirm assumptions A-01 to A-12 | CSR Program Office | Design sign-off | Open |
| OI-02 | Choose the Indic LLM provider; confirm contractual data residency and no-training clauses | Architecture + Legal | R3 | Open |
| OI-03 | Psychometrics partner contract — instrument packages, norms, licence per student | CSR Program Office | R2 | Open |
| OI-04 | DLT registration of SMS templates and sender ID | Program Office + provider | R1 | Open |
| OI-05 | Safeguarding policy for mentor–minor interactions, codified into UC-08 rules | CSR + NGO partners | R1 | Open |
| OI-06 | Whether Finance prefers CSV upload or portal entry as the primary disbursement channel | Finance | R2 | Open |
| OI-07 | Azure landing zone ownership and subscription model (pilot vs. scale) | Brillio IT | R1 | Open |
| OI-08 | Golden evaluation set collection plan — real anonymised student queries, three languages | Data science | R3 | Open |
| OI-09 | Restore the three missing documents: Requirement and Build Order v1.0, Model Strategy and Cost v1.0, Architecture Drawing Set v1.0 | Niraj | Design review | Open |
| OI-10 | Architecture review board sign-off on HLD/LLD v1.0 | Architecture | R1 start | Open |

## Notes

**OI-05 deserves early attention.** It is the only item on this list where being late has a
child-safety consequence rather than a schedule one. Mentor–minor interaction rules — session
visibility, one-to-one contact, escalation duties — shape UC-08 and UC-09 designs, and both are
easier to build right than to retrofit.

**OI-09.** The HLD/LLD cites specific findings and section numbers from all three missing
documents (Model Strategy §2 and §5; requirement-document Findings 1–4). Restoring the
originals is preferable to reconstruction, which would produce plausible text that the HLD
citations then point at incorrectly.

## Added during Azure cost modelling (September 2026)

| ID | Item | Owner | Blocks | Status |
|---|---|---|---|---|
| OI-11 | HLD §9 specifies a dedicated `NC8-T4` GPU workload profile at ₹589.58/hour (~₹2.83 lakh/month in production alone). Serverless GPU (consumption NC T4 v3) at ₹34.92/hour does the same work. Raise an ADR to change the profile, and correct the §14 cost line, which is not achievable as specified. | Architecture | Budget sign-off | Open |
| OI-12 | Confirm the PostgreSQL Flexible Server D4ds_v5 rate for Central India in the Azure Pricing Calculator. It is the largest single line in the estimate and the one rate that could not be read from the retail price API. | Architecture | Budget sign-off | Open |
| OI-13 | Decide the production APIM tier. Developer (HLD default) carries no SLA; Basic v2 does, at roughly three times the cost; Standard is ~₹66,000/month. Confirm Basic v2 meets the VNet-integration requirement. | Architecture + Brillio IT | R1 | Open |
| OI-14 | Set a Log Analytics sampling policy before R1. Ingestion bills at ₹307.66/GB and unsampled OpenTelemetry from every service will exceed the modelled 30 GB/month. | Architecture | R1 | Open |
| OI-15 | Confirm the Entra External ID monthly-active-user allowance and the per-MAU rate beyond it. Immaterial at 10k students, a headline cost at 100k. | Brillio IT | R2 | Open |

Reference: `Bringing_Smiles_Azure_Topology_v1.0.html` in this folder — annotated deployment diagrams
for staging and production with per-node pricing configuration.

# ADR-028: Security, Privacy & Compliance Architecture for RuVector DNA Analyzer

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: RuVector Security Architecture Team
**Deciders**: Architecture Review Board, Security Team, Privacy Officer
**Technical Area**: Genomic Data Security, Differential Privacy, Homomorphic Encryption, Zero-Knowledge Proofs, Compliance
**Parent ADRs**: ADR-001 (Core Architecture), ADR-007 (Security Review), ADR-012 (Security Remediation), ADR-CE-008 (Multi-Tenant Isolation), ADR-CE-017 (Unified Audit Trail), ADR-DB-010 (Delta Security Model)

---

## Context and Problem Statement

The RuVector DNA Analyzer operates on genomic data -- the most sensitive category of personal information in existence. Unlike passwords, credit card numbers, or social security numbers, genomic data is **immutable**: a compromised genome cannot be rotated, revoked, or regenerated. A breach of genomic data constitutes permanent, irrevocable exposure of an individual's most intimate biological identity.

Furthermore, genomic data is **inherently shared**. An individual's genome encodes information about biological relatives: approximately 50% shared with parents and children, 25% with grandparents and grandchildren, 12.5% with first cousins. A single breach therefore cascades across an entire family tree, including individuals who never consented to analysis.

Published research has demonstrated that as few as **75 single-nucleotide polymorphisms (SNPs)** suffice to re-identify an individual from an ostensibly anonymized dataset (Gymrek et al., 2013; Erlich & Narayanan, 2014). Traditional de-identification techniques are therefore insufficient. The system must employ cryptographic and information-theoretic guarantees rather than relying on statistical anonymization alone.

### Regulatory Landscape

Genomic data falls under overlapping and sometimes conflicting regulatory frameworks:

| Regulation | Jurisdiction | Key Requirement | Genomic Specificity |
|------------|-------------|-----------------|---------------------|
| HIPAA | United States | PHI safeguards | Genomic data is PHI when linked to a covered entity |
| GINA | United States | Non-discrimination | Prohibits use of genetic information in health insurance and employment |
| GDPR Article 9 | European Union | Special category data | Explicit consent required; right to erasure applies |
| FDA 21 CFR Part 11 | United States | Electronic records | Applies when genomic analysis supports clinical decisions |
| ISO 27001 / 27701 | International | Information security / Privacy | Framework for ISMS and PIMS |
| California CCPA/CPRA | California | Consumer privacy rights | Genetic data classified as sensitive personal information |

### Threat Actor Taxonomy

| Actor | Motivation | Capability | Primary Targets |
|-------|-----------|------------|-----------------|
| Nation-state | Population-level intelligence, bioweapons | Advanced persistent threat, supply-chain compromise | Entire genomic databases, ancestry correlations |
| Insurance actuaries | Risk discrimination | Legal or semi-legal data acquisition, linkage attacks | Disease predisposition variants, pharmacogenomic markers |
| Law enforcement | Forensic identification | Familial DNA searching, compelled disclosure | STR profiles, Y-chromosome haplotypes, mitochondrial sequences |
| Employers | Workforce risk assessment | GINA violations through third-party data brokers | Late-onset disease genes (BRCA, Huntington's HTT, APOE e4) |
| Criminal extortion | Blackmail | Breach-and-threaten | Ancestry secrets, disease predispositions, paternity |
| Academic competitors | Priority, intellectual credit | Reconstruction attacks on published summary statistics | Rare variant frequencies, novel associations |

---

## 1. Threat Model for Genomic Data

### 1.1 Fundamental Properties of Genomic Threats

**Immutability.** Genomic data cannot be changed after compromise. The threat horizon is the lifetime of the individual plus the lifetimes of all descendants for whom the data remains predictive. Conservatively, a single breach has a **multi-generational impact window of 100+ years**.

**Re-identification surface.** Homer et al. (2008) demonstrated that the presence of an individual in a genome-wide association study (GWAS) cohort can be inferred from aggregate allele frequency statistics. Subsequent work has reduced the threshold to 75 independent SNPs for high-confidence re-identification. The RuVector HNSW index, by design, enables rapid nearest-neighbor retrieval -- the same property that makes it useful for genomic similarity search also makes it a powerful re-identification engine if access controls fail.

**Familial transitivity.** Compromising one genome partially compromises all biological relatives. The system must treat familial linkage as a first-class security concern, not an afterthought.

### 1.2 Attack Vectors Specific to RuVector

**Side-channel timing attacks on HNSW search.** The hierarchical navigable small world graph traverses different numbers of nodes depending on the query vector's proximity to existing data points. An attacker who can measure query latency with sufficient precision (microsecond-level, achievable via network timing) can infer whether a query genome is "near" entries in the database, enabling membership inference. Mitigation requires constant-time traversal padding or oblivious RAM (ORAM) techniques at the index layer.

**Model inversion on embeddings.** If the RuVector embedding model maps genomic sequences to dense vectors, an adversary with access to the embedding space can attempt to reconstruct the original sequence. For genomic data, even partial reconstruction (recovering the values of clinically significant SNPs) constitutes a severe breach. The relationship between the embedding function `E: Genome -> R^d` and the underlying genotype must be analyzed under the lens of membership inference and attribute inference attacks.

**Delta replay and injection.** Per ADR-DB-010, delta-based updates introduce the risk of replaying old deltas or injecting crafted deltas. In the genomic context, a replayed delta could revert a patient's corrected variant annotation, while an injected delta could alter clinical-grade variant calls.

**Embedding linkage across datasets.** If the same embedding model is used across multiple institutions, an adversary with access to embeddings from two datasets can perform linkage attacks, matching individuals across ostensibly separate cohorts. This is the vector-space analog of a record linkage attack.

### 1.3 Attack Surface Diagram

```
                        EXTERNAL BOUNDARY
                              |
          +-------------------+-------------------+
          |                   |                   |
    [API Gateway]      [WASM Client]      [MCP Interface]
          |                   |                   |
          +-------------------+-------------------+
                              |
                    [Authentication Layer]
                    [Claims Evaluator (ADR-010)]
                              |
          +-------------------+-------------------+
          |                   |                   |
    [Query Engine]     [Embedding Engine]  [Variant Caller]
    - HNSW traversal   - Model inference   - VCF processing
    - Timing leaks     - Inversion risk    - Path traversal
          |                   |                   |
          +-------------------+-------------------+
                              |
                    [Storage Layer]
                    - Encrypted vectors
                    - Delta chain integrity
                    - Key management
                              |
                    [Audit Trail (ADR-CE-017)]
                    - Hash-chained witnesses
                    - Tamper-evident log
```

---

## 2. Differential Privacy for Genomic Queries

### 2.1 Privacy Model

All population-level frequency queries (allele frequency, genotype frequency, haplotype frequency) must satisfy **(epsilon, delta)-differential privacy**. The system guarantees that the output of any query changes by at most a bounded amount whether or not any single individual's genome is included in the dataset.

**Definition.** A randomized mechanism M satisfies (epsilon, delta)-differential privacy if for all datasets D1 and D2 differing in one individual's record, and for all sets S of possible outputs:

```
Pr[M(D1) in S] <= exp(epsilon) * Pr[M(D2) in S] + delta
```

### 2.2 Noise Calibration for Genomic Queries

The existing `DifferentialPrivacy` implementation in `crates/ruvector-dag/src/qudag/crypto/differential_privacy.rs` provides Laplace and Gaussian mechanisms. For the DNA Analyzer, we extend this with genomic-specific calibration.

**Allele frequency queries.** For a biallelic SNP in a cohort of N individuals (2N chromosomes), the sensitivity of the allele frequency estimator is `Delta_f = 1/(2N)` (adding or removing one individual changes the count by at most 1 out of 2N alleles). The Laplace mechanism adds noise with scale `b = Delta_f / epsilon = 1 / (2N * epsilon)`.

```rust
/// Genomic-specific differential privacy for allele frequency queries.
pub struct GenomicDpConfig {
    /// Privacy budget per query
    pub epsilon: f64,
    /// Failure probability
    pub delta: f64,
    /// Cohort size (number of individuals)
    pub cohort_size: usize,
    /// Ploidy (2 for diploid organisms)
    pub ploidy: usize,
}

impl GenomicDpConfig {
    /// Sensitivity of allele frequency for a single biallelic locus.
    /// Adding/removing one individual changes allele count by at most `ploidy`
    /// out of `cohort_size * ploidy` total alleles.
    pub fn allele_frequency_sensitivity(&self) -> f64 {
        self.ploidy as f64 / (self.cohort_size * self.ploidy) as f64
    }

    /// Laplace noise scale for allele frequency queries.
    pub fn laplace_scale(&self) -> f64 {
        self.allele_frequency_sensitivity() / self.epsilon
    }
}
```

**Multi-SNP queries.** For queries spanning k correlated SNPs, we apply the composition theorem. Under basic composition, the total privacy loss is `k * epsilon`. Under advanced composition (Dwork, Roth, & Vadhan, 2010), for k queries each satisfying (epsilon, delta)-DP:

```
Total epsilon' = sqrt(2k * ln(1/delta')) * epsilon + k * epsilon * (exp(epsilon) - 1)
```

The existing `advanced_privacy_loss` method in the codebase implements this correctly.

### 2.3 Privacy Budget Accounting

Each dataset and each user maintains a **privacy budget ledger**. Every query consumes a portion of the budget. When the budget is exhausted, further queries are denied.

```rust
pub struct PrivacyBudgetLedger {
    /// Maximum total epsilon allowed
    pub max_epsilon: f64,
    /// Maximum total delta allowed
    pub max_delta: f64,
    /// Running total of epsilon consumed (advanced composition)
    pub consumed_epsilon: f64,
    /// Running total of delta consumed
    pub consumed_delta: f64,
    /// Per-query log for audit
    pub query_log: Vec<PrivacyExpenditure>,
}

pub struct PrivacyExpenditure {
    pub timestamp: u64,
    pub query_hash: [u8; 32],
    pub epsilon_spent: f64,
    pub delta_spent: f64,
    pub mechanism: PrivacyMechanism,
    pub requester_id: String,
}

pub enum PrivacyMechanism {
    Laplace,
    Gaussian,
    ExponentialMechanism,
    SparseVector,
    /// Renyi DP with alpha parameter
    Renyi { alpha: f64 },
}
```

**Budget policy.** Default budget: epsilon_total = 10.0, delta_total = 1e-5 per dataset per calendar year. Clinical queries (pharmacogenomic lookups) draw from a separate, more generous budget since the patient has consented to their own data use.

### 2.4 Secure Multi-Party Computation for Cross-Institutional Studies

For federated GWAS across multiple institutions, no institution should reveal its raw genotype data. We specify a two-phase protocol:

**Phase 1: Secure aggregation of allele counts.** Each institution i holds counts (a_i, n_i) for allele a at locus L in cohort of size n_i. Institutions use additive secret sharing: each institution splits its count into k shares (one per other institution plus one for itself), distributes shares, and all institutions sum their received shares locally. The result is the global sum without any institution learning another's individual count.

**Phase 2: Differentially private release.** The aggregated count is perturbed with calibrated noise before release. The sensitivity is 1 (one institution contributing one additional individual changes the global count by at most 1 allele per locus for diploid).

**Protocol specification.**

```
Protocol: DP-SecureAlleleAggregation
Participants: Institutions I_1, ..., I_k
Input: Each I_j holds (count_j, total_j) for a locus
Output: Noisy global allele frequency

1. Each I_j generates random shares s_j1, ..., s_jk such that
   sum(s_j1..s_jk) = count_j  (modular arithmetic over Z_p, p > N_total)
2. I_j sends s_ji to I_i for all i != j, retains s_jj
3. Each I_i computes local_sum_i = sum over j of s_ji
4. All institutions broadcast local_sum_i
5. Global count = sum(local_sum_i) mod p
6. Trusted noise oracle (or threshold decryption) adds Laplace(1/epsilon) noise
7. Release noisy global frequency = (global_count + noise) / N_total
```

---

## 3. Homomorphic Encryption for Secure Analysis

### 3.1 Scheme Selection

We adopt the **CKKS** (Cheon-Kim-Kim-Song) scheme for operations on encrypted genomic vectors. CKKS supports approximate arithmetic on encrypted real-valued vectors, which maps directly to the vector operations required by the RuVector core engine.

**Rationale for CKKS over BFV/BGV.** Genomic similarity computations (cosine distance, dot product) operate on floating-point vectors. CKKS natively encodes real numbers with bounded precision, avoiding the integer-encoding overhead of BFV/BGV schemes. The approximate nature of CKKS (results carry a small additive error) is acceptable for similarity search where ranking, not exact distances, determines results.

### 3.2 Target Operations and Performance Bounds

| Operation | Plaintext Baseline | Encrypted Target | Max Overhead |
|-----------|-------------------|------------------|-------------|
| Cosine similarity (384-dim) | 0.5us | 5us | 10x |
| HNSW distance comparison | 0.3us | 2.4us | 8x |
| Variant genotype lookup | 0.1us | 1.0us | 10x |
| Batch embedding (1000 vectors) | 50ms | 400ms | 8x |
| Allele frequency aggregation | 1ms | 8ms | 8x |

**Parameter selection.** For 128-bit security with CKKS:

```
Ring dimension (N):     2^15 = 32768
Coefficient modulus:    log(Q) = 438 bits (chain of 14 primes)
Scaling factor:         2^40
Max multiplicative depth: 12
Key-switching method:   Hybrid (decomposition base = 2^60)
```

These parameters provide sufficient depth for a single HNSW layer traversal (involving ~log(N) distance comparisons, each requiring one multiplication and one addition on encrypted data).

### 3.3 Selective Encryption Architecture

Full homomorphic encryption of the entire genome is computationally prohibitive for interactive queries. Instead, we employ **selective encryption** with a three-tier classification:

| Tier | Classification | Encryption | Examples |
|------|---------------|------------|---------|
| **Tier 1: Sensitive** | Clinically actionable, high discrimination risk | Full CKKS encryption | BRCA1/2, APOE, HTT, CFTR, HLA region |
| **Tier 2: Moderate** | Research-relevant, moderate re-identification risk | Encrypted at rest, decrypted in TEE for computation | Common GWAS hits, pharmacogenomic loci (CYP2D6, CYP2C19) |
| **Tier 3: Reference** | Low sensitivity, publicly catalogued variants | Cleartext with integrity protection (HMAC) | Synonymous variants, intergenic SNPs with >5% MAF |

The tier assignment is driven by a policy engine that considers:
- ClinVar clinical significance classification
- Allele frequency (rare variants are more identifying)
- Gene-disease association strength (OMIM, ClinGen)
- Regulatory classification under GINA protected categories

```rust
pub enum EncryptionTier {
    /// Full CKKS homomorphic encryption. All computation on ciphertext.
    Sensitive,
    /// Encrypted at rest. Decrypted only inside TEE for computation.
    Moderate,
    /// Cleartext with HMAC integrity verification.
    Reference,
}

pub struct GenomicRegionPolicy {
    pub chromosome: u8,
    pub start_position: u64,
    pub end_position: u64,
    pub tier: EncryptionTier,
    pub justification: &'static str,
    pub regulatory_references: Vec<RegulatoryReference>,
}

pub enum RegulatoryReference {
    Gina { section: &'static str },
    Hipaa { standard: &'static str },
    Gdpr { article: u32 },
    Fda21Cfr11 { section: &'static str },
}
```

### 3.4 Key Management

Encryption keys follow a hierarchy rooted in a hardware security module (HSM) or, where HSM is unavailable, a software-based key derivation chain using HKDF-SHA256.

```
Master Key (HSM-resident, never exported)
  |
  +-- Dataset Encryption Key (DEK) -- per-dataset, AES-256-GCM for at-rest
  |     |
  |     +-- Region Key -- per-genomic-region, derived via HKDF
  |
  +-- CKKS Public Key (for homomorphic operations)
  |
  +-- CKKS Secret Key (HSM-resident, used only for final decryption)
  |
  +-- CKKS Evaluation Keys (galois keys, relinearization keys -- public)
```

**Key rotation.** DEKs rotate on a 90-day cycle. On rotation, existing ciphertexts are re-encrypted under the new key via a background migration process. The old key is retained in a "decrypt-only" state for 180 days to handle in-flight operations, then destroyed. This mechanism also supports **cryptographic deletion** (see Section 5.5).

---

## 4. Zero-Knowledge Proofs for Genomic Attestation

### 4.1 Motivation

Clinical and social scenarios require proving properties of a genome without revealing the genome itself:

- A patient proves compatibility with a prescribed drug (pharmacogenomics) without revealing their CYP2D6 metabolizer status to the pharmacist's information system.
- A couple proves genetic compatibility (absence of shared recessive disease carrier status) without disclosing individual genotypes to each other or a third party.
- An individual proves membership in a specific ancestry cluster for a research study without revealing full ancestry composition.

### 4.2 Construction: zk-SNARK for Genotype Predicates

We define a general-purpose **genotype predicate circuit** that proves statements of the form:

```
"I possess a genotype at locus L that satisfies predicate P,
 committed under Pedersen commitment C."
```

The circuit operates over the BLS12-381 curve (chosen for its efficient pairing operations and compatibility with the Groth16 proof system).

**Circuit definition (R1CS).**

```
Public inputs:  commitment C, predicate hash H(P), locus identifier L
Private inputs: genotype g (encoded as {0, 1, 2} for {homozygous ref, het, homozygous alt}),
                blinding factor r

Constraints:
1. C == g * G + r * H                    (Pedersen commitment opens correctly)
2. g in {0, 1, 2}                         (valid diploid genotype)
3. P(g) == 1                              (predicate satisfied)
```

**Predicate examples.**

| Use Case | Predicate P(g) | Statement Proven |
|----------|---------------|------------------|
| Pharmacogenomic safety | g != 2 at CYP2D6*4 | "I am not a CYP2D6 poor metabolizer" |
| Carrier screening | g1 + g2 < 4 for both partners | "We do not both carry two copies of the same recessive allele" |
| Ancestry membership | embedding(genome) in cluster C | "My ancestry falls within cluster C" |
| Disease risk threshold | risk_score(genotypes) < T | "My polygenic risk score is below threshold T" |

### 4.3 Proof Parameters and Performance

| Parameter | Value |
|-----------|-------|
| Curve | BLS12-381 |
| Proof system | Groth16 (succinct, constant-size proofs) |
| Proof size | 192 bytes (3 group elements) |
| Verification time | <5ms |
| Proving time (single locus) | <500ms |
| Proving time (polygenic, 100 loci) | <10s |
| Trusted setup | Powers of Tau ceremony + circuit-specific phase 2 |

**Implementation note.** The existing ZK proof infrastructure in the codebase (see `/home/user/ruvector/docs/security/zk_security_audit_report.md`) has been audited and found to contain critical vulnerabilities in its proof-of-concept implementation. For genomic attestation, we mandate the use of production-grade libraries:

- **arkworks-rs** (ark-groth16, ark-bls12-381) for proof generation and verification
- **merlin** for Fiat-Shamir transcript management
- **curve25519-dalek** for Pedersen commitments where Ristretto255 suffices
- **subtle** crate for all constant-time operations

The custom hash function, fake bulletproof verification, and blinding-in-commitment-struct patterns identified in the audit report are explicitly prohibited in the genomic security module.

### 4.4 Genomic Attestation Protocol

```
Protocol: ZK-GenomicAttestation

Setup (once per predicate class):
  1. Define R1CS circuit for predicate P
  2. Run trusted setup (Powers of Tau + Phase 2)
  3. Publish verification key VK; retain proving key PK

Prove (per attestation request):
  1. Patient retrieves encrypted genotype from RuVector
  2. Decryption occurs inside TEE (or client-side)
  3. Patient computes Pedersen commitment C = g*G + r*H
  4. Patient generates Groth16 proof pi using PK, (C, H(P), L), (g, r)
  5. Patient sends (C, pi, L, H(P)) to verifier

Verify:
  1. Verifier checks pi against VK with public inputs (C, H(P), L)
  2. If valid: predicate P holds for the committed genotype
  3. Verifier learns NOTHING about g beyond P(g) == true
```

---

## 5. Access Control and Audit

### 5.1 Claims-Based Authorization for Genomic Data

Extending the existing claims system (per the claims-authorizer agent specification), genomic access control introduces domain-specific claim types.

**Role hierarchy.**

```
Level 4: Patient (data subject)
  - Full access to own genomic data
  - Can grant/revoke access to others
  - Can request erasure
  - Can export in standard formats (VCF, FHIR)

Level 3: Clinician (treating physician)
  - Access to patient's clinically relevant variants (with consent)
  - Cannot access raw sequence data without explicit authorization
  - Time-limited access tokens (expire with episode of care)

Level 2: Researcher (IRB-approved)
  - Access to differentially private aggregate statistics
  - No individual-level data without explicit consent + IRB approval
  - Budget-limited queries (see Section 2.3)

Level 1: Analyst (institutional)
  - Access to pre-computed, anonymized summary reports
  - No query capability against individual records
  - Read-only access to published results
```

### 5.2 Fine-Grained Access Specifications

Access control operates at three granularity levels.

```rust
pub enum GenomicAccessScope {
    /// Access to a specific gene (e.g., "BRCA1")
    Gene { gene_symbol: String },
    /// Access to a specific variant (e.g., rs1234567)
    Variant { rsid: String },
    /// Access to a genomic region (e.g., chr17:41196312-41277500)
    Region {
        chromosome: u8,
        start: u64,
        end: u64,
    },
    /// Access to a functional category (e.g., all pharmacogenomic variants)
    Category { category: GenomicCategory },
    /// Access to aggregate statistics only (no individual genotypes)
    AggregateOnly,
}

pub enum GenomicCategory {
    Pharmacogenomic,
    CancerPredisposition,
    CardiovascularRisk,
    CarrierScreening,
    Ancestry,
    Forensic,    // Highly restricted: STR profiles, Y-haplogroups
}
```

**Policy example.** A pharmacist checking drug interactions receives a claim of the form:

```yaml
claim:
  role: clinician
  scope: genomic:category:pharmacogenomic
  patient_id: "patient-uuid"
  valid_from: "2026-02-11T00:00:00Z"
  valid_until: "2026-02-11T23:59:59Z"
  access_level: predicate_only    # Can verify ZK proof, cannot see raw genotype
  audit_required: true
```

### 5.3 Immutable Audit Log

All genomic data access events are recorded in a hash-chained, append-only audit log extending the UnifiedWitnessLog (ADR-CE-017).

```rust
pub struct GenomicAccessWitness {
    /// Sequential event ID
    pub event_id: u64,
    /// Hash of previous witness (chain integrity)
    pub prev_hash: [u8; 32],
    /// Timestamp (from trusted time source)
    pub timestamp: u64,
    /// Who accessed the data
    pub accessor: AccessorIdentity,
    /// What was accessed
    pub resource: GenomicAccessScope,
    /// Access type
    pub action: AccessAction,
    /// Authorization decision
    pub decision: AuthorizationDecision,
    /// Claims presented
    pub claims_presented: Vec<String>,
    /// Consent reference (if applicable)
    pub consent_id: Option<String>,
    /// Hash of this record
    pub self_hash: [u8; 32],
}

pub enum AccessAction {
    Read,
    Query { query_hash: [u8; 32] },
    Export { format: ExportFormat },
    ZkProofGeneration { predicate_hash: [u8; 32] },
    AggregateQuery { epsilon_spent: f64 },
    Deletion { scope: DeletionScope },
}
```

The audit log is stored in a separate, write-only partition with independent backup. The hash chain provides tamper evidence: any modification to a historical record breaks the chain, detectable in O(n) time via sequential verification or O(log n) via Merkle tree indexing.

### 5.4 Consent Management

GDPR Article 7 requires freely given, specific, informed, and unambiguous consent. For genomic data (Article 9), explicit consent is mandatory.

```rust
pub struct GenomicConsent {
    pub consent_id: String,
    pub patient_id: String,
    pub granted_to: Vec<ConsentRecipient>,
    pub scope: Vec<GenomicAccessScope>,
    pub purpose: ConsentPurpose,
    pub granted_at: u64,
    pub expires_at: Option<u64>,
    pub revocable: bool,          // Must be true for GDPR compliance
    pub revoked_at: Option<u64>,
    pub signature: ConsentSignature,
}

pub enum ConsentPurpose {
    ClinicalCare,
    Pharmacogenomics,
    ResearchSpecific { study_id: String, irb_approval: String },
    ResearchBroad,      // Requires re-consent for each new study under GDPR
    CarrierScreening,
    AncestryAnalysis,
}
```

### 5.5 GDPR Article 17: Right to Erasure via Cryptographic Deletion

Physical deletion of genomic data from all backups, replicas, and derived products is operationally difficult. Instead, we implement **cryptographic deletion**: the data remains encrypted, but the encryption key is irrevocably destroyed, rendering the ciphertext computationally indistinguishable from random noise.

**Protocol.**

```
CryptographicDeletion(patient_id):
  1. Identify all DEKs associated with patient_id
  2. Re-verify erasure request (consent, identity verification)
  3. Record deletion request in audit log (this record is RETAINED)
  4. Destroy all copies of the patient-specific DEK:
     a. HSM: invoke key destruction command with audit witness
     b. All replicas: send key revocation via authenticated channel
     c. Key escrow (if any): destroy escrowed copy
  5. Mark all ciphertext blocks as "cryptographically deleted" in metadata
  6. Record completion in audit log with HSM attestation of key destruction
  7. Retain: audit log entries, anonymized aggregate statistics (already DP-protected)
  8. Return erasure confirmation with audit trail hash
```

**Key isolation requirement.** Each patient's genomic data must be encrypted under a patient-specific key (or a key derivable only with patient-specific material). Shared encryption keys across patients would make individual erasure impossible without re-encrypting all other patients' data.

---

## 6. Compliance Framework Mapping

### 6.1 HIPAA Compliance

| HIPAA Requirement | Implementation | Section |
|-------------------|---------------|---------|
| 164.312(a)(1) Access control | Claims-based RBAC with genomic scopes | 5.1, 5.2 |
| 164.312(b) Audit controls | Hash-chained GenomicAccessWitness log | 5.3 |
| 164.312(c)(1) Integrity | Signed deltas (ADR-DB-010), HMAC on Tier 3 data | 3.3 |
| 164.312(d) Authentication | mTLS, JWT with claims, TEE attestation | 5.1 |
| 164.312(e)(1) Transmission security | TLS 1.3 minimum, CKKS for computation | 3.1, 7.1 |
| 164.530(c) Minimum necessary | Fine-grained per-gene/variant access scopes | 5.2 |
| 164.524 Right of access | Patient-level export in VCF/FHIR formats | 5.1 |

### 6.2 GINA Compliance

| GINA Provision | Implementation |
|----------------|---------------|
| Title I: Health insurance non-discrimination | Genetic data inaccessible to insurance claims processors; enforced via claims system with no `insurance_underwriting` scope |
| Title II: Employment non-discrimination | Employer roles excluded from all genomic access scopes; audit log flags any access attempt from employer-classified entities |
| Forensic exemption (GINA does not cover law enforcement) | All law enforcement access requires court order; separate audit trail; data subject notified unless court orders otherwise |

### 6.3 GDPR Compliance (Article 9 -- Special Category Data)

| GDPR Requirement | Implementation | Section |
|-------------------|---------------|---------|
| Article 9(2)(a) Explicit consent | GenomicConsent with purpose limitation | 5.4 |
| Article 17 Right to erasure | Cryptographic deletion via key destruction | 5.5 |
| Article 20 Data portability | VCF export with patient's own key | 5.1 |
| Article 25 Data protection by design | Selective encryption, differential privacy by default | 2, 3.3 |
| Article 32 Security of processing | TEE, CKKS, HNSW timing protection | 3, 7 |
| Article 35 DPIA requirement | Mandatory before any new genomic processing activity | Operational |
| Recital 51 Special category processing | All genomic data classified as Article 9 by default | Architecture-wide |

### 6.4 FDA 21 CFR Part 11

When the DNA Analyzer supports clinical decision-making (e.g., pharmacogenomic dosing recommendations, variant pathogenicity classification used in diagnosis):

| Requirement | Implementation |
|-------------|---------------|
| 11.10(a) Validation | Validated variant calling pipeline with known-truth benchmarks (Genome in a Bottle) |
| 11.10(b) Accurate copies | Cryptographic hash verification on all data copies |
| 11.10(c) Record protection | Encryption at rest (Tier 1/2), integrity protection (all tiers) |
| 11.10(d) System access control | Claims-based access with role hierarchy |
| 11.10(e) Audit trail | GenomicAccessWitness with hash chain |
| 11.10(g) Authority checks | Consent verification before any write operation |
| 11.50 Electronic signatures | Ed25519 signatures on clinical reports, linked to identity |
| 11.70 Signature binding | Signature covers document hash; any modification invalidates |

### 6.5 ISO 27001 / 27701 Controls

| Control | Description | Genomic Implementation |
|---------|-------------|----------------------|
| A.8.2 Information classification | Three-tier classification (Sensitive/Moderate/Reference) | 3.3 |
| A.9.4 System and application access control | Claims-based RBAC with genomic scopes | 5.1 |
| A.10.1 Cryptographic controls | CKKS for computation, AES-256-GCM at rest, TLS 1.3 in transit | 3 |
| A.12.4 Logging and monitoring | Hash-chained audit trail with tamper detection | 5.3 |
| A.18.1 Compliance with legal requirements | Compliance mapping table (this section) | 6 |
| 27701-7.2.2 Identifying purposes | ConsentPurpose enum with explicit purpose limitation | 5.4 |
| 27701-7.3.4 Providing mechanism to withdraw consent | Consent revocation with cryptographic deletion | 5.4, 5.5 |

---

## 7. Secure Computation Pipeline

### 7.1 Trusted Execution Environments

All genomic computations that require access to plaintext genotype data occur inside a **trusted execution environment** (TEE). The TEE provides hardware-enforced memory encryption and attestation.

**Supported TEE platforms.**

| Platform | Technology | Use Case |
|----------|-----------|----------|
| Intel | TDX (Trust Domain Extensions) | Cloud-based variant calling, batch processing |
| Intel (legacy) | SGX (Software Guard Extensions) | Enclave-based key management, small computations |
| ARM | TrustZone + CCA | Edge/mobile genomic analysis |
| AMD | SEV-SNP | Cloud VMs with encrypted memory |

### 7.2 Memory Encryption for Vector Operations

Within the TEE, the RuVector HNSW index operates on decrypted vectors. The TEE's memory encryption engine (e.g., Intel TME or AMD SME) ensures that even a physical memory dump by a hypervisor or co-tenant yields only ciphertext.

**HNSW timing attack mitigation.** Inside the TEE, we additionally implement:

1. **Constant-iteration traversal.** Each HNSW layer search always visits exactly `ef_construction` nodes, with dummy comparisons for nodes that would not normally be visited. This prevents timing-based inference about query proximity to stored vectors.

2. **Oblivious memory access.** For Tier 1 (Sensitive) vectors, memory access patterns are made data-independent via Path ORAM. The overhead is O(log^2 N) per access, but applies only to the ~2% of vectors classified as Tier 1.

```rust
pub struct ObliviousHnswConfig {
    /// Enable constant-time traversal (pads to max iterations)
    pub constant_time_search: bool,
    /// Enable Path ORAM for sensitive-tier vectors
    pub oram_for_sensitive: bool,
    /// Maximum ORAM block count (determines tree height)
    pub oram_capacity: usize,
    /// Enable dummy distance computations
    pub dummy_comparisons: bool,
}
```

### 7.3 Attestation Chain for Result Provenance

Every computation result carries an **attestation chain** proving that:

1. The computation occurred inside a genuine TEE (hardware attestation quote).
2. The TEE was running the expected software version (measurement/hash of loaded code).
3. The input data was integrity-verified (hash of input ciphertexts).
4. The output was produced by the attested code from the attested inputs.

```rust
pub struct ComputationAttestation {
    /// TEE platform attestation quote (Intel TDX report or SGX quote)
    pub tee_quote: Vec<u8>,
    /// Hash of the binary loaded into the TEE
    pub code_measurement: [u8; 48],
    /// Hash of all input ciphertexts
    pub input_hash: [u8; 32],
    /// Hash of the computation output
    pub output_hash: [u8; 32],
    /// Timestamp from TEE-internal trusted clock
    pub timestamp: u64,
    /// Signature over the above fields using TEE-bound signing key
    pub signature: Vec<u8>,
    /// Certificate chain linking TEE signing key to platform root of trust
    pub certificate_chain: Vec<Vec<u8>>,
}
```

For clinical-grade results (FDA 21 CFR Part 11), the attestation chain serves as the electronic record's integrity proof. The certificate chain is rooted in the hardware manufacturer's root certificate (Intel, AMD, or ARM), providing a hardware root of trust independent of the software operator.

### 7.4 End-to-End Secure Pipeline

```
Patient Device          Cloud TEE                    Clinician
     |                      |                            |
     |-- Encrypted genome ->|                            |
     |                      |-- TEE attestation -------->|
     |                      |                            |-- Verify attestation
     |                      |<- Attested public key -----|
     |                      |                            |
     |                      | [Inside TEE:]              |
     |                      |   Decrypt genome           |
     |                      |   Run variant calling      |
     |                      |   Generate ZK proof        |
     |                      |   Re-encrypt results       |
     |                      |   Sign attestation         |
     |                      |                            |
     |<- Encrypted result --|                            |
     |<- Attestation -------|-- Attestation copy ------->|
     |<- ZK proof ----------|-- ZK proof copy ---------->|
     |                      |                            |
     |                      | [TEE scrubs memory]        |
     |                      |                            |
```

---

## 8. Implementation Priorities

### Phase 1 (Weeks 1-4): Foundation

| Task | Priority | Dependency |
|------|----------|------------|
| Implement GenomicAccessScope and claims extensions | P0 | ADR-010 claims system |
| Deploy GenomicAccessWitness audit log | P0 | ADR-CE-017 witness log |
| Implement privacy budget ledger | P0 | Existing DP module |
| Define encryption tier classification policy | P1 | ClinVar/OMIM integration |
| Implement cryptographic deletion protocol | P1 | Key management infrastructure |

### Phase 2 (Weeks 5-8): Cryptographic Infrastructure

| Task | Priority | Dependency |
|------|----------|------------|
| CKKS integration for Tier 1 vector operations | P0 | Scheme parameter selection |
| Groth16 circuit for genotype predicates | P1 | arkworks-rs integration |
| TEE attestation chain implementation | P1 | Hardware availability |
| HNSW constant-time traversal mode | P2 | Core HNSW refactor |

### Phase 3 (Weeks 9-12): Protocol Integration

| Task | Priority | Dependency |
|------|----------|------------|
| Secure multi-party aggregation protocol | P1 | Institutional partnerships |
| End-to-end encrypted analysis pipeline | P1 | Phase 2 completion |
| GDPR consent management system | P0 | Legal review |
| Compliance certification preparation | P1 | All phases |

---

## Consequences

### Benefits
- Genomic data protected by defense-in-depth: encryption, access control, differential privacy, and zero-knowledge proofs
- Compliance with all major regulatory frameworks (HIPAA, GINA, GDPR, FDA 21 CFR Part 11)
- Cryptographic deletion provides a technically sound implementation of the right to erasure
- ZK proofs enable new use cases (pharmacogenomic safety checks, carrier screening) without requiring full genotype disclosure

### Risks
- CKKS overhead may exceed 10x target for complex multi-locus queries requiring deep multiplicative circuits
- Trusted setup ceremony for Groth16 introduces a trust assumption (mitigable via MPC-based setup)
- TEE availability varies across cloud providers; must maintain software-only fallback
- Privacy budget exhaustion may frustrate researchers; requires clear communication of budget policies

### Breaking Changes
- All genomic data access now requires claims with GenomicAccessScope; existing API consumers must migrate
- Audit log schema change from generic WitnessLog to GenomicAccessWitness
- Embedding model outputs for Tier 1 regions are encrypted; downstream consumers must support CKKS ciphertext handling

---

## References

1. Homer, N., et al. (2008). "Resolving Individuals Contributing Trace Amounts of DNA to Highly Complex Mixtures Using High-Density SNP Genotyping Microarrays." *PLoS Genetics*.
2. Gymrek, M., et al. (2013). "Identifying Personal Genomes by Surname Inference." *Science*.
3. Erlich, Y., & Narayanan, A. (2014). "Routes for Breaching and Protecting Genetic Privacy." *Nature Reviews Genetics*.
4. Dwork, C., Roth, A., & Vadhan, S. (2010). "Boosting and Differential Privacy." *FOCS*.
5. Cheon, J.H., Kim, A., Kim, M., & Song, Y. (2017). "Homomorphic Encryption for Arithmetic of Approximate Numbers." *ASIACRYPT*.
6. Groth, J. (2016). "On the Size of Pairing-Based Non-interactive Arguments." *EUROCRYPT*.
7. NIST SP 800-188: De-Identifying Government Datasets.
8. OWASP Genomic Data Security Guidelines (2025).
9. GA4GH Framework for Responsible Sharing of Genomic and Health-Related Data.

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | Security Architecture Team | Initial proposal |

# Part II — Reproducible Computational Verification Protocol
**CCDR v7.10 / Synthesis v3.9 Standard Model derivation · `rakovpublic/algebraphysicsflows`**

Companion to *Algebra-Level Standard Model Structure in CCDR v7.10 and Synthesis v3.9* (Part I). Status tags are the same as in Part I: **[R]** = in the recorded Maven run, **[S]** = in the supplementary class `CCDRStandardModelDerivationTest`, **[I]** = interpretive / hand proof.

---

## 0. Read this first: provenance and scope

| Item | Fact |
|---|---|
| Recorded run | `mvn -f groupimp/pom.xml clean` then `mvn -f groupimp/pom.xml test`; Java 17.0.4.1, Maven 3.9.11, Surefire 3.2.5; source/target level 8 |
| Result | **95 test classes, 937 tests, 937 passed, 0 failures, 0 errors, 0 skipped**; 66 s wall time (53.958 s summed Surefire time) |
| Tested commit / branch | `d577dca` on `codex/ccdr-v710-synthesis-v39-pipeline`; branch tip `a2672fb` adds `docs/CCDR_SM_DERIVATION_VERIFICATION_REPORT.md` |
| **Not `master`** | `master` (`43547a6` when checked) has **93** test classes and lacks the CCDR pipeline classes. The 95/937 state exists only on the branch above until it is merged. Always check out the branch. |
| CCDR suites | 7 tests in the Maven count: 2 in `CCDRStandardModelPipelineTest`, 5 in `CCDRPhysicalDataValidationTest`; `CCDRStandardModelFlowTest` adds 1 |
| Supplementary class | `CCDRStandardModelDerivationTest`, 6 tests. **Not** part of the 937. It was compiled with `javac` and run with JUnit 4.13.2 against the branch classes in an environment where Maven Central was unreachable. **It has not been run under Maven**; run it yourself (§3). |
| External data | PDG 2025 numbers are those of the verification report. The repository does not derive any measurement. |

What this protocol does **not** establish: any assignment of the 16 Weyl states to minimal left ideals of `DixZ2(Q)`; a physical gauge identification; RG running of 3/8; a full Hahn–Mal'cev–Neumann division ring; a source for the void-kurtosis anchor 4.45 ± 0.20.

---

## 1. Prerequisites and checkout

```bash
git clone https://github.com/rakovpublic/algebraphysicsflows.git
cd algebraphysicsflows
git fetch origin codex/ccdr-v710-synthesis-v39-pipeline
git checkout d577dca            # exact tested commit; or the branch tip a2672fb (adds the report)
java -version                   # 17 recommended (project targets Java 8 bytecode)
mvn -version                    # 3.9.x
```

Sanity checks on the checkout:

```bash
# 95 test classes expected on the branch (93 on master)
find groupimp/src/test -name '*Test.java' | wc -l

# Part I, Prop. 6.1: algebraic carriers import nothing from mathematics.topology (no output expected)
cd groupimp/src/main/java/mathematics/numbers
grep -l "import mathematics.topology" RationalTwistedLatticeAlgebra.java RationalTwistedHahnSeries.java \
  RationalDixonZ2.java RationalOctonionDerivations.java StandardModelWeylGeneration.java \
  RationalCyclotomic.java HighPrecisionAlgebraMetrics.java CCDRChiralFold.java
cd -
```

---

## 2. Full reproduction (the headline claim)

```bash
mvn -f groupimp/pom.xml clean
mvn -f groupimp/pom.xml test
```

`clean` prevents stale Surefire XML from corrupting totals. Expect `Tests run: 937, Failures: 0, Errors: 0, Skipped: 0`. To recount from the XML reports:

```bash
python3 - <<'EOF'
import glob, xml.etree.ElementTree as ET
t=f=e=s=0; n=0
for p in glob.glob('groupimp/target/surefire-reports/TEST-*.xml'):
    r=ET.parse(p).getroot(); n+=1
    t+=int(r.get('tests'));f+=int(r.get('failures'));e+=int(r.get('errors'));s+=int(r.get('skipped'))
print(n,'classes',t,'tests',f,'failures',e,'errors',s,'skipped')
EOF
```

The CCDR-only subset (prints the residual lines quoted in §5):

```bash
mvn -f groupimp/pom.xml -Dtest='CCDRStandardModelPipelineTest,CCDRStandardModelFlowTest,CCDRPhysicalDataValidationTest' test
```
Expected: 8 tests, 0 failures. **[R]**

---

## 3. Supplementary class (adds the checks marked [S])

```bash
cp CCDRStandardModelDerivationTest.java groupimp/src/test/java/operations/
mvn -f groupimp/pom.xml -Dtest=CCDRStandardModelDerivationTest test       # expect 6 tests, OK
```

Fallback if Maven is unavailable (what was done when this package was prepared): compile `groupimp/src/main/java` and the tests with `javac` (log4j jars from your `~/.m2` on the classpath) and run

```bash
java -cp <classes>:<test-classes>:junit-4.13.2.jar:hamcrest-core-1.3.jar \
  org.junit.runner.JUnitCore operations.CCDRStandardModelDerivationTest
```
Expected: `OK (6 tests)`; with the three upstream CCDR classes added, `OK (14 tests)`.

The class asserts exact algebraic identities, sign bookkeeping and parameter-free numerics only. Helpers marked `TEST-LOCAL` (e.g. `specializeFirst`, `halveFirst`, the Dixon lift) are proposed extensions, not library methods.

---

## 4. Step-by-step protocol

Common idiom for flows: `ConcreteMathematics math = new ConcreteMathematics();` then
`math.flow(math.<algebra>, Collections.singletonList(x)).<T>performAlgebraTransfer("<op>")` or `.performOneOperandOperation("<op>")`; read a result with `flow.collectAlgebraItems().get(0).perform().getResult()` or `flow.collect()` (string form). Operation names: the short alias is used when the first algebra owns the operation; transfers *into* an algebra use the long id, e.g. `"HMN(H(Q);Z^N,sigma,tau).group-basis"`.

### Step 1 — Runtime-rank twisted bulk (Axioms 1–2, Prop. 2.2) · class `RationalTwistedLatticeAlgebra`
Rank is an argument, never a constant.
```java
int rank = 5;                                   // any N >= 1 (>= 5 for the fold); try 5, 8, 11
int[][] c = new int[rank][rank]; c[1][2] = c[2][1] = 1;   // alternating F2 matrix
RationalTwistedLatticeAlgebra gx = RationalTwistedLatticeAlgebra.basis(rank, c,
        new RationalTwistedLatticeAlgebra.Exponent(new long[]{0,1,0,0,0}), q);   // X^g q
// product rule: (X^g q)(X^h r) = tau(g,h) X^{g+h} q sigma_g(r)
assertEquals(a.multiply(b).multiply(d), a.multiply(b.multiply(d)));      // associativity
assertNotEquals(gx.multiply(hy), hy.multiply(gx));                        // genuine non-commutativity
```
- **[R]** `CCDRStandardModelPipelineTest.fullTwoTierDerivationPipeline…` at N = 5, 8; sign matrix `c12 = c21 = 1`.
- **[S]** `step2_arbitraryRankBulk_associativity_and_BSM1_residue_endomorphism`: random associativity at N ∈ {1,2,3,4,5,6,7,9,11}.
- Expected: no failure; ranks outside {1,2,4,8} behave identically (the Bott–Milnor–Kervaire trap is avoided because the bulk is infinite-dimensional).

### Step 2 — BSM1 and the residue (Prop. 2.3) · `darkMatterEndomorphism`, `hasOddFirstCoordinateSupport`
```java
bulk.darkMatterEndomorphism();                  // g0 -> 2 g0; refuses if the cocycle touches coordinate 0
```
Checks: exponent `(1,0^7) -> (2,0^7)` at rank 8 **[R]**; multiplicative, additive, injective; image has no odd-g0 support; `X^{e0}` central **[S]**; a cocycle with `S01 = 1` raises `MathFailure` `OPERATION_UNDEFINED` with witness `tau(e0,e1) = -1`, `tau(2e0,e1) = +1` **[S]**; flow route `…group-basis` → `dark-matter-endomorphism` → `has-dark-matter-residue` gives `true` then `false` **[S]**.

The t = 1 specialization (Prop. 2.4): `RationalTwistedHahnSeries.projectChi5ToOneInDixon()` or flow op `"chi5-to-one-dixon"`; `GROUP_GENERATOR ↦ i`. **[R]** for the generator, **[S]**
`step2b_residue_projection_is_evaluation_not_valuation_and_kernel_is_BSM1_stable` for the ring-homomorphism property, the ideal `(t−1)`, the valuation-versus-residue example and pole refusal. Interpret as the (t−1)-adic residue, **not** `group-valuation` / `leading-coefficient`.

### Step 3 — The 5→4 fold and the orientation sign (Props. 3.2, 3.4) · `foldParityClassesToDixonBoundary`, `SimplicialCollapse`, `CCDRChiralFold`
```java
RationalDixonZ2 boundary = bulk.foldParityClassesToDixonBoundary();   // rank >= 5; packed index 4*class + component
// flow: math.flow(math.twistedLattice, Collections.singletonList(bulk)).performAlgebraTransfer("parity-fold-to-dixon-boundary")
```
- **[R]** `fiveToFourParityProjectionIsSurjectiveAdditiveAndNotMultiplicative`: all 128 coordinates hit; additive; not multiplicative; folds equal at rank 5 and 8.
- **[S]** `step6_parity_fold_properties_chiral_lock_and_end_to_end_flow`: coordinates beyond the fifth ignored; explicit counterexample `fold(X^{e0}²) = +1` but `fold(X^{e0})² = −1`; kernel is an ideal at rank 5 and **not** at rank > 5 when the cocycle couples a beyond-fifth coordinate; flow equals direct call at ranks 5, 6, 8, 11.
- Orientation **[R]**: collapse of the 2-simplex at face (0,2): chain-homotopy entry −1. **[S]** `step3_cdt_cells_signed_collapse_subdivision_and_orientation_lock`: incidence `(−1)^position`; incidence alone is convention-dependent; the SM-D11 observable is the determinant against the foliation frame, (4,1) → +1, (1,4) → −1; signed subdivision of ∂Δ⁴ is a chain map with 5·4! = 120 pieces and an isomorphism on H₃; 15 collapse steps to a point.
- Sign lock **[R][S]**: `new CCDRChiralFold(ε5, arrow, simplexSign, cmbSign)`; `chiralitySign() = ε5·arrow`; left-chiral iff > 0. Report input `(−1,+1,−1,−1)` → `−1`, `smD11Consistent()`, `pA30Consistent()`, `cl9Consistent()` all true, **right-chiral by repository convention**. OP20: `(+1,−1)` and `(−1,+1)` both give −1; all four (ε5, arrow) pairs split 2 : 2 between chiralities.
- Caveat: the upstream tests pass the collapse incidence (−1) directly as the sign inputs. That is a bookkeeping choice, not a derivation of SM-D11.

### Step 4 — Dixon boundary, gauge generators, anomalies (Props. 4.1–4.3) · `RationalDixonZ2`, `RationalOctonionDerivations`, `StandardModelWeylGeneration`
```java
List<RationalMatrix> su3 = RationalOctonionDerivations.su3StabilizerCandidates();      // 8, annihilate l = e4
List<RationalMatrix> su2 = RationalOctonionDerivations.quaternionAdjointGenerators();  // 3
StandardModelWeylGeneration.states();               // 16 left-handed Weyl states, Q = T3 + Y
```
- **[R]** closure in the rational span; octonion and Dixon associator witnesses (`e1,e2,e4`); 128×128 left-multiplication matrices; Witt idempotents `w₊+w₋ = 1`, `w₊w₋ = 0`; traces `Tr Y = 0`, `Tr Y³ = 0`, `Tr T3² = 2`, `Tr Q² = 16/3`, mixed and cubic SU(3) anomalies 0, doublet parity 0, ratio `3/8`.
- **[S]** `step4_gauge_algebras_alpha_centralizer_and_weyl_anomalies`: derivation property on all 64 basis products; exact Killing forms negative definite (compact); **α normalizes su(3) with `tr Ad_α = −2` (3 fixed + 5 flipped); fixed elements lift to derivations of `DixZ2`, flipped ones do not.** Consequence: the full su(3) closes on `O(Q)` but is **not** a Lie algebra of derivations of `DixZ2` under the natural lift `1⊗1⊗D`. Report this honestly; it does not exclude other lifts or a reading on `O(Q)` alone.
- Not asserted anywhere: a map from the 16 states to minimal left ideals.

### Step 5 — Cyclotomic flavour closure and PDG residuals (Props. 5.1–5.2) · `RationalCyclotomic`, `HighPrecisionAlgebraMetrics`
```java
RationalCyclotomic z = RationalCyclotomic.zeta(12);            // class of x mod x^4 - x^2 + 1
z.evaluate(40);                                                 // sqrt(3)/2 + i/2
HighPrecisionAlgebraMetrics.koide(me, mmu, mtau, 40);
HighPrecisionAlgebraMetrics.ckmJarlskog(s12,s23,s13,c12,c23,c13, phaseOrRadians, 40);
```
- **[R]** `CCDRPhysicalDataValidationTest` (5 tests) prints the residual lines in §5.
- **[S]** `step5_cyclotomic_closure_koide_parametrization_and_jarlskog_residuals`: `Φ₁₂`, `ζ³ = i`, `ζ⁶ = −1`, norm 1; Koide `Q = 1/3 + ρ²/6` (so `Q = 2/3 ⇔ ρ² = 2`; this identity is an algebraic remark, not a repository claim); report values reproduced to ≤ 1e-30; Jarlskog linear in sin δ and sign-flipped by conjugation; exact rational-point phase `(1−t²+2it)/(1+t²)` has norm 1 and, with left chirality, `smD8MatchesNu` holds while right chirality returns the negative sine.

### Step 6 — Native `IAlgebraFlow` execution and the JUnit template
- **[R]** a simplicial flow `SimplicialCollapse.from-absolute-face` → `on-absolute-chain` → `boundary` → `coordinates` returns an `IAlgebraFlow<IntegerVector>` whose collected coordinates equal the direct computation; parity and `chi5-to-one` transfers run through flows.
- **[S]** the supplementary class (§3) bundles Steps 1–5 and the flow equalities as the JUnit template `CCDRStandardModelDerivationTest`. Add it to `groupimp/src/test/java/operations/`; it needs the branch classes.

---

## 5. Expected numerical output (40-digit `BigDecimal`)

| Quantity | Expected | Anchor / status |
|---|---|---|
| Koide Q, PDG 2025 central | `0.6666644634145001750698620530280868826435`; Q − 2/3 ≈ −2.20325217×10⁻⁶ | 2/3 · m_τ = 1776.93 ± 0.09 MeV · **Pass**: 1σ interval [0.66665938232272398, 0.66666954423834577] brackets 2/3 |
| Koide Q, m_τ = 1776.86 | `0.66666051147738557` | historical value |
| CKM J, measured δ = 1.147 | `3.1169935287554704725726246712e-5` (residual −3.00647×10⁻⁸) | PDG J = 3.12 (+0.13/−0.12)×10⁻⁵ · **Pass** (inputs s12 = 0.22501, s23 = 0.04183, s13 = 0.003732); validates the evaluator, not a derivation |
| CKM J, ζ₁₂ phase | `1.7097509558246345705944971779e-5` | **Tension (SM-D8)**; ≈ 12 lower widths below the PDG central |
| sin(1.147) | `0.9115343723414106708707` | vs ν = 0.00508 ± 0.0012 (residual 0.9064543723) and 4ν = 0.02032 (residual 0.8912143723) |
| Tr T3²/Tr Q² | `3/8 = 0.375` | vs sin²θ_W(M_Z) MS-bar = 0.23122 ± 0.00006 (Δ = 0.14378): **scale-dependent**, RG required |
| BSM1, rank 8 | exponent `(1,0⁷) → (2,0⁷)`; δ = ν·m_DM = 0.508 GeV (m_DM = 100 GeV) | ν, m_DM assumed |
| σ₀ = σ_eff/ν² | `1.937503875007750015500031…e-44 cm²` for assumed σ_eff = 5×10⁻⁴⁹ cm² | σ_eff is 20× below the supplied 10⁻⁴⁷ floor and 4.4× below LZ 4.2 t·yr strongest SI bound 2.2×10⁻⁴⁸ cm² at 40 GeV/c²; **conditional**, nothing measured |
| (N−4)² | `16` vs k₄ ≈ 4.45 ± 0.20 | **conditional / OP20**; no derived map; no primary source for the anchor was found in project data or the VAST publication |

Only the documented parameters ν ≈ 5.08×10⁻³, k₄ ≈ 4.45, δ_BSM1 ≈ 0.5 GeV are used; no other empirical number is introduced.

---

## 6. Failure modes

| Symptom | Cause / fix |
|---|---|
| `Tests run` is 93-class / lower total | You are on `master`; check out the branch. |
| Totals differ from 937 | Stale `target/`; run `mvn clean` first. |
| `MathFailure OPERATION_UNDEFINED` from `darkMatterEndomorphism` | Cocycle touches coordinate 0 (correct refusal, Prop. 2.3). |
| `MathFailure` from the fold | Rank < 5 (correct). |
| Supplementary class does not compile | Missing branch classes (`orientCyclotomicPhase`, `HighPrecisionAlgebraMetrics`). |
| Platform-encoding warning (Cp1251) | Harmless; set `-Dfile.encoding=UTF-8`. |
| Maven cannot resolve dependencies | Network/proxy; use the `javac` fallback in §3 and say so when reporting. |

## 7. Referee checklist
1. Branch/commit match; 95 classes; 937/937. 2. Supplementary class 6/6 (state whether run under Maven). 3. Rank varied, never fixed at 8. 4. BSM1 refuses a bad cocycle. 5. Fold: additive, surjective, not multiplicative (counterexample −1 vs +1). 6. su(3) lift boundary stated. 7. Residual table matches §5, Koide and J(measured) flagged pass, SM-D8 flagged tension, 3/8 flagged scale-dependent. 8. Non-claims in §0 repeated in any summary.

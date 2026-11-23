package operations;

import algebra.IAlgebraItem;
import algebra.concrete.ConcreteMathematics;
import algebraflow.IAlgebraFlow;
import mathematics.calculus.Polynomial;
import mathematics.core.MathFailure;
import mathematics.foundations.FiniteSet;
import mathematics.linear.IntegerMatrix;
import mathematics.linear.IntegerVector;
import mathematics.linear.RationalMatrix;
import mathematics.linear.RationalVector;
import mathematics.numbers.*;
import mathematics.structures.AbelianGroupHomomorphism;
import mathematics.topology.*;
import org.junit.Test;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.*;

import static org.junit.Assert.*;

/**
 * End-to-end algebra-level verification of the CCDR v7.10 / Synthesis v3.9 two-tier architecture on the
 * concrete exact carriers of algebraphysicsflows.
 *
 * SCOPE. Every assertion below is an exact algebraic identity, a sign-bookkeeping identity, or a
 * parameter-free numerical identity. Nothing here asserts a physical map from fields to Dixon
 * coordinates or to a minimal left ideal, and nothing fits an empirical number.
 *
 * Helpers marked "TEST-LOCAL" are not part of the library; they are proposed extensions.
 *
 * REQUIREMENTS. Place in groupimp/src/test/java/operations/. It needs the classes of branch
 * codex/ccdr-v710-synthesis-v39-pipeline (HighPrecisionAlgebraMetrics, RationalCyclotomic,
 * CCDRChiralFold.orientCyclotomicPhase/smD8*), which master (43547a6) does not yet contain.
 * Run: mvn -f groupimp/pom.xml -Dtest=CCDRStandardModelDerivationTest test   (6 tests expected).
 */
public class CCDRStandardModelDerivationTest {
    // ------------------------------------------------------------------ utilities
    private static BigInteger z(long n) { return BigInteger.valueOf(n); }
    private static Rational r(long n) { return Rational.of(n); }
    private static final MathContext MC = new MathContext(80, RoundingMode.HALF_EVEN);
    private static final String GROUP_BASIS = "HMN(H(Q);Z^N,sigma,tau).group-basis";

    /** Deterministic pseudo-random source so that every run is reproducible. */
    private static final class Rng {
        private long s;
        Rng(long seed) { s = seed; }
        int next(int bound) { s = s * 6364136223846793005L + 1442695040888963407L; return (int) ((s >>> 33) % bound); }
    }

    private static <T> T result(IAlgebraFlow<T> flow) {
        List<IAlgebraItem<T>> items = flow.collectAlgebraItems();
        assertEquals(1, items.size());
        return items.get(0).perform().getResult();
    }

    /** Admissible alternating F2 sign matrix: trivial on coordinate 0 (the BSM1 coordinate). */
    private static int[][] cocycle(int n) {
        int[][] m = new int[n][n];
        for (int i = 1; i + 1 < n; i += 2) { m[i][i + 1] = 1; m[i + 1][i] = 1; }
        return m;
    }

    private static RationalQuaternion randomQuaternion(Rng g) {
        RationalQuaternion q;
        do {
            q = new RationalQuaternion(r(g.next(5) - 2), r(g.next(5) - 2), r(g.next(5) - 2), r(g.next(5) - 2));
        } while (q.equals(RationalQuaternion.ZERO));
        return q;
    }

    private static RationalTwistedLatticeAlgebra randomElement(int n, int[][] c, Rng g, int terms) {
        Map<RationalTwistedLatticeAlgebra.Exponent, RationalQuaternion> map = new TreeMap<>();
        for (int t = 0; t < terms; t++) {
            long[] e = new long[n];
            for (int i = 0; i < n; i++) e[i] = g.next(5) - 2;
            map.merge(new RationalTwistedLatticeAlgebra.Exponent(e), randomQuaternion(g), RationalQuaternion::add);
        }
        return new RationalTwistedLatticeAlgebra(n, c, map);
    }

    /** TEST-LOCAL. Specialisation of the central BSM1 coordinate at 1: X^(g0,g') -> X^(0,g'). */
    private static RationalTwistedLatticeAlgebra specializeFirst(RationalTwistedLatticeAlgebra x) {
        Map<RationalTwistedLatticeAlgebra.Exponent, RationalQuaternion> out = new TreeMap<>();
        for (Map.Entry<RationalTwistedLatticeAlgebra.Exponent, RationalQuaternion> e : x.terms().entrySet()) {
            List<BigInteger> c = new ArrayList<>();
            for (int i = 0; i < x.rank(); i++) c.add(i == 0 ? BigInteger.ZERO : e.getKey().coordinate(i));
            out.merge(new RationalTwistedLatticeAlgebra.Exponent(c), e.getValue(), RationalQuaternion::add);
        }
        out.values().removeIf(q -> q.equals(RationalQuaternion.ZERO));
        return new RationalTwistedLatticeAlgebra(x.rank(), x.cocycleMatrix(), out);
    }

    /** TEST-LOCAL. Inverse of BSM1 on even-g0 support; returns null if some support has odd g0. */
    private static RationalTwistedLatticeAlgebra halveFirst(RationalTwistedLatticeAlgebra y) {
        Map<RationalTwistedLatticeAlgebra.Exponent, RationalQuaternion> out = new TreeMap<>();
        for (Map.Entry<RationalTwistedLatticeAlgebra.Exponent, RationalQuaternion> e : y.terms().entrySet()) {
            if (e.getKey().firstCoordinateOdd()) return null;
            List<BigInteger> c = new ArrayList<>();
            for (int i = 0; i < y.rank(); i++) c.add(i == 0 ? e.getKey().coordinate(0).shiftRight(1) : e.getKey().coordinate(i));
            out.put(new RationalTwistedLatticeAlgebra.Exponent(c), e.getValue());
        }
        return new RationalTwistedLatticeAlgebra(y.rank(), y.cocycleMatrix(), out);
    }

    private static RationalTwistedLatticeAlgebra x0(int n, int[][] c) { // X^(e_0)
        long[] e = new long[n]; e[0] = 1;
        return RationalTwistedLatticeAlgebra.basis(n, c, new RationalTwistedLatticeAlgebra.Exponent(e), RationalQuaternion.ONE);
    }

    private static BigDecimal sqrt(BigDecimal v) {
        BigDecimal x = new BigDecimal(Math.sqrt(v.doubleValue()));
        for (int i = 0; i < 12; i++) x = x.add(v.divide(x, MC), MC).divide(BigDecimal.valueOf(2), MC);
        return x;
    }

    private static void assertClose(BigDecimal expected, BigDecimal actual, String tol) {
        assertTrue("expected " + expected + " got " + actual, expected.subtract(actual).abs().compareTo(new BigDecimal(tol)) < 0);
    }

    // ------------------------------------------------------------------ STEP 2: arbitrary-N bulk and BSM1
    @Test public void step2_arbitraryRankBulk_associativity_and_BSM1_residue_endomorphism() {
        ConcreteMathematics math = new ConcreteMathematics();
        Rng g = new Rng(20260522L);
        // N is a runtime value; ranks outside {1,2,4,8} are deliberately included (Principle 1).
        for (int n : new int[]{1, 2, 3, 4, 5, 6, 7, 9, 11}) {
            int[][] c = cocycle(n);
            RationalTwistedLatticeAlgebra a = randomElement(n, c, g, 3), b = randomElement(n, c, g, 3), d = randomElement(n, c, g, 3);
            // twisted associativity: sigma_g sigma_h = sigma_{g+h}, tau bilinear
            assertEquals("associativity, N=" + n, a.multiply(b).multiply(d), a.multiply(b.multiply(d)));
            // BSM1 is a ring endomorphism, injective, and lands in the even-g0 subring
            RationalTwistedLatticeAlgebra pa = a.darkMatterEndomorphism(), pb = b.darkMatterEndomorphism();
            assertEquals("BSM1 multiplicative, N=" + n, a.multiply(b).darkMatterEndomorphism(), pa.multiply(pb));
            assertEquals(a.add(b).darkMatterEndomorphism(), pa.add(pb));
            assertFalse(pa.hasOddFirstCoordinateSupport());
            assertEquals("injective: halving recovers the source", a, halveFirst(pa));
            // non-surjective: the odd-g0 residue sector is never in the image
            RationalTwistedLatticeAlgebra residue = x0(n, c);
            assertTrue(residue.hasOddFirstCoordinateSupport());
            assertNull(halveFirst(residue));
            // the central coordinate commutes with everything (needed for BSM1 to be a ring map)
            assertEquals(residue.multiply(a), a.multiply(residue));
            // image(BSM1) = full even sector, and A = A_even + X0*A_even (Z/2-grading by g0 mod 2)
            RationalTwistedLatticeAlgebra aEven = pa; // an arbitrary even element
            assertEquals(aEven, halveFirst(aEven).darkMatterEndomorphism());
        }
        // twisted non-commutativity is genuine: X^(e_1) j = sigma(j) X^(e_1) = -j X^(e_1)
        int[][] c3 = cocycle(3);
        RationalTwistedLatticeAlgebra x1 = RationalTwistedLatticeAlgebra.basis(3, c3, new RationalTwistedLatticeAlgebra.Exponent(0, 1, 0), RationalQuaternion.ONE);
        RationalTwistedLatticeAlgebra jj = RationalTwistedLatticeAlgebra.fromQuaternion(3, c3, RationalQuaternion.J);
        assertEquals(x1.multiply(jj), jj.multiply(x1).negate());
        // sign cocycle: X^(e_1) X^(e_2) = - X^(e_2) X^(e_1)
        RationalTwistedLatticeAlgebra x2 = RationalTwistedLatticeAlgebra.basis(3, c3, new RationalTwistedLatticeAlgebra.Exponent(0, 0, 1), RationalQuaternion.ONE);
        assertEquals(x1.multiply(x2), x2.multiply(x1).negate());

        // The admissibility check: a cocycle that touches coordinate 0 must be refused...
        int[][] bad = new int[3][3]; bad[0][1] = 1; bad[1][0] = 1;
        RationalTwistedLatticeAlgebra bx = RationalTwistedLatticeAlgebra.basis(3, bad, new RationalTwistedLatticeAlgebra.Exponent(1, 0, 0), RationalQuaternion.ONE);
        assertEquals(MathFailure.Kind.OPERATION_UNDEFINED, assertThrows(MathFailure.class, bx::darkMatterEndomorphism).kind());
        // ...because doubling g[0] would then break multiplicativity: tau(e0,e1)=-1 but tau(2e0,e1)=+1.
        RationalTwistedLatticeAlgebra by = RationalTwistedLatticeAlgebra.basis(3, bad, new RationalTwistedLatticeAlgebra.Exponent(0, 1, 0), RationalQuaternion.ONE);
        RationalTwistedLatticeAlgebra product = bx.multiply(by);                 // -X^(1,1,0)
        assertEquals(-1, RationalTwistedLatticeAlgebra.tau(new RationalTwistedLatticeAlgebra.Exponent(1, 0, 0), new RationalTwistedLatticeAlgebra.Exponent(0, 1, 0), bad));
        assertEquals(+1, RationalTwistedLatticeAlgebra.tau(new RationalTwistedLatticeAlgebra.Exponent(2, 0, 0), new RationalTwistedLatticeAlgebra.Exponent(0, 1, 0), bad));
        assertEquals(RationalQuaternion.ONE.negate(), product.terms().firstEntry().getValue());

        // Registered-flow route (AlgebraFlow): Z^4 exponent -> basis element -> BSM1 -> residue predicate
        IntegerVector exponent = new IntegerVector(z(1), z(1), z(0), z(0));
        boolean before = result(math.flow(math.integerVectors, Collections.singletonList(exponent))
                .<RationalTwistedLatticeAlgebra>performAlgebraTransfer(GROUP_BASIS).<Boolean>performAlgebraTransfer("has-dark-matter-residue"));
        boolean after = result(math.flow(math.integerVectors, Collections.singletonList(exponent))
                .<RationalTwistedLatticeAlgebra>performAlgebraTransfer(GROUP_BASIS).performOneOperandOperation("dark-matter-endomorphism")
                .<Boolean>performAlgebraTransfer("has-dark-matter-residue"));
        assertTrue(before);
        assertFalse(after);
    }

    // ------------------------------------------------------------------ STEP 2b: pi_{5->4} (residue/specialisation)
    @Test public void step2b_residue_projection_is_evaluation_not_valuation_and_kernel_is_BSM1_stable() {
        ConcreteMathematics math = new ConcreteMathematics();
        // (a) rank-one registered carrier: pi = evaluation at t=1 on the local subring; ring homomorphism there.
        RationalTwistedHahnSeries t = RationalTwistedHahnSeries.fromQuaternion(RationalQuaternion.I.negate()).multiply(RationalTwistedHahnSeries.GROUP_GENERATOR); // t = i^-1 X
        RationalTwistedHahnSeries f = RationalTwistedHahnSeries.fromQuaternion(new RationalQuaternion(r(1), r(2), r(0), r(-1))).multiply(t).add(RationalTwistedHahnSeries.fromQuaternion(RationalQuaternion.J));
        RationalTwistedHahnSeries h = RationalTwistedHahnSeries.fromQuaternion(RationalQuaternion.K).add(t.multiply(t));
        assertEquals(f.projectChi5ToOneInDixon().multiply(h.projectChi5ToOneInDixon()), f.multiply(h).projectChi5ToOneInDixon());
        assertEquals(RationalDixonZ2.fromQuaternion(RationalQuaternion.I), RationalTwistedHahnSeries.GROUP_GENERATOR.projectChi5ToOneInDixon());
        // (b) the two valuations differ: (t-1) has group-valuation 0 and leading coefficient -1, yet lies in ker(pi).
        RationalTwistedHahnSeries tm1 = RationalTwistedHahnSeries.fromPolynomial(new Polynomial(r(-1), r(1)));
        assertEquals(BigInteger.ZERO, tm1.groupValuation());
        assertEquals(RationalQuaternion.scalar(r(-1)), tm1.leadingCoefficient());
        assertTrue(tm1.projectChi5ToOneInDixon().isZero());
        // kernel is a two-sided ideal: pi((t-1) w) = 0 for any w in the local subring
        assertTrue(tm1.multiply(f).projectChi5ToOneInDixon().isZero());
        assertTrue(f.multiply(tm1).projectChi5ToOneInDixon().isZero());
        // (c) poles are outside the domain of pi (not compactifiable: no 4D boundary value)
        RationalTwistedHahnSeries pole = RationalTwistedHahnSeries.scalar(new mathematics.calculus.RationalFunction(new Polynomial(Rational.ONE), new Polynomial(r(-1), r(1))));
        assertThrows(RuntimeException.class, pole::projectChi5ToOneInDixon);
        // (d) the same map through AlgebraFlow
        RationalDixonZ2 viaFlow = result(math.flow(math.twistedHahnMalcevNeumann, Collections.singletonList(f)).<RationalDixonZ2>performAlgebraTransfer("chi5-to-one-dixon"));
        assertEquals(f.projectChi5ToOneInDixon(), viaFlow);

        // (e) rank-N analogue (TEST-LOCAL): specialise the central BSM1 coordinate X0 -> 1.
        Rng g = new Rng(5408L);
        for (int n : new int[]{1, 3, 4, 5, 7}) {
            int[][] c = cocycle(n);
            RationalTwistedLatticeAlgebra a = randomElement(n, c, g, 4), b = randomElement(n, c, g, 4);
            // ring endomorphism (a retraction onto the X0-free subring)
            assertEquals(specializeFirst(a.multiply(b)), specializeFirst(a).multiply(specializeFirst(b)));
            assertEquals(specializeFirst(a), specializeFirst(specializeFirst(a)));
            // BSM1 commutes with the specialisation: 1^2 = 1
            assertEquals(specializeFirst(a), specializeFirst(a.darkMatterEndomorphism()));
            // kernel (X0-1)*A: in ker, absorbing, BSM1-stable
            RationalTwistedLatticeAlgebra one = RationalTwistedLatticeAlgebra.fromQuaternion(n, c, RationalQuaternion.ONE);
            RationalTwistedLatticeAlgebra k = x0(n, c).subtract(one).multiply(a);
            assertTrue(specializeFirst(k).isZero());
            assertTrue(specializeFirst(b.multiply(k)).isZero());
            assertTrue(specializeFirst(k.multiply(b)).isZero());
            assertTrue("BSM1 maps ker into ker", specializeFirst(k.darkMatterEndomorphism()).isZero());
        }
        // (f) cross-check of the two carriers on Laurent polynomials sum_n q_n t^n: evaluation at 1 = sum of q_n
        RationalQuaternion q0 = new RationalQuaternion(r(1), r(0), r(2), r(0)), q1 = new RationalQuaternion(r(0), r(3), r(0), r(-1)), q2 = new RationalQuaternion(r(2), r(-1), r(1), r(1));
        RationalTwistedHahnSeries tt = t.multiply(t);
        RationalTwistedHahnSeries poly = RationalTwistedHahnSeries.fromQuaternion(q0)
                .add(RationalTwistedHahnSeries.fromQuaternion(q1).multiply(t)).add(RationalTwistedHahnSeries.fromQuaternion(q2).multiply(tt));
        Map<RationalTwistedLatticeAlgebra.Exponent, RationalQuaternion> lat = new TreeMap<>();
        lat.put(new RationalTwistedLatticeAlgebra.Exponent(0), q0);
        lat.put(new RationalTwistedLatticeAlgebra.Exponent(1), q1);
        lat.put(new RationalTwistedLatticeAlgebra.Exponent(2), q2);
        RationalTwistedLatticeAlgebra latPoly = new RationalTwistedLatticeAlgebra(1, lat);
        assertEquals(poly.projectChi5ToOneInDixon(), RationalDixonZ2.fromQuaternion(specializeFirst(latPoly).toQuaternion()));
    }


    /** Number of simplices of all degrees in a relative pair (absolute pairs: ordinary count). */
    private static int total(RelativeSimplicialComplex p) {
        int s = 0;
        for (int d = 0; d <= p.dimension(); d++) s += p.simplexCount(z(d));
        return s;
    }

    // ------------------------------------------------------------------ STEP 3: CDT cells, collapse, orientation lock
    @Test public void step3_cdt_cells_signed_collapse_subdivision_and_orientation_lock() {
        FiniteSimplicialComplex cell = new FiniteSimplicialComplex(Collections.singletonList(FiniteSet.of(0, 1, 2, 3, 4)));
        // (4,1) cell: vertices 0..3 at time t, vertex 4 at time t+1. (1,4) cell: vertex 0 at t, vertices 1..4 at t+1.
        FiniteSet<BigInteger> bottom41 = new FiniteSet<>(Arrays.asList(z(0), z(1), z(2), z(3)));
        FiniteSet<BigInteger> top14 = new FiniteSet<>(Arrays.asList(z(1), z(2), z(3), z(4)));
        int[] incidence = new int[5];
        for (int omitted = 0; omitted < 5; omitted++) {
            List<BigInteger> face = new ArrayList<>();
            for (int v = 0; v < 5; v++) if (v != omitted) face.add(z(v));
            FiniteSet<BigInteger> fs = new FiniteSet<>(face);
            SimplicialCollapse c = SimplicialCollapse.absolute(cell, fs);
            List<FiniteSet<BigInteger>> tets = c.source().simplexBasis(z(3));
            incidence[omitted] = c.chainHomotopyMatrix(z(3)).get(0, tets.indexOf(fs)).intValueExact();
            assertEquals("signed incidence is (-1)^position(omitted vertex)", (omitted % 2 == 0) ? 1 : -1, incidence[omitted]);
            assertEquals(total(c.source()) - 2, total(c.target())); // elementary: removes face and coface only
        }
        // Convention-dependence made explicit: spatial faces of both cells have incidence +1 in time-ordered labelling,
        // whereas time-like faces alternate. The incidence sign alone is therefore NOT the SM-D11 observable.
        assertEquals(+1, incidence[4]); // bottom of the (4,1) cell
        assertEquals(+1, incidence[0]); // top of the (1,4) cell
        assertNotNull(bottom41); assertNotNull(top14);

        // The SM-D11 observable is the orientation of the whole cell against a fixed foliation frame (x,y,z,t).
        // Rows are v_k - v_0, with a positively oriented spatial tetrahedron on each slice.
        RationalMatrix frame41 = new RationalMatrix(new Rational[][]{
                {r(1), r(0), r(0), r(0)}, {r(0), r(1), r(0), r(0)}, {r(0), r(0), r(1), r(0)}, {r(0), r(0), r(0), r(1)}});
        RationalMatrix frame14 = new RationalMatrix(new Rational[][]{
                {r(0), r(0), r(0), r(1)}, {r(1), r(0), r(0), r(1)}, {r(0), r(1), r(0), r(1)}, {r(0), r(0), r(1), r(1)}});
        int s41 = frame41.determinant().signum(), s14 = frame14.determinant().signum();
        assertEquals(+1, s41);
        assertEquals(-1, s14);
        // Abstract multiplicities (NOT data): the signed asymmetry is sign(N41*s41 + N14*s14) for any N41 > N14 >= 0.
        int simD11 = Long.signum(2L * s41 + 1L * s14);
        assertEquals(+1, simD11);

        // Signed subdivision of the spatial 3-sphere boundary(Delta^4): chain map, orientation cycle carried to orientation cycle.
        FiniteSimplicialComplex s3 = cell.skeleton(3);
        SimplicialSubdivision sd = SimplicialSubdivision.absolute(s3);
        IntegerMatrix d3o = sd.original().boundaryMatrix(z(3)), d3s = sd.subdivided().boundaryMatrix(z(3));
        IntegerMatrix sd3 = sd.chainMatrix(z(3)), sd2 = sd.chainMatrix(z(2));
        assertEquals("sd is a chain map", d3s.multiply(sd3), sd2.multiply(d3o));
        List<IntegerVector> fundamental = d3o.kernelBasis();
        assertEquals(1, fundamental.size());                            // H_3(S^3)=Z
        IntegerVector zOrig = fundamental.get(0), zSub = sd3.multiply(zOrig);
        assertTrue(d3s.multiply(zSub).entries().stream().allMatch(e -> e.signum() == 0));
        assertEquals(5 * 24, zSub.entries().stream().filter(e -> e.signum() != 0).count());      // 5 tetrahedra x 4! pieces
        assertTrue(zSub.entries().stream().allMatch(e -> e.abs().compareTo(BigInteger.ONE) <= 0)); // coherent +-1 orientation
        assertEquals(1, d3s.kernelBasis().size());
        AbelianGroupHomomorphism h3 = sd.homologyMap(z(3));
        assertTrue(h3.isIsomorphism());

        // Collapse of the full cell to a point: homology is preserved (all steps are elementary free-face collapses).
        SimplicialCollapseSequence seq = SimplicialCollapseSequence.reduceAbsolute(cell);
        assertEquals(z(15), seq.stepCount());
        assertEquals(1, total(seq.target()));

        // CCDRChiralFold: three projections of the single terminal-fold orientation epsilon_5 (CL9).
        CCDRChiralFold fold = new CCDRChiralFold(+1, +1, simD11, +1);
        assertEquals(+1, fold.chiralitySign());
        assertTrue(fold.isLeftChiral());
        assertTrue(fold.smD11Consistent() && fold.pA30Consistent() && fold.cl9Consistent());
        // flipping eps5 flips the chirality sign and breaks both locks unless simplex/CMB signs flip with it
        CCDRChiralFold flipped = new CCDRChiralFold(-1, +1, simD11, +1);
        assertEquals(-1, flipped.chiralitySign());
        assertFalse(flipped.isLeftChiral());
        assertFalse(flipped.cl9Consistent());
        assertTrue(new CCDRChiralFold(-1, +1, -1, -1).cl9Consistent());       // consistent but right-chiral: OP20 scope
        assertFalse(new CCDRChiralFold(+1, +1, +1, -1).cl9Consistent());      // P-A30 violated
        assertFalse(new CCDRChiralFold(+1, +1, -1, +1).cl9Consistent());      // SM-D11 violated
        assertThrows(IllegalArgumentException.class, () -> new CCDRChiralFold(0, 1, 1, 1));
        // candidate chirality projection onto the crossed-product group-parity blocks (not a minimal-ideal identification)
        RationalDixonZ2 w = RationalDixonZ2.fromQuaternion(new RationalQuaternion(r(1), r(2), r(0), r(3)))
                .add(RationalDixonZ2.fromOctonion(RationalOctonion.IL).multiply(RationalDixonZ2.U));
        assertEquals(w.evenPart(), fold.projectDixonParity(w));
        assertEquals(w.oddPart(), flipped.projectDixonParity(w));
        assertEquals(w, w.evenPart().add(w.oddPart()));
    }

    // ------------------------------------------------------------------ STEP 4: gauge algebras, Dixon, anomalies
    private static RationalMatrix comm(RationalMatrix a, RationalMatrix b) {
        return a.multiply(b).add(b.multiply(a).scale(r(-1)));
    }
    private static RationalVector flat(RationalMatrix m) {
        Rational[] v = new Rational[m.rows() * m.columns()];
        for (int i = 0; i < m.rows(); i++) for (int j = 0; j < m.columns(); j++) v[i * m.columns() + j] = m.get(i, j);
        return new RationalVector(v);
    }
    private static RationalMatrix columns(List<RationalMatrix> basis) {
        int dim = basis.get(0).rows() * basis.get(0).columns();
        Rational[][] a = new Rational[dim][basis.size()];
        for (int k = 0; k < basis.size(); k++) { RationalVector v = flat(basis.get(k)); for (int i = 0; i < dim; i++) a[i][k] = v.get(i); }
        return new RationalMatrix(a);
    }
    /** Exact structure constants f[a][b][c]: [B_a,B_b] = sum_c f[a][b][c] B_c; asserts closure. */
    private static Rational[][][] structureConstants(List<RationalMatrix> basis) {
        RationalMatrix a = columns(basis), pinv = a.pseudoinverse();
        int n = basis.size();
        Rational[][][] f = new Rational[n][n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            RationalVector target = flat(comm(basis.get(i), basis.get(j))), coordinates = pinv.multiply(target);
            assertEquals("Lie bracket closes on the span", target, a.multiply(coordinates));
            for (int k = 0; k < n; k++) f[i][j][k] = coordinates.get(k);
        }
        return f;
    }
    private static RationalMatrix killing(Rational[][][] f) {
        int n = f.length; Rational[][] k = new Rational[n][n];
        for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) {
            Rational s = Rational.ZERO;
            for (int c = 0; c < n; c++) for (int d = 0; d < n; d++) s = s.add(f[a][c][d].multiply(f[b][d][c]));
            k[a][b] = s;
        }
        return new RationalMatrix(k);
    }
    private static boolean negativeDefinite(RationalMatrix k) { // exact Sylvester test for -K
        for (int m = 1; m <= k.rows(); m++) {
            Rational[][] sub = new Rational[m][m];
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) sub[i][j] = k.get(i, j).negate();
            if (new RationalMatrix(sub).determinant().signum() <= 0) return false;
        }
        return true;
    }


    // ---- Dixon lift of an 8x8 octonion operator: acts on the O-index of all 64 Q(i)-coordinates
    private static RationalDixonZ2 lift(RationalMatrix d, RationalDixonZ2 x) {
        RationalComplex[] out = new RationalComplex[64];
        for (int blk = 0; blk < 2; blk++) for (int q = 0; q < 4; q++) for (int o = 0; o < 8; o++) {
            Rational re = Rational.ZERO, im = Rational.ZERO;
            for (int c = 0; c < 8; c++) {
                RationalComplex v = x.coordinate(blk * 32 + q * 8 + c);
                re = re.add(d.get(o, c).multiply(v.real)); im = im.add(d.get(o, c).multiply(v.imaginary));
            }
            out[blk * 32 + q * 8 + o] = new RationalComplex(re, im);
        }
        Rational[] v = new Rational[128];
        for (int n = 0; n < 64; n++) { v[2 * n] = out[n].real; v[2 * n + 1] = out[n].imaginary; }
        return RationalDixonZ2.fromVector(new RationalVector(v));
    }
    private static RationalDixonZ2 randomDixon(Rng g, int terms) {
        RationalDixonZ2 x = RationalDixonZ2.ZERO;
        for (int t = 0; t < terms; t++) {
            int n = g.next(64);
            Rational re = r(g.next(7) - 3), im = r(g.next(5) - 2);
            Rational[] v = new Rational[128]; Arrays.fill(v, Rational.ZERO); v[2 * n] = re; v[2 * n + 1] = im;
            x = x.add(RationalDixonZ2.fromVector(new RationalVector(v)));
        }
        return x;
    }
    private static boolean isDerivationOfDixon(RationalMatrix d, Rng g) {
        for (int trial = 0; trial < 6; trial++) {
            RationalDixonZ2 x = randomDixon(g, 5), y = randomDixon(g, 5);
            RationalDixonZ2 lhs = lift(d, x.multiply(y)), rhs = lift(d, x).multiply(y).add(x.multiply(lift(d, y)));
            if (!lhs.equals(rhs)) return false;
        }
        return true;
    }
    private static RationalMatrix alphaMatrix() {
        Rational[][] a = new Rational[8][8];
        for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) a[i][j] = (i == j) ? (i >= 4 ? r(-1) : r(1)) : Rational.ZERO;
        return new RationalMatrix(a);
    }

    /** Lift of a 4x4 quaternion-index operator to the Q-index of all 64 coordinates. */
    private static RationalDixonZ2 liftQ(RationalMatrix d, RationalDixonZ2 x) {
        Rational[] v = new Rational[128];
        for (int blk = 0; blk < 2; blk++) for (int q = 0; q < 4; q++) for (int o = 0; o < 8; o++) {
            Rational re = Rational.ZERO, im = Rational.ZERO;
            for (int c = 0; c < 4; c++) {
                RationalComplex w = x.coordinate(blk * 32 + c * 8 + o);
                re = re.add(d.get(q, c).multiply(w.real)); im = im.add(d.get(q, c).multiply(w.imaginary));
            }
            int n = blk * 32 + q * 8 + o; v[2 * n] = re; v[2 * n + 1] = im;
        }
        return RationalDixonZ2.fromVector(new RationalVector(v));
    }
    private static boolean isQDerivationOfDixon(RationalMatrix d, Rng g) {
        for (int trial = 0; trial < 6; trial++) {
            RationalDixonZ2 x = randomDixon(g, 5), y = randomDixon(g, 5);
            if (!liftQ(d, x.multiply(y)).equals(liftQ(d, x).multiply(y).add(x.multiply(liftQ(d, y))))) return false;
        }
        return true;
    }
    private static RationalMatrix combo(List<RationalMatrix> basis, RationalVector c) {
        RationalMatrix acc = basis.get(0).scale(c.get(0));
        for (int k = 1; k < basis.size(); k++) acc = acc.add(basis.get(k).scale(c.get(k)));
        return acc;
    }

    // ------------------------------------------------------------------ STEP 4 (executable)
    @Test public void step4_gauge_algebras_alpha_centralizer_and_weyl_anomalies() {
        List<RationalMatrix> su3 = RationalOctonionDerivations.su3StabilizerCandidates();
        List<RationalMatrix> su2 = RationalOctonionDerivations.quaternionAdjointGenerators();
        assertEquals(8, su3.size()); assertEquals(3, su2.size());

        // (a) su(3): derivations of O(Q) annihilating l = e4; closed, compact (exact Killing form negative definite).
        Rational[] lv = new Rational[8]; Arrays.fill(lv, Rational.ZERO); lv[4] = Rational.ONE;
        RationalVector l = new RationalVector(lv);
        RationalOctonion[] eb = new RationalOctonion[8];
        for (int i = 0; i < 8; i++) { Rational[] c = new Rational[8]; Arrays.fill(c, Rational.ZERO); c[i] = Rational.ONE; eb[i] = new RationalOctonion(c); }
        for (RationalMatrix d : su3) {
            assertEquals(new RationalVector(Rational.ZERO, Rational.ZERO, Rational.ZERO, Rational.ZERO, Rational.ZERO, Rational.ZERO, Rational.ZERO, Rational.ZERO), d.multiply(l));
            for (int a = 0; a < 8; a++) for (int b = 0; b < 8; b++) { // D(e_a e_b) = D(e_a) e_b + e_a D(e_b)
                RationalVector da = d.multiply(basisVec(a)), db = d.multiply(basisVec(b)), dab = d.multiply(vec(eb[a].multiply(eb[b])));
                RationalOctonion rhs = octo(da).multiply(eb[b]).add(eb[a].multiply(octo(db)));
                assertEquals(vec(rhs), dab);
            }
        }
        Rational[][][] f3 = structureConstants(su3);
        RationalMatrix k3 = killing(f3);
        assertTrue("su(3)-slice is compact: exact Killing form negative definite", negativeDefinite(k3));
        Rational[][][] f2 = structureConstants(su2);
        assertTrue("su(2) adjoint triple is compact", negativeDefinite(killing(f2)));

        // (b) The Cayley-Dickson parity alpha normalizes su(3) but its fixed subalgebra is 3-dimensional.
        RationalMatrix alpha = alphaMatrix(), A = columns(su3), pinv = A.pseudoinverse();
        int n = su3.size(); Rational[][] ad = new Rational[n][n];
        for (int j = 0; j < n; j++) {
            RationalMatrix conj = alpha.multiply(su3.get(j)).multiply(alpha);
            RationalVector c = pinv.multiply(flat(conj));
            assertEquals("alpha normalizes the su(3) slice", flat(conj), A.multiply(c));
            for (int i = 0; i < n; i++) ad[i][j] = c.get(i);
        }
        RationalMatrix Ad = new RationalMatrix(ad), I8 = RationalMatrix.identity(n);
        assertEquals(r(-2), Ad.trace());                     // = dim(+1) - dim(-1) = 3 - 5
        List<RationalVector> fixed = Ad.add(I8.scale(r(-1))).nullspace(), flipped = Ad.add(I8).nullspace();
        assertEquals(3, fixed.size()); assertEquals(5, flipped.size());
        Rng g = new Rng(20260101);
        List<RationalMatrix> fixedBasis = new ArrayList<>();
        for (RationalVector c : fixed) {
            RationalMatrix d = combo(su3, c); fixedBasis.add(d);
            assertEquals(d, alpha.multiply(d).multiply(alpha));
            assertTrue("alpha-commuting element lifts to a derivation of DixonZ2", isDerivationOfDixon(d, g));
        }
        for (RationalVector c : flipped) {
            RationalMatrix d = combo(su3, c);
            assertEquals(d.scale(r(-1)), alpha.multiply(d).multiply(alpha));
            assertFalse("alpha-anticommuting element is NOT a derivation of DixonZ2", isDerivationOfDixon(d, g));
        }
        assertTrue("alpha-fixed subalgebra is a compact 3-dim Lie algebra", negativeDefinite(killing(structureConstants(fixedBasis))));
        // The quaternionic su(2) acts on the H-factor and does lift.
        for (RationalMatrix d : su2) assertTrue(isQDerivationOfDixon(d, g));

        // (c) Exact 16-state Weyl generation and anomaly traces.
        assertEquals(16, StandardModelWeylGeneration.stateCount());
        assertEquals(Rational.ZERO, StandardModelWeylGeneration.hyperchargeTrace());
        assertEquals(Rational.ZERO, StandardModelWeylGeneration.cubicHyperchargeTrace());
        assertEquals(Rational.ZERO, StandardModelWeylGeneration.mixedSu2HyperchargeTrace());
        assertEquals(Rational.ZERO, StandardModelWeylGeneration.mixedSu3HyperchargeTrace());
        assertEquals(0, StandardModelWeylGeneration.cubicSu3AnomalyIndex());
        assertEquals(0, StandardModelWeylGeneration.su2DoubletParity());
        assertTrue(StandardModelWeylGeneration.anomalyFreeHyperchargeSignature());
        assertEquals(r(2), StandardModelWeylGeneration.t3SquaredTrace());
        assertEquals(r(16).divide(r(3)), StandardModelWeylGeneration.electricChargeSquaredTrace());
        assertEquals(r(3).divide(r(8)), StandardModelWeylGeneration.treeLevelGUTWeinbergRatio());
        for (StandardModelWeylGeneration.State s : StandardModelWeylGeneration.states())
            assertEquals(s.toString(), s.electricCharge(), s.t3().add(s.hypercharge()));
    }
    private static RationalVector basisVec(int i) { Rational[] c = new Rational[8]; Arrays.fill(c, Rational.ZERO); c[i] = Rational.ONE; return new RationalVector(c); }
    private static RationalVector vec(RationalOctonion o) { Rational[] c = new Rational[8]; for (int i = 0; i < 8; i++) c[i] = o.coordinate(i); return new RationalVector(c); }
    private static RationalOctonion octo(RationalVector v) { Rational[] c = new Rational[8]; for (int i = 0; i < 8; i++) c[i] = v.get(i); return new RationalOctonion(c); }

    // ------------------------------------------------------------------ STEP 5: Q(zeta12), Koide, Jarlskog
    private static BigDecimal bd(String v) { return new BigDecimal(v); }

    @Test public void step5_cyclotomic_closure_koide_parametrization_and_jarlskog_residuals() {
        final int P = 40;
        RationalCyclotomic z = RationalCyclotomic.zeta(12);
        assertEquals(new Polynomial(r(1), r(0), r(-1), r(0), r(1)), z.cyclotomicPolynomial());     // x^4 - x^2 + 1
        assertEquals(4, z.degree());
        RationalCyclotomic i12 = RationalCyclotomic.fromComplex(12, new RationalComplex(r(0), r(1)));
        assertEquals(i12, z.pow(3));                                                                // i = zeta^3
        assertEquals(RationalCyclotomic.fromRational(12, r(-1)), z.pow(6));
        assertEquals(RationalCyclotomic.one(12), z.pow(12));
        assertEquals(RationalCyclotomic.one(12), z.multiply(z.conjugate()));                        // exact norm one
        // sin(pi/6)=1/2, cos(pi/6)=sqrt(3)/2 from the exact element
        RationalCyclotomic.DecimalComplex e = z.evaluate(P);
        assertClose(bd("0.5"), e.imaginary, "1e-38");
        assertClose(sqrt(bd("3")).divide(bd("2"), MC), e.real, "1e-38");

        // Koide: for sqrt(m_k) = 1 + r cos(theta + 2 pi k/3), Q = 1/3 + r^2/6 (proof: sum of cosines vanishes).
        // theta = 0 and theta = pi/6 use only exact Q(zeta12) cosines.
        String sqrt2 = "1.4142135623730950488016887242096980785696718753769";
        String[][] cases = {{"0", "0.5"}, {"0", "1"}, {"0", sqrt2}, {"1", "0.5"}, {"1", "1"}};   // r*|cos| < 1 keeps every sqrt(m_k) > 0
        for (String[] cs : cases) {
            int turns = Integer.parseInt(cs[0]);
            {
                String rr = cs[1];
                BigDecimal rad = bd(rr);
                BigDecimal[] m = new BigDecimal[3];
                for (int k = 0; k < 3; k++) {
                    RationalCyclotomic ph = z.pow(turns + 4 * k);              // angle = turns*pi/6 + 2*pi*k/3
                    BigDecimal root = BigDecimal.ONE.add(rad.multiply(ph.evaluate(P + 10).real, MC), MC);
                    assertTrue("positive square roots required", root.signum() > 0);
                    m[k] = root.multiply(root, MC);
                }
                BigDecimal q = HighPrecisionAlgebraMetrics.koide(m[0], m[1], m[2], P);
                BigDecimal expected = bd("1").divide(bd("3"), MC).add(rad.multiply(rad, MC).divide(bd("6"), MC), MC);
                assertClose(expected, q, "1e-30");
            }
        }
        // Q = 2/3 requires r^2 = 2 (own remark, algebraic).
        assertEquals(r(2).divide(r(3)), r(1).divide(r(3)).add(r(2).divide(r(6))));

        // PDG-2025 lepton audit as in the verification report (inputs taken from the report, not from this test).
        BigDecimal me = bd("0.51099895000"), mmu = bd("105.6583755"), tau = bd("1776.93"), dtau = bd("0.09");
        BigDecimal two3 = bd("2").divide(bd("3"), new MathContext(P, RoundingMode.HALF_EVEN));
        BigDecimal qc = HighPrecisionAlgebraMetrics.koide(me, mmu, tau, P);
        assertClose(bd("0.6666644634145001750698620530280868826435"), qc, "1e-30");                 // report value
        BigDecimal lo = HighPrecisionAlgebraMetrics.koide(me, mmu, tau.subtract(dtau), P), hi = HighPrecisionAlgebraMetrics.koide(me, mmu, tau.add(dtau), P);
        assertTrue("tau 1-sigma interval brackets 2/3", lo.compareTo(two3) <= 0 && hi.compareTo(two3) >= 0);
        assertClose(bd("0.66666051147738557"), HighPrecisionAlgebraMetrics.koide(me, mmu, bd("1776.86"), P), "1e-15");

        // Jarlskog: linear in sin(delta); exact cyclotomic phase vs measured phase.
        BigDecimal s12 = bd("0.22501"), s23 = bd("0.04183"), s13 = bd("0.003732");
        BigDecimal c12 = sqrt(BigDecimal.ONE.subtract(s12.multiply(s12), MC)), c23 = sqrt(BigDecimal.ONE.subtract(s23.multiply(s23), MC)), c13 = sqrt(BigDecimal.ONE.subtract(s13.multiply(s13), MC));
        BigDecimal jz = HighPrecisionAlgebraMetrics.ckmJarlskog(s12, s23, s13, c12, c23, c13, z, P);
        BigDecimal jm = HighPrecisionAlgebraMetrics.ckmJarlskog(s12, s23, s13, c12, c23, c13, bd("1.147"), P);
        BigDecimal sinM = RationalCyclotomic.sinRadians(bd("1.147"), P);
        assertClose(bd("0.9115343723414106708707"), sinM, "1e-20");
        assertClose(bd("1.7097509558246345705944971779e-5"), jz, "1e-33");                          // report J(zeta12)
        assertClose(bd("3.1169935287554704725726246712e-5"), jm, "1e-33");                          // report J(measured delta)
        assertClose(jm.multiply(bd("0.5"), MC).divide(sinM, MC), jz, "1e-30");                      // J ∝ sin(delta)
        BigDecimal jErr = bd("0.13e-5");
        assertTrue("measured delta reproduces PDG J within band", jm.subtract(bd("3.12e-5")).abs().compareTo(jErr) < 0);
        assertTrue("pi/6 phase does not", jz.subtract(bd("3.12e-5")).abs().compareTo(jErr) > 0);
        assertEquals(-1, HighPrecisionAlgebraMetrics.ckmJarlskog(s12, s23, s13, c12, c23, c13, z.conjugate(), P).signum());
        // SM-D8 tension: nu ~ 5.08e-3 is nowhere near sin(delta_CP) ~ 0.91; 4 nu = 0.02032 also not.
        BigDecimal nu = bd("5.08e-3"), dnu = bd("1.2e-3");
        assertTrue(sinM.subtract(nu).abs().compareTo(dnu) > 0);
        assertTrue(sinM.subtract(nu.multiply(bd("4"))).abs().compareTo(dnu.multiply(bd("4"))) > 0);
        assertClose(bd("0.02032"), nu.multiply(bd("4")), "1e-12");

        // Exact rational-point phases in Q(i) c Q(zeta12): (1-t^2+2it)/(1+t^2) has norm exactly 1.
        Rational t = r(1).divide(r(400)), d = r(1).add(t.multiply(t));
        RationalCyclotomic tilt = RationalCyclotomic.fromComplex(12, new RationalComplex(r(1).subtract(t.multiply(t)).divide(d), r(2).multiply(t).divide(d)));
        assertEquals(RationalCyclotomic.one(12), tilt.multiply(tilt.conjugate()));
        BigDecimal sinTilt = tilt.evaluate(P).imaginary;
        assertClose(bd("2").multiply(bd("0.0025"), MC).divide(bd("1.00000625"), MC), sinTilt, "1e-30");
        // Orientation lock: left chirality keeps sin(delta), right chirality flips its sign.
        CCDRChiralFold left = new CCDRChiralFold(1, 1, 1, 1), right = new CCDRChiralFold(-1, 1, -1, -1);
        assertTrue(left.smD8MatchesNu(tilt, nu, dnu, P));
        assertFalse(right.smD8MatchesNu(tilt, nu, dnu, P));
        assertClose(sinTilt.negate(), right.smD8SignedSine(tilt, P), "1e-30");
    }

    // ------------------------------------------------------------------ STEP 3b/6: 5->4 parity fold, chirality, flow
    @Test public void step6_parity_fold_properties_chiral_lock_and_end_to_end_flow() {
        ConcreteMathematics math = new ConcreteMathematics();
        RationalQuaternion[] qb = {RationalQuaternion.ONE, RationalQuaternion.I, RationalQuaternion.J, RationalQuaternion.K};
        // rank is an observable: the fold needs rank >= 5 and ignores coordinates beyond the fifth.
        assertThrows(RuntimeException.class, () -> RationalTwistedLatticeAlgebra.fromQuaternion(4, RationalQuaternion.I).foldParityClassesToDixonBoundary());
        Rng g = new Rng(99);
        for (int rank : new int[]{5, 6, 8, 11}) {
            int[][] c = cocycle(rank);
            RationalTwistedLatticeAlgebra a = randomElement(rank, c, g, 6), b = randomElement(rank, c, g, 6);
            RationalDixonZ2 fa = a.foldParityClassesToDixonBoundary(), fb = b.foldParityClassesToDixonBoundary();
            assertEquals("additive", a.add(b).foldParityClassesToDixonBoundary(), fa.add(fb));
            // shifting any coordinate beyond the fifth by 1 changes nothing
            if (rank > 5) {
                Map<RationalTwistedLatticeAlgebra.Exponent, RationalQuaternion> shifted = new TreeMap<>();
                for (Map.Entry<RationalTwistedLatticeAlgebra.Exponent, RationalQuaternion> en : a.terms().entrySet()) {
                    List<BigInteger> co = new ArrayList<>();
                    for (int i = 0; i < rank; i++) co.add(en.getKey().coordinate(i).add(i >= 5 ? BigInteger.ONE : BigInteger.ZERO));
                    shifted.merge(new RationalTwistedLatticeAlgebra.Exponent(co), en.getValue(), RationalQuaternion::add);
                }
                assertEquals(fa, new RationalTwistedLatticeAlgebra(rank, c, shifted).foldParityClassesToDixonBoundary());
            }
            assertEquals(Collections.singletonList(fa.toString()),
                    math.flow(math.twistedLattice, Collections.singletonList(a)).<RationalDixonZ2>performAlgebraTransfer("parity-fold-to-dixon-boundary").collect());
        }
        // not multiplicative: X0 * X0 = X^(2e0) has parity class 0, but fold(X0)^2 = (e_2)^2 = -1 in the Dixon boundary
        RationalTwistedLatticeAlgebra x = x0(5, cocycle(5));
        assertNotEquals(x.multiply(x).foldParityClassesToDixonBoundary(), x.foldParityClassesToDixonBoundary().multiply(x.foldParityClassesToDixonBoundary()));
        // surjective: each of the 128 coordinates is hit by a single basis monomial
        for (int coord = 0; coord < 128; coord++) {
            long[] ex = new long[5];
            for (int ax = 0; ax < 5; ax++) if (((coord / 4) & (1 << ax)) != 0) ex[ax] = 1;
            Rational[] want = new Rational[128]; Arrays.fill(want, Rational.ZERO); want[coord] = Rational.ONE;
            assertEquals(new RationalVector(want), RationalTwistedLatticeAlgebra.basis(5, new RationalTwistedLatticeAlgebra.Exponent(ex), qb[coord % 4]).foldParityClassesToDixonBoundary().toVector());
        }
        // At rank 5 the fold kernel (even shifts of exponents) is a two-sided ideal: sigma and tau depend only on exponent parity.
        {
            int[][] c5 = cocycle(5);
            RationalTwistedLatticeAlgebra a0 = randomElement(5, c5, g, 4), b0 = randomElement(5, c5, g, 4);
            long[] two = new long[5]; two[1] = 2; two[3] = 2;
            RationalTwistedLatticeAlgebra shiftOp = RationalTwistedLatticeAlgebra.basis(5, c5, new RationalTwistedLatticeAlgebra.Exponent(two), RationalQuaternion.ONE);
            RationalTwistedLatticeAlgebra kernelElement = a0.add(a0.multiply(shiftOp).negate());  // a0 - a0*X^(2e1+2e3): same classes, opposite sign
            assertTrue(kernelElement.foldParityClassesToDixonBoundary().isZero());
            assertTrue(kernelElement.multiply(b0).foldParityClassesToDixonBoundary().isZero());
            assertTrue(b0.multiply(kernelElement).foldParityClassesToDixonBoundary().isZero());
        }
        // At rank > 5 the fold ignores coordinates that the cocycle still couples, so its kernel is not an ideal there.
        {
            int[][] c7 = cocycle(7);                                     // couples coordinates (1,2), (3,4), (5,6)
            long[] e5 = new long[7]; e5[5] = 1; long[] e6 = new long[7]; e6[6] = 1;
            RationalTwistedLatticeAlgebra one = RationalTwistedLatticeAlgebra.fromQuaternion(7, c7, RationalQuaternion.ONE);
            RationalTwistedLatticeAlgebra x5 = RationalTwistedLatticeAlgebra.basis(7, c7, new RationalTwistedLatticeAlgebra.Exponent(e5), RationalQuaternion.ONE);
            RationalTwistedLatticeAlgebra x6 = RationalTwistedLatticeAlgebra.basis(7, c7, new RationalTwistedLatticeAlgebra.Exponent(e6), RationalQuaternion.ONE);
            RationalTwistedLatticeAlgebra k = one.add(x5.negate());
            assertTrue(k.foldParityClassesToDixonBoundary().isZero());
            assertFalse(k.multiply(x6).foldParityClassesToDixonBoundary().isZero());
        }

        // Chirality lock: the product eps5*arrow is what is predicted; the factors are not individually identified (OP20).
        int[][] signs = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        int leftCount = 0;
        for (int[] sg : signs) {
            CCDRChiralFold f = new CCDRChiralFold(sg[0], sg[1], sg[0] * sg[1], sg[0] * sg[1]);
            assertEquals(sg[0] * sg[1], f.chiralitySign()); assertTrue(f.cl9Consistent());
            if (f.isLeftChiral()) leftCount++;
            RationalDixonZ2 w = RationalDixonZ2.fromQuaternion(RationalQuaternion.I).add(RationalDixonZ2.U).add(RationalDixonZ2.basis(1, 4));
            assertEquals(f.isLeftChiral() ? w.evenPart() : w.oddPart(), f.projectDixonParity(w));
            assertFalse(new CCDRChiralFold(sg[0], sg[1], -sg[0] * sg[1], sg[0] * sg[1]).cl9Consistent());     // wrong simplex sign breaks SM-D11
            assertFalse(new CCDRChiralFold(sg[0], sg[1], sg[0] * sg[1], -sg[0] * sg[1]).cl9Consistent());     // wrong CMB sign breaks P-A30
        }
        assertEquals(2, leftCount);                    // two of four (eps5, arrow) assignments give each chirality: OP20
        assertTrue(new CCDRChiralFold(1, -1, -1, -1).isRightChiral());   // right chirality in the repo convention
        assertTrue(new CCDRChiralFold(-1, 1, -1, -1).isRightChiral());
    }
}

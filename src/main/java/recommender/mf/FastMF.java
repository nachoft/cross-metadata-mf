package recommender.mf;

import data.PreferenceData;
import java.util.Locale;
import java.util.Set;
import java.util.function.IntFunction;
import java.util.function.ToIntFunction;
import java.util.stream.IntStream;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import util.MatrixUtils;

/**
 * Implementation of ALS-based matrix factorization for ranking that uses a fast
 * approximation to the ridge regularizer:
 * <p>
 * <code>
 * Pilászy, I., Zibriczky, D., Tikk, D.: Fast als-based matrix factorization for
 * explicit and implicit feedback datasets. RecSys 2010
 * </code>
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class FastMF extends ImplicitMF {

    /**
     * Create a new instance of the Fast ALS iMF recommender using the provided
     * training data and the default hyper-parameters, i.e. <code>k = 10 and
     * alpha = 1</code>.
     *
     * @param train Training preference data.
     */
    public FastMF(PreferenceData train) {
        super(train);
    }

    /**
     * Compute the eigenvector matrix G for the synthetic negative examples.
     *
     * @param Q User/item factor matrix with the examples.
     * @param lambda Regularization parameter.
     * @return The G eigenvector matrix.
     */
    protected float[][] computeG(float[][] Q, float lambda) {
        // Compute A0
        double[][] A = new double[numFactors][numFactors];
        MatrixUtils.transposeTimes(Q, A);
        for (int k = 0; k < numFactors; k++) {
            A[k][k] += lambda;
        }

        // Obtain G from the eigenvalue decomposition of A
        RealMatrix M = new Array2DRowRealMatrix(A, false);
        EigenDecomposition eig = new EigenDecomposition(M);
        RealMatrix V = eig.getV();

        // Use columns of V to obtain rows of G instead of G transposed, as they
        // will be easier to access later
        float[][] G = new float[numFactors][numFactors];
        for (int k = 0; k < numFactors; k++) {
            double a = Math.sqrt(eig.getRealEigenvalue(k));
            for (int i = 0; i < numFactors; i++) {
                G[k][i] = (float) (a * V.getEntry(i, k));
            }
        }

        return G;
    }

    // P is the set of parameters to optimize, and Q those that are fixed
    @Override
    protected <Q> void leastSquares(float[][] P, float[][] Q, IntFunction<Set<Q>> prefs, ToIntFunction<Q> id) {
        // Compute G (not Gt, as rows are easier)
        float[][] G = computeG(Q, lambda);
        // Optimize for each user/item
        IntStream.range(0, P.length)
                .parallel()
                .forEach(u -> minimize(prefs.apply(u), id, P[u], Q, G));
    }

    @Override
    protected <Q> void minimize(Set<Q> prefs, ToIntFunction<Q> id, float[] w, float[][] Q, float[][] G) {
        int K = numFactors;
        int N = prefs.size();

        float[][] x = new float[K + N][K];
        float[] y = new float[K + N];
        float[] c = new float[K + N];

        // First K synthetic negative implicit feedback examples
        for (int k = 0; k < K; k++) {
            x[k] = G[k];
            y[k] = 0;
            c[k] = 1;
        }

        // Merge negative implicit feedback cancelation examples and
        // aggregation of positive implicit feedbacks
        int j = K;
        for (Q item : prefs) {
            int i = id.applyAsInt(item);
            x[j] = Q[i];
            // Note that in our implementation we only deal with binary feedback
            y[j] = (1 + alpha) / alpha;
            c[j] = alpha;
            j++;
        }

        // Perform a single cycle of RR1
        solveRR1(1, w, x, y, c);
    }

    protected void solveRR1(int L, float[] w, float[][] x, float[] y, float[] c) {
        int K = x[0].length;
        int N = x.length;

        // Compute all the errors
        float[] e = new float[N];
        for (int i = 0; i < N; i++) {
            float pred = 0;
            for (int k = 0; k < K; k++) {
                pred += w[k] * x[i][k];
            }
            e[i] = y[i] - pred;
        }

        // Cycle RR1
        for (int l = 0; l < L; l++) {
            // One cycle
            for (int k = 0; k < K; k++) {
                // New temporary error
                for (int i = 0; i < N; i++) {
                    e[i] += w[k] * x[i][k];
                }

                float a = 0;
                float d = 0;
                for (int i = 0; i < N; i++) {
                    a += c[i] * x[i][k] * x[i][k];
                    d += c[i] * x[i][k] * e[i];
                }
                w[k] = d / (lambda + a);

                // Update error
                for (int i = 0; i < N; i++) {
                    e[i] -= w[k] * x[i][k];
                }
            }
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ENGLISH, "FastALS_k=%d_l=%s_c=%s_n=%d",
                numFactors, lambda, alpha, numIterations);
    }

}

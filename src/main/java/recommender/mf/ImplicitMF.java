package recommender.mf;

import data.PreferenceData;
import java.util.Locale;
import java.util.Set;
import java.util.function.IntFunction;
import java.util.function.ToIntFunction;
import java.util.stream.IntStream;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import util.MatrixUtils;

/**
 * Implementation of the matrix factorization algorithm for implicit feedback as
 * proposed in the following paper:
 * <p>
 * <code>Hu, Y., Koren, Y., Volinsky, C.: Collaborative Filtering for Implicit
 * Feedback Datasets. ICDM 2008</code>
 * <p>
 * <strong>NOTE: </strong>this implementation only supports unary/binary
 * feedback, and frequencies are not considered. Also, the confidence value of
 * each rating is assumed to be as in the paper:
 * <p>
 * <code>c(u,i) = 1 + alpha * r(u,i)</code>
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class ImplicitMF extends MFRecommender {

    // Confidence parameter for unobserved interactions
    protected float alpha;

    /**
     * Create a new instance of the matrix factorization algorithm for positive-
     * only feedback using the given training data.
     * <p>
     * The default value of the confidence parameter is <code>alpha = 1</code>
     *
     * @param train training dataset
     */
    public ImplicitMF(PreferenceData train) {
        super(train);
        debug = false;
        // Default in MML
        alpha = 1;
    }

    /**
     * Returns the confidence parameter over the non-observed interactions.
     *
     * @return The current value of the confidence parameter.
     */
    public float getAlpha() {
        return alpha;
    }

    /**
     * Updates the value of the confidence parameter.
     *
     * @param alpha new value for the confidence parameter
     */
    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }

    /**
     * Computes the loss function over the training set for the current state of
     * the model.
     * <p>
     * Note that this method iterates over all possible (user, item) pairs and
     * is likely to take some time.
     *
     * @return The value of the loss function over the training set for the
     * current state of the model.
     */
    public float computeLoss() {
        // Loss function is defined over all (user, item) pairs
        double loss = train.users().stream()
                .parallel()
                .mapToDouble(user -> {
                    float userLoss = 0;
                    for (String item : train.items()) {
                        float p = train.existsPreference(user, item) ? 1f : 0;
                        float c = 1 + alpha * p;

                        float err = p - predictScore(user, item);
                        userLoss += c * err * err;
                    }
                    return userLoss;
                })
                .sum();

        // Regularization of latent factors
        if (lambda > 0) {
            float userReg = MatrixUtils.norm2(userFactors);
            float itemReg = MatrixUtils.norm2(itemFactors);
            loss += lambda * (userReg + itemReg);
        }

        return (float) loss;
    }

    /**
     * Trains this implicit MF algorithm using ALS.
     */
    @Override
    public void train() {
        init();
        // Perform ALS for a fixed number of iterations
        for (int iter = 0; iter < numIterations; iter++) {
            long tic = System.currentTimeMillis();

            // P step: fix item factors, optimize user factors
            userLeastSquares();
            // Q step: fix user factors, optimize item factors
            itemLeastSquares();

            float time = (System.currentTimeMillis() - tic) / 1000f;
            if (debug) {
                System.err.println("\t# " + (iter + 1) + "\tTime = " + time + "\tLoss = " + computeLoss());
            } else {
                System.err.println("\t# " + (iter + 1) + "\tTime = " + time);
            }
        }
    }

    protected void userLeastSquares() {
        leastSquares(userFactors, itemFactors, u -> train.userItems(train.user(u)), train::itemId);
    }

    protected void itemLeastSquares() {
        leastSquares(itemFactors, userFactors, i -> train.itemUsers(train.item(i)), train::userId);
    }

    protected <Q> void leastSquares(float[][] P, float[][] Q, IntFunction<Set<Q>> prefs, ToIntFunction<Q> qId) {
        // Notation: optimize P, fixed is Q
        // Compute Q'Q
        float[][] QtQ = MatrixUtils.transposeTimes(Q);
        // Optimize each P
        IntStream.range(0, P.length)
                .parallel()
                .forEach(p -> minimize(prefs.apply(p), qId, P[p], Q, QtQ));
    }

    protected <Q> void minimize(Set<Q> data, ToIntFunction<Q> qId, float[] w, float[][] Q, float[][] QtQ) {
        // Compute Q'CuQ + reg*I = Q'Q + Q'(Cu - I)Q + reg*I
        double[][] QtCQ = new double[numFactors][numFactors];
        for (int k1 = 0; k1 < numFactors; k1++) {
            for (int k2 = k1; k2 < numFactors; k2++) {
                float s = 0;
                // If Rui = 0, then Cuii - 1 = 0
                for (Q q : data) {
                    int i = qId.applyAsInt(q);
                    s += Q[i][k2] * Q[i][k1];
                }
                QtCQ[k1][k2] = QtQ[k1][k2] + s * alpha;
                QtCQ[k2][k1] = QtQ[k2][k1] + s * alpha;

                // Add regularization
                if (k1 == k2) {
                    QtCQ[k1][k2] += lambda;
                }
            }
        }

        // Compute Q'Cp
        // Size: k x |I| · |I| x |I| · |I| x 1 = k x 1
        float[] QCp = new float[numFactors];
        for (int k = 0; k < numFactors; k++) {
            float s = 0;
            for (Q q : data) {
                int i = qId.applyAsInt(q);
                s += Q[i][k];
            }
            QCp[k] = s * (1 + alpha);
        }

        // Compute the inverse of the matrix through LU decomposition
        RealMatrix M = new Array2DRowRealMatrix(QtCQ, false);
        LUDecomposition lu = new LUDecomposition(M);
        RealMatrix Minv = lu.getSolver().getInverse();

        // Update the user vector
        // w = Minv * Q'Cp
        for (int k1 = 0; k1 < numFactors; k1++) {
            float s = 0;
            for (int k2 = 0; k2 < numFactors; k2++) {
                s += Minv.getEntry(k1, k2) * QCp[k2];
            }
            w[k1] = (float) s;
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ENGLISH, "iMF_k=%d_l=%s_c=%s_n=%d",
                numFactors, lambda, alpha, numIterations);
    }

}

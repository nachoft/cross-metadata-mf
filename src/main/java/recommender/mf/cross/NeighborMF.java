package recommender.mf.cross;

import recommender.mf.*;
import data.PreferenceData;
import data.ScoredItem;
import java.util.HashSet;
import java.util.Locale;
import java.util.Queue;
import java.util.Set;
import similarity.ItemNeighborhoods;
import util.MatrixUtils;

/**
 * Cross-domain MF recommender with neighbor distance regularization based on
 * fast ALS.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class NeighborMF extends FastMF {

    // Item neighborhoods
    private final ItemNeighborhoods neighborhoods;
    // Sets of source and target domain items
    private final Set<String> sourceItems;
    private final Set<String> targetItems;
    // Regularization for cross-domain item factors
    private float lambdaCross;

    /**
     * Create a new instance of the Fast ALS cross-domain neighbor distance MF.
     *
     * @param train Training preference data.
     * @param neighborhoods Cross-domain item neighborhoods.
     * @param targetItems Items of the target domain. Source items are assumed
     * to be the set of training items that are not target items (set
     * difference).
     */
    public NeighborMF(PreferenceData train, ItemNeighborhoods neighborhoods, Set<String> targetItems) {
        super(train);
        this.neighborhoods = neighborhoods;
        this.sourceItems = new HashSet<>(train.items());
        this.sourceItems.removeAll(targetItems);
        this.targetItems = targetItems;
        this.lambdaCross = 0.015f;
    }

    @Override
    public float computeLoss() {
        double loss = super.computeLoss();

        if (lambdaCross > 0) {
            double simReg = targetItems.stream()
                    .parallel()
                    .mapToDouble(tgtItem -> {
                        float itemReg = 0;
                        int i = train.itemId(tgtItem);

                        Queue<ScoredItem> neighs = neighborhoods.neighbors(tgtItem);
                        if (neighs != null) {
                            for (ScoredItem ni : neighs) {
                                String neigh = ni.getItem();
                                int j = train.itemId(neigh);
                                float s = ni.getScore();
                                itemReg += s * MatrixUtils.distance2(itemFactors[i], itemFactors[j]);
                            }
                        }

                        return itemReg;
                    })
                    .sum();
            loss += lambdaCross * simReg;
        }

        return (float) loss;
    }

    @Override
    protected void itemLeastSquares() {
        // Compute G as in FastALS
        float[][] G = computeG(userFactors, lambda);

        // Update first source item factors
        sourceItems.stream()
                .parallel()
                .forEach(i -> updateSourceItem(i, train.itemUsers(i), G));

        // Then update target item factors
        targetItems.stream()
                .parallel()
                .forEach(j -> updateTargetItem(j, train.itemUsers(j), G));
    }

    protected void updateTargetItem(String item, Set<String> itemUsers, float[][] G) {
        int K = numFactors;
        int N = itemUsers.size();

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
        int n = K;
        for (String user : itemUsers) {
            int u = train.userId(user);
            x[n] = userFactors[u];
            // Note that in our implementation we only deal with binary feedback
            y[n] = (1 + alpha) / alpha;
            c[n] = alpha;
            n++;
        }

        // Compute the source neighbors contribution
        float sum = 0;
        float[] centroid = new float[numFactors];
        Queue<ScoredItem> neighs = neighborhoods.neighbors(item);
        if (neighs != null) {
            // Could be null if e.g. all sims are NaN
            for (ScoredItem neigh : neighs) {
                int j = train.itemId(neigh.getItem());
                float s = neigh.getScore();
                sum += s;
                MatrixUtils.add(centroid, itemFactors[j], s);
            }
        }

        // Perform a single cycle of RR1
        int i = train.itemId(item);
        solveExtendedRR1(1, itemFactors[i], x, y, c, centroid, sum);
    }

    protected void updateSourceItem(String item, Set<String> itemUsers, float[][] G) {
        int K = numFactors;
        int N = itemUsers.size();
        int j = train.itemId(item);

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
        int n = K;
        for (String user : itemUsers) {
            int u = train.userId(user);
            x[n] = userFactors[u];
            // Note that in our implementation we only deal with binary feedback
            y[n] = (1 + alpha) / alpha;
            c[n] = alpha;
            n++;
        }

        // Compute cross terms
        float sum = 0;
        float[] aux = new float[numFactors];
        Set<ScoredItem> invNeighs = neighborhoods.invNeighbors(item);
        if (invNeighs != null) {
            for (ScoredItem neigh : invNeighs) {
                int i = train.itemId(neigh.getItem());
                float s = neigh.getScore();
                sum += s;
                MatrixUtils.add(aux, itemFactors[i], s);
            }
        }

        // Perform a single cycle of RR1
        solveExtendedRR1(1, itemFactors[j], x, y, c, aux, sum);
    }

    protected void solveExtendedRR1(int L, float[] w, float[][] x, float[] y, float[] c, float[] num, float den) {
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
                w[k] = (d + lambdaCross * num[k]) / (a + lambda + lambdaCross * den);

                // Update error
                for (int i = 0; i < N; i++) {
                    e[i] -= w[k] * x[i][k];
                }
            }
        }
    }

    /**
     * @return Cross-domain regularization.
     */
    public float getLambdaCross() {
        return lambdaCross;
    }

    /**
     * Change the regularization for cross-domain regularization.
     *
     * @param lambdaCross New value for the cross-domain regularization.
     */
    public void setLambdaCross(float lambdaCross) {
        this.lambdaCross = lambdaCross;
    }

    @Override
    public String toString() {
        return String.format(Locale.ENGLISH, "NeighborMF_k=%d_l=%s_lc=%s_c=%s_n=%d",
                numFactors, lambda, lambdaCross, alpha, numIterations);
    }

}

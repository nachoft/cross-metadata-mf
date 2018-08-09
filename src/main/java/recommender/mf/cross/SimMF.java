package recommender.mf.cross;

import data.PreferenceData;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import recommender.mf.FastMF;
import similarity.ISimilarity;
import util.MatrixUtils;

/**
 * Implementation of the cross-domain item similarity iMF recommender that uses
 * the RR1 variant proposed by Pilászy et al. (RecSys 2010) to speed up ALS.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class SimMF extends FastMF {

    // Strategy to compute cross-domain item similarities
    private final ISimilarity sim;
    // Sets of source and target domain items
    private final Set<String> sourceItems;
    private final Set<String> targetItems;
    // Regularization for cross-domain item factors
    // Note: if set to 0 we recover a bit less efficient iMF
    private float lambdaCross;

    /**
     * Create a new instance of the Fast ALS cross-domain similarity-based iMF
     * recommender.
     *
     * @param train Training preference data.
     * @param sim Cross-domain item similarity strategy.
     * @param targetItems Items of the target domain. Source items are assumed
     * to be the set of training items that are not target items (set
     * difference).
     */
    public SimMF(PreferenceData train, ISimilarity sim, Set<String> targetItems) {
        super(train);
        this.sim = sim;
        this.sourceItems = new HashSet<>(train.items());
        this.sourceItems.removeAll(targetItems);
        this.targetItems = targetItems;
        this.lambdaCross = 0.015f;
    }

    @Override
    public float computeLoss() {
        // Loss is the same as in normal iMF + additional regularization
        double loss = super.computeLoss();

        // Similarity regularization
        if (lambdaCross > 0) {
            double simReg = sourceItems.stream()
                    .parallel()
                    .mapToDouble(srcItem -> {
                        float itemReg = 0;
                        int i = train.itemId(srcItem);

                        for (String tgtItem : targetItems) {
                            int j = train.itemId(tgtItem);
                            float s = sim.compute(srcItem, tgtItem);
                            float prod = MatrixUtils.dotProduct(itemFactors[i], itemFactors[j]);
                            itemReg += (s - prod) * (s - prod);
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
        train.items().stream()
                .filter(sourceItems::contains)
                .parallel()
                .forEach(i -> minimizeItem(i, train.itemUsers(i), targetItems, G));

        // Then update target item factors
        train.items().stream()
                .filter(targetItems::contains)
                .parallel()
                .forEach(j -> minimizeItem(j, train.itemUsers(j), sourceItems, G));
    }

    protected void minimizeItem(String item, Set<String> itemUsers, Set<String> otherItems, float[][] G) {
        int K = numFactors;
        int N = itemUsers.size() + otherItems.size();

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

        // Cross-domain regularization examples
        for (String otherItem : otherItems) {
            int j = train.itemId(otherItem);
            x[n] = itemFactors[j];
            y[n] = sim.compute(item, otherItem);
            c[n] = lambdaCross;
            n++;
        }

        // Perform a single cycle of RR1
        int i = train.itemId(item);
        solveRR1(1, itemFactors[i], x, y, c);
    }

    /**
     * @return Regularization for cross-domain item factor similarities.
     */
    public float getLambdaCross() {
        return lambdaCross;
    }

    /**
     * Change the regularization for cross-domain item factor similarities.
     *
     * @param lambdaCross New value for the cross-domain item factor
     * regularization.
     */
    public void setLambdaCross(float lambdaCross) {
        this.lambdaCross = lambdaCross;
    }

    @Override
    public String toString() {
        return String.format(Locale.ENGLISH, "SimMF_k=%d_l=%s_lc=%s_c=%s_n=%d",
                numFactors, lambda, lambdaCross, alpha, numIterations);
    }

}

package recommender.mf;

import data.PreferenceData;
import recommender.AbstractPointwiseRecommender;
import util.MatrixUtils;

/**
 * Abstract class for MF-based algorithms.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public abstract class MFRecommender extends AbstractPointwiseRecommender {

    // Latent factors, row-wise
    protected float[][] userFactors;
    protected float[][] itemFactors;
    // Number of latent factors
    protected int numFactors;
    // Learning parameters
    protected int numIterations;
    protected float lambda;
    // Show learning information
    protected boolean debug;
    // Parameters for random initialization
    protected static final float INIT_MEAN = 0;
    protected static final float INIT_STD = 0.1f;

    public MFRecommender(PreferenceData train) {
        super(train);
        this.debug = false;
        // Default parameters, taken from MML
        this.numFactors = 10;
        this.numIterations = 15;
        this.lambda = 0.015f;
    }

    protected void init() {
        // Create matrices U x k and I x k to hold latent factors
        userFactors = new float[train.maxUserID() + 1][numFactors];
        itemFactors = new float[train.maxItemID() + 1][numFactors];
        // Initialize factors randomly
        MatrixUtils.setRandGaussian(userFactors, INIT_MEAN, INIT_STD);
        MatrixUtils.setRandGaussian(itemFactors, INIT_MEAN, INIT_STD);
    }

    public abstract void train();

    @Override
    public float predictScore(String user, String item) {
        // If the user or the item is unknown, we cannot compute a prediction
        if (!train.users().contains(user) || !train.items().contains(item)) {
            return Float.NaN;
        }

        int u = train.userId(user);
        int i = train.itemId(item);
        return MatrixUtils.dotProduct(userFactors[u], itemFactors[i]);
    }

    public int getNumFactors() {
        return numFactors;
    }

    public void setNumFactors(int numFactors) {
        this.numFactors = numFactors;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public void setNumIterations(int numIterations) {
        this.numIterations = numIterations;
    }

    public float getLambda() {
        return lambda;
    }

    public void setLambda(float lambda) {
        this.lambda = lambda;
    }

    public boolean isDebug() {
        return debug;
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

}

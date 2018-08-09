package recommender.knn;

import data.PreferenceData;
import java.util.Set;
import recommender.AbstractPointwiseRecommender;
import similarity.ISimilarity;

/**
 * Implementation of the Item KNN recommender for positive-only feedback.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class ItemKnn extends AbstractPointwiseRecommender {

    private final ISimilarity sim;

    /**
     * Creates a new Item kNN recommender using the given similarity function.
     *
     * @param train Training preference data.
     * @param similarity Item similarity function.
     */
    public ItemKnn(PreferenceData train, ISimilarity similarity) {
        super(train);
        this.sim = similarity;
    }

    @Override
    public float predictScore(String user, String item) {
        Set<String> userItems = train.userItems(user);
        // We cannot compute a prediction for users without items (new to the
        // system)
        if (userItems == null) {
            return Float.NaN;
        }

        double score = userItems.stream()
                // ignore target item
                .filter(userItem -> !item.equals(userItem))
                // compute similarities
                .mapToDouble(j -> sim.compute(item, j))
                // discard NaNs
                .filter(s -> !Double.isNaN(s))
                .sum();

        return (float) score;
    }

    @Override
    public String toString() {
        return "ItemKNN_" + sim;
    }

}

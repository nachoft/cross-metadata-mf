package recommender;

import data.ScoredItem;
import java.util.List;
import java.util.Set;

/**
 * Interface for all recommender algorithms.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public interface IRecommender {

    /**
     * Computes a list of recommendations for the given user from all the
     * training items.
     *
     * @param user Target user for which the recommendations are computed.
     * @param howMany Maximum size of the recommendation list. The number of
     * recommended items can be smaller than this value if the algorithm is not
     * able to compute as many predictions or if the set of candidate items is
     * not big enough.
     * @return The list of recommended {@link ScoredItem scored items}, sorted
     * by decreasing preference score.
     */
    List<ScoredItem> recommend(String user, int howMany);

    /**
     * Computes a list of recommendedations for a user from the given set of
     * candidate items.
     *
     * @param user Target user for which the recommendations are computed.
     * @param howMany Maximum size of the recommendation list. The number of
     * recommended items can be smaller than this value if the algorithm is not
     * able to compute as many predictions or if the set of candidate items is
     * not big enough.
     * @param candidateItems Set of possible items to be recommended.
     * @return The list of recommended {@link ScoredItem scored items}, sorted
     * by decreasing preference score.
     */
    List<ScoredItem> recommend(String user, int howMany, Set<String> candidateItems);
}

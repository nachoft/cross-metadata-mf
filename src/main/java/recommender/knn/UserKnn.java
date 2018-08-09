package recommender.knn;

import data.PreferenceData;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import java.util.PriorityQueue;
import java.util.Queue;
import recommender.AbstractPointwiseRecommender;
import similarity.ISimilarity;

/**
 * User-based nearest neighbors recommender for binary feedback.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class UserKnn extends AbstractPointwiseRecommender {

    private final int numNeighbors;
    private final TIntObjectMap<Queue<SimilarUser>> neighborhoods;
    private final ISimilarity sim;

    /**
     * Creates a new User KNN recommender using the given similarity function
     * and number of neighbors.
     *
     * @param train Training preference data.
     * @param sim User similarity function.
     * @param neighbors Number of neighbors to use for predictions.
     */
    public UserKnn(PreferenceData train, ISimilarity sim, int neighbors) {
        super(train);
        this.sim = sim;
        neighborhoods = new TIntObjectHashMap<>();
        numNeighbors = neighbors;
    }

    private void computeUserNeighborhood(int u) {
        String user = train.user(u);

        Queue<SimilarUser> neighbors = new PriorityQueue<>(numNeighbors);
        for (String otherUser : train.users()) {
            // Ignore target user
            if (u == train.userId(otherUser)) {
                continue;
            }

            float s = sim.compute(user, otherUser);

            if (neighbors.size() < numNeighbors) {
                neighbors.offer(new SimilarUser(otherUser, s));
            } else if (neighbors.peek().sim < s) {
                neighbors.poll();
                neighbors.offer(new SimilarUser(otherUser, s));
            }
        }

        neighborhoods.put(u, neighbors);
    }

    /**
     * @return The number of neighbors used by this User KNN recommender.
     */
    public int getNumNeighbors() {
        return numNeighbors;
    }

    @Override
    public float predictScore(String user, String item) {
        // This algorithm is unable to compute predictions for unknown users
        if (!train.containsUser(user)) {
            return Float.NaN;
        }

        int u = train.userId(user);

        // Compute neighborhoods on demand
        if (!neighborhoods.containsKey(u)) {
            computeUserNeighborhood(u);
        }
        Queue<SimilarUser> neighbors = neighborhoods.get(u);

        float score = 0;
        boolean foundNeighbor = false;

        // Order of the neighbors is not important here
        for (SimilarUser neighbor : neighbors) {
            if (train.existsPreference(neighbor.user, item)) {
                score += neighbor.sim;
                foundNeighbor = true;
            }
        }

        return foundNeighbor ? score : Float.NaN;
    }

    @Override
    public String toString() {
        return "UserKNN_" + numNeighbors + "_" + sim;
    }

    // Class that stores (user,similarity) pairs for neighborhoods
    private class SimilarUser implements Comparable<SimilarUser> {

        String user;
        float sim;

        public SimilarUser(String user, float sim) {
            this.user = user;
            this.sim = sim;
        }

        @Override
        public int compareTo(SimilarUser o) {
            return Float.compare(this.sim, o.sim);
        }
    }
}

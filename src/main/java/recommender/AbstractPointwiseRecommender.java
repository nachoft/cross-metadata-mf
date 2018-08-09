package recommender;

import data.PreferenceData;
import data.ScoredItem;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;

/**
 * Abstract recommender implementation that ranks items by individual score,
 * that is, without looking at the rest of the list.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public abstract class AbstractPointwiseRecommender implements IRecommender {

    /**
     * Training dataset
     */
    protected final PreferenceData train;

    public AbstractPointwiseRecommender(PreferenceData trainData) {
        this.train = trainData;
    }

    @Override
    public List<ScoredItem> recommend(String user, int howMany) {
        return recommend(user, howMany, train.items());
    }

    @Override
    public List<ScoredItem> recommend(String user, int howMany, Set<String> candidateItems) {
        List<ScoredItem> recommended = new LinkedList<>();

        Queue<ScoredItem> queue = new PriorityQueue<>(howMany);

        for (String item : candidateItems) {
            // Discard items in the user's training set
            if (train.existsPreference(user, item)) {
                continue;
            }

            float score = predictScore(user, item);
            // Ignore the item if we were not able to compute a prediction
            if (Float.isNaN(score)) {
                continue;
            }

            ScoredItem si = new ScoredItem(item, score);
            if (queue.size() < howMany) {
                queue.offer(si);
            } else if (queue.peek().compareTo(si) < 0) {
                queue.poll();
                queue.offer(si);
            }
        }

        while (!queue.isEmpty()) {
            recommended.add(0, queue.poll());
        }

        return recommended;
    }

    /**
     * Predicts the preference score of the given user towards the given item.
     *
     * @param user Target user.
     * @param item Target item.
     * @return The predicted score.
     */
    public abstract float predictScore(String user, String item);

}

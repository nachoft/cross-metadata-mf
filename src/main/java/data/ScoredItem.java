package data;

import java.util.Locale;

/**
 * Class that represents a generic item-score pair.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class ScoredItem implements Comparable<ScoredItem> {

    private String item;
    private float score;

    /**
     * Creates a new scored item.
     *
     * @param item Item object.
     * @param score Computed preference score.
     */
    public ScoredItem(String item, float score) {
        this.item = item;
        this.score = score;
    }

    /**
     * Retrieves the item object.
     *
     * @return The stored item object.
     */
    public String getItem() {
        return item;
    }

    /**
     * @return The score corresponding to this item.
     */
    public float getScore() {
        return score;
    }

    /**
     * Updates the score of this item.
     *
     * @param score New score for the item.
     */
    public void setScore(float score) {
        this.score = score;
    }

    /**
     * Returns a string representation of this scored item, in the format
     * item:score.
     *
     * @return A string representation of this scored item.
     */
    @Override
    public String toString() {
        return String.format(Locale.ENGLISH, "%s:%s", item, score);
    }

    /**
     * Compares this scored item's score against the given scored item's score.
     *
     * @param o Target scored item to compare to.
     * @return The result of comparing this score against the given score, see
     * the
     * {@link Float#compare(float, float) compare method in the Float class}.
     */
    @Override
    public int compareTo(ScoredItem o) {
        // Compare by score
        return Float.compare(this.score, o.score);
    }

}

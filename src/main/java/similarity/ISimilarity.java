package similarity;

/**
 * Interface for similarity functions.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public interface ISimilarity {

    /**
     * Computes the similarity score between two elements (users or items).
     *
     * @param first First element.
     * @param second Second element.
     * @return The computed similarity score.
     */
    float compute(String first, String second);
}

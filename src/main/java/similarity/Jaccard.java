package similarity;

import java.util.Set;
import java.util.function.Function;

/**
 * Similarity implementation based on Jaccard's coefficient.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class Jaccard implements ISimilarity {

    private final Function<String, Set<String>> setFun;

    /**
     * Creates a new Jaccard similarity using the given function
     * to retrieve the set of elements for each user/item.
     *
     * @param elemSetFun Function that retrieves the set of elements
     * for users/items.
     */
    public Jaccard(Function<String, Set<String>> elemSetFun) {
        this.setFun = elemSetFun;
    }

    private <T> float jaccard(Set<T> setA, Set<T> setB) {
        int sizeA = setA.size();
        int sizeB = setB.size();

        Set<T> small = sizeA < sizeB ? setA : setB;
        Set<T> large = sizeA < sizeB ? setB : setA;

        long intersection = small.stream().filter(large::contains).count();
        long union = sizeA + sizeB - intersection;

        return (float) intersection / union;
    }

    @Override
    public float compute(String first, String second) {
        return jaccard(setFun.apply(first), setFun.apply(second));
    }

}

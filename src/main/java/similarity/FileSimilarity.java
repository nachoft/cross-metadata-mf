package similarity;

import gnu.trove.map.TObjectFloatMap;
import gnu.trove.map.hash.TObjectFloatHashMap;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Implementation that loads pre-computed similarities into memory. The
 * similarities are required to be symmetrical.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class FileSimilarity implements ISimilarity {

    private final TObjectFloatMap<String> similarities;

    /**
     * Creates a new similarity from the values in the given file. The format
     * must be: item1 TAB item2 TAB score
     *
     * @param file File with the pre-computed similarities.
     * @throws java.io.IOException
     */
    public FileSimilarity(String file) throws IOException {
        similarities = new TObjectFloatHashMap<>();
        loadSimilarities(file);
    }

    private void loadSimilarities(String file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        reader.lines().forEach(line -> {
            String[] tok = line.split("\t");

            String elem1 = tok[0];
            String elem2 = tok[1];
            float value = Float.parseFloat(tok[2]);

            if (Float.isNaN(value)) {
                return;
            }

            // Since the similarity is assumed symmetric store always
            // the smaller element first
            String lookup = elem1.compareTo(elem2) < 0
                            ? elem1 + ":" + elem2
                            : elem2 + ":" + elem1;
            similarities.put(lookup, value);
        });
        reader.close();
    }

    @Override
    public float compute(String first, String second) {
        String lookup = first.compareTo(second) < 0
                        ? first + ":" + second
                        : second + ":" + first;
        return similarities.containsKey(lookup) ? similarities.get(lookup) : 0;
    }
}

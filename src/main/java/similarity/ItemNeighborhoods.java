package similarity;

import data.PreferenceData;
import data.ScoredItem;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;

/**
 * Computes and mantains item neighborhoods.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class ItemNeighborhoods {

    private final int numNeighbors;
    private final Map<String, Queue<ScoredItem>> neighbors;
    private final Map<String, Set<ScoredItem>> invNeighbors;

    public ItemNeighborhoods(PreferenceData train, int num, String simFile, boolean normalize) throws IOException {
        numNeighbors = num;
        neighbors = new HashMap<>();
        invNeighbors = new HashMap<>();
        loadNeighborhoods(train, simFile, normalize);
    }

    private void loadNeighborhoods(PreferenceData train, String file, boolean normalize) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        reader.lines().forEach(line -> {
            String[] tok = line.split("\t");

            String itemA = tok[0];
            String itemB = tok[1];
            float sim = Float.parseFloat(tok[2]);

            if (!train.containsItem(itemA) || !train.containsItem(itemB) || Float.isNaN(sim)) {
                return;
            }

            ScoredItem neigh = new ScoredItem(itemB, sim);

            Queue<ScoredItem> top = neighbors.computeIfAbsent(itemA, i -> new PriorityQueue<>(numNeighbors));
            if (top.size() < numNeighbors) {
                top.offer(neigh);
            } else if (top.peek().compareTo(neigh) < 0) {
                top.poll();
                top.offer(neigh);
            }
        });

        for (String item : neighbors.keySet()) {
            Queue<ScoredItem> neighs = neighbors.get(item);

            if (normalize) {
                double sum = neighs.stream().mapToDouble(n -> n.getScore()).sum();
                neighs.forEach((ScoredItem si) -> si.setScore(si.getScore() / (float) sum));
            }

            for (ScoredItem ni : neighs) {
                String neigh = ni.getItem();
                ScoredItem si = new ScoredItem(item, ni.getScore());
                invNeighbors.computeIfAbsent(neigh, n -> new HashSet<>()).add(si);
            }
        }

        reader.close();
    }

    public Queue<ScoredItem> neighbors(String item) {
        return neighbors.get(item);
    }

    public Set<ScoredItem> invNeighbors(String item) {
        return invNeighbors.get(item);
    }
}

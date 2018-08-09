package util;

import data.PreferenceData;
import data.ScoredItem;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import recommender.IRecommender;
import recommender.knn.ItemKnn;
import recommender.knn.UserKnn;
import recommender.mf.cross.SimMF;
import recommender.mf.FastMF;
import recommender.mf.ImplicitMF;
import recommender.mf.MFRecommender;
import recommender.mf.cross.CentroidMF;
import recommender.mf.cross.NeighborMF;
import similarity.FileSimilarity;
import similarity.ItemNeighborhoods;
import similarity.Jaccard;

/**
 * Command-line tool to run recommendation algorithms for the experiments.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class RecommenderRunner {

    private final PreferenceData train;
    private final PreferenceData test;
    private final Set<String> targetItems;
    private IRecommender recommender;

    public RecommenderRunner(PreferenceData train, PreferenceData test, Set<String> targetItems) {
        this.train = train;
        this.test = test;
        this.targetItems = targetItems;
    }

    private void buildUserKnn(int neighbors) {
        Jaccard jaccard = new Jaccard(train::userItems);
        recommender = new UserKnn(train, jaccard, neighbors);
    }

    private void buildItemKnn() {
        Jaccard jaccard = new Jaccard(train::itemUsers);
        recommender = new ItemKnn(train, jaccard);
    }

    private void buildMF(MFRecommender mf, int factors, float reg, int iters) {
        mf.setNumFactors(factors);
        mf.setLambda(reg);
        mf.setNumIterations(iters);
        mf.train();
        recommender = mf;
    }

    private void buildIMF(int factors, float reg, int iters, float conf) {
        ImplicitMF mf = new ImplicitMF(train);
        mf.setAlpha(conf);
        buildMF(mf, factors, reg, iters);
    }

    private void buildFastIMF(int factors, float reg, int iters, float conf) {
        FastMF mf = new FastMF(train);
        mf.setAlpha(conf);
        buildMF(mf, factors, reg, iters);
    }

    private void buildSimMF(int factors, float reg, int iters, float conf, float lambdaCross, String simFile) throws IOException {
        FileSimilarity sim = new FileSimilarity(simFile);
        SimMF r = new SimMF(train, sim, targetItems);
        r.setLambdaCross(lambdaCross);
        r.setAlpha(conf);
        buildMF(r, factors, reg, iters);
    }

    private void buildCentroidMF(int factors, float reg, int iters, float conf, float lambdaCross, String simFile, int neighbors, boolean normalize) throws IOException {
        ItemNeighborhoods neighs = new ItemNeighborhoods(train, neighbors, simFile, normalize);
        CentroidMF r = new CentroidMF(train, neighs, targetItems);
        r.setLambdaCross(lambdaCross);
        r.setAlpha(conf);
        buildMF(r, factors, reg, iters);
    }

    private void buildNeighborMF(int factors, float reg, int iters, float conf, float lambdaCross, String simFile, int neighbors, boolean normalize) throws IOException {
        ItemNeighborhoods neighs = new ItemNeighborhoods(train, neighbors, simFile, normalize);
        NeighborMF r = new NeighborMF(train, neighs, targetItems);
        r.setLambdaCross(lambdaCross);
        r.setAlpha(conf);
        buildMF(r, factors, reg, iters);
    }

    private void run(int numRecs) {
        for (String user : test.users()) {
            List<ScoredItem> list = recommender.recommend(user, numRecs, targetItems);
            list.forEach(i -> System.out.println(user + "\t" + i.getItem() + "\t" + i.getScore()));
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 4) {
            System.err.println("Usage: <source> <target> <test> <nrecs> ...");
            System.err.println("    userknn <k> : user knn with k neighbors");
            System.err.println("    itemknn : item knn");
            System.err.println("    imf <k> <reg> <iters> <conf> : MF for implicit feedback");
            System.err.println("    fastimf <k> <reg> <iters> <conf> : fast-ALS iMF trained with RR1");
            System.err.println("    simmf <k> <reg> <iters> <conf> <crossreg> <file> : cross similarity MF");
            System.err.println("    centroidmf <k> <reg> <iters> <conf> <crossreg> <file> <neighs> <normalize> : cross centroid MF");
            System.err.println("    neighbormf <k> <reg> <iters> <conf> <crossreg> <file> <neighs> <normalize> : cross neighbor MF");
            return;
        }

        String sourceFile = args[0];
        String trainFile = args[1];
        String testFile = args[2];
        int numRecs = Integer.parseInt(args[3]);

        // Load datasets into memory
        PreferenceData source = PreferenceData.fromFile(sourceFile);
        PreferenceData train = PreferenceData.fromFile(trainFile);
        PreferenceData test = PreferenceData.fromFile(testFile);

        // Provide some stats
        printStats("Source", source);
        printStats("Target", train);
        printStats("Test  ", test);

        // All source domain is training data, but candidate items are only
        // those in the target
        Set<String> targetItems = new HashSet<>(train.items());
        train.merge(source);
        printStats("Train ", train);

        RecommenderRunner cdr = new RecommenderRunner(train, test, targetItems);

        String rec = args[4];
        switch (rec) {
            case "userknn": {
                int k = Integer.parseInt(args[5]);
                cdr.buildUserKnn(k);
                break;
            }
            case "itemknn":
                cdr.buildItemKnn();
                break;
            case "imf": {
                int k = Integer.parseInt(args[5]);
                float reg = Float.parseFloat(args[6]);
                int iters = Integer.parseInt(args[7]);
                float conf = Float.parseFloat(args[8]);
                cdr.buildIMF(k, reg, iters, conf);
                break;
            }
            case "fastimf": {
                int k = Integer.parseInt(args[5]);
                float reg = Float.parseFloat(args[6]);
                int iters = Integer.parseInt(args[7]);
                float conf = Float.parseFloat(args[8]);
                cdr.buildFastIMF(k, reg, iters, conf);
                break;
            }
            case "simmf": {
                int k = Integer.parseInt(args[5]);
                float reg = Float.parseFloat(args[6]);
                int iters = Integer.parseInt(args[7]);
                float conf = Float.parseFloat(args[8]);
                float lambdaCross = Float.parseFloat(args[9]);
                String simFile = args[10];
                cdr.buildSimMF(k, reg, iters, conf, lambdaCross, simFile);
                break;
            }
            case "centroidmf": {
                int k = Integer.parseInt(args[5]);
                float reg = Float.parseFloat(args[6]);
                int iters = Integer.parseInt(args[7]);
                float conf = Float.parseFloat(args[8]);
                float lambdaCross = Float.parseFloat(args[9]);
                String simFile = args[10];
                int neighs = Integer.parseInt(args[11]);
                boolean normalize = "1".equals(args[12]);
                cdr.buildCentroidMF(k, reg, iters, conf, lambdaCross, simFile, neighs, normalize);
                break;
            }
            case "neighbormf": {
                int k = Integer.parseInt(args[5]);
                float reg = Float.parseFloat(args[6]);
                int iters = Integer.parseInt(args[7]);
                float conf = Float.parseFloat(args[8]);
                float lambdaCross = Float.parseFloat(args[9]);
                String simFile = args[10];
                int neighs = Integer.parseInt(args[11]);
                boolean normalize = "1".equals(args[12]);
                cdr.buildNeighborMF(k, reg, iters, conf, lambdaCross, simFile, neighs, normalize);
                break;
            }
            default:
                throw new AssertionError("Unknown recommender.");
        }

        cdr.run(numRecs);
    }

    private static void printStats(String name, PreferenceData data) {
        int users = data.users().size();
        int items = data.items().size();
        int likes = data.size();
        System.err.format("%s:\t%d users\t%d items\t%d likes\n", name, users, items, likes);
    }
}

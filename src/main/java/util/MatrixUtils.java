package util;

import java.util.Random;
import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * Utility class with methods for matrix operations.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class MatrixUtils {

    // Global seed for reproducibility
    public static final long RAND_SEED = 20141207L;

    private MatrixUtils() {
    }

    public static void add(float[] target, float[] toAdd, float scale) {
        for (int i = 0; i < target.length; i++) {
            target[i] += scale * toAdd[i];
        }
    }

    public static float distance2(float[] v, float[] w) {
        float dist = 0;

        for (int i = 0; i < v.length; i++) {
            dist += (v[i] - w[i]) * (v[i] - w[i]);
        }

        return dist;
    }

    /**
     * Initializes the given matrix with independent draws from a gaussian
     * distribution.
     *
     * @param matrix matrix to initialize
     * @param mean mean of the Gaussian distribution to draw from
     * @param std standard deviation of the Gaussian distribution to draw from
     */
    public static void setRandGaussian(float[][] matrix, float mean, float std) {
        Random rand = new Random(RAND_SEED);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = (float) (mean + std * rand.nextGaussian());
            }
        }
    }

    public static void setRandUniform(float[][] matrix, float scale) {
        Random rand = new Random(RAND_SEED);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = scale * rand.nextFloat();
            }
        }
    }

    /**
     * Computes the Frobenius norm of the given matrix, i.e. the sum of the
     * squares of its elements.
     *
     * @param matrix input matrix to compute the norm of
     * @return A scalar with the Frobenius norm of the matrix.
     */
    public static float norm2(float[][] matrix) {
        float norm = 0;

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                norm += matrix[i][j] * matrix[i][j];
            }
        }

        return norm;
    }

    /**
     * Computes the square of the L2 norm of the given vector, i.e. the sum of
     * the squares of its elements.
     *
     * @param vector Input vector.
     * @return The squared L2 norm of the given vector.
     */
    public static float norm2(float[] vector) {
        float norm = 0;

        for (int i = 0; i < vector.length; i++) {
            norm += vector[i] * vector[i];
        }

        return norm;
    }

    /**
     * Computes the dot product between two vectors of floats.
     *
     * @param x first vector
     * @param y second vector
     * @return The dot product between the vectors.
     */
    public static float dotProduct(float[] x, float[] y) {
        float prod = 0;

        for (int k = 0; k < x.length; k++) {
            prod += x[k] * y[k];
        }

        return prod;
    }

    /**
     * Computes the vector sum of the given rows in a matrix.
     *
     * @param rows Stream with the indices of the rows to be added.
     * @param matrix Data matrix.
     * @return A new vector with the sum of the given rows.
     */
    public static float[] sumRows(IntStream rows, float[][] matrix) {
        int cols = matrix[0].length;
        float[] result = new float[cols];

        rows.forEach(i -> {
            for (int j = 0; j < cols; j++) {
                result[j] += matrix[i][j];
            }
        });

        return result;
    }

    /**
     * Given a matrix A, computes the product <code>A'A</code>, where A' is the
     * transpose of A. It is possible to specify which rows of A should be used
     * in the computation.
     *
     * @param matrix Input matrix.
     * @param rowSelector Predicate that returns <tt>true</tt> for row indices
     * that should be used in the computation of <code>A'A</code>.
     * @return The product <code>A'A</code>.
     */
    public static float[][] transposeTimes(float[][] matrix, IntPredicate rowSelector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] res = new float[cols][cols];

        for (int i = 0; i < cols; i++) {
            for (int j = i; j < cols; j++) {
                float x = 0;
                for (int k = 0; k < rows; k++) {
                    if (rowSelector.test(k)) {
                        x += matrix[k][i] * matrix[k][j];
                    }
                }
                res[i][j] = x;
                res[j][i] = x;
            }
        }

        return res;
    }

    /**
     * Given a matrix A, computes the product <code>A'A</code>, where A' is the
     * transpose of A. It is possible to specify which rows of A should be used
     * in the computation.
     *
     * @param inputMatrix Input matrix A.
     * @param outputMatrix Output matrix in which the result A'A is stored.
     */
    public static void transposeTimes(float[][] inputMatrix, double[][] outputMatrix) {
        int rows = inputMatrix.length;
        int cols = inputMatrix[0].length;

        for (int i = 0; i < cols; i++) {
            for (int j = i; j < cols; j++) {
                float x = 0;
                for (int k = 0; k < rows; k++) {
                    x += inputMatrix[k][i] * inputMatrix[k][j];
                }
                outputMatrix[i][j] = x;
                outputMatrix[j][i] = x;
            }
        }
    }

    /**
     * Given a matrix A, computes the product <code>A'A</code>, where A' is the
     * transpose of A.
     *
     * @param matrix Input matrix.
     * @return The product <code>A'A</code>.
     */
    public static float[][] transposeTimes(float[][] matrix) {
        return transposeTimes(matrix, k -> true);
    }

}

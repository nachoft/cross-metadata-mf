package data;

import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import java.util.Collection;
import java.util.Set;

/**
 * Index that maps objects to consecutive, zero-based, integer Ids.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 *
 * @param <T> Type of the elements.
 */
public class Index<T> {

    // Object to ID mapping
    private final TObjectIntMap<T> idMap;
    // ID to object inverse mapping
    private final TIntObjectMap<T> revIdMap;
    // Max id
    private int maxId;

    /**
     * Creates a new empty index.
     */
    public Index() {
        // Instead checking if the index contains a key and throwing an
        // exception, which requires 2 lookups, we set the no-entry value to
        // -1, which is not a valid Id that should break when indexing
        this.idMap = new TObjectIntHashMap<>(10, 0.5f, -1);
        this.revIdMap = new TIntObjectHashMap<>();
        maxId = -1;
    }

    /**
     * Creates a new index for the given collection.
     *
     * @param elements Collection of elements from which the index is built.
     */
    public Index(Collection<T> elements) {
        this();
        addElements(elements);
    }

    /**
     * Adds the given collection of elements to this index.
     *
     * @param elements Collection of elements to be added.
     */
    public final void addElements(Collection<T> elements) {
        int idx = maxId;

        for (T element : elements) {
            // The collection may contain duplicates, create a new id only once
            if (!idMap.containsKey(element)) {
                idx++;
                idMap.put(element, idx);
                revIdMap.put(idx, element);
            }
        }

        maxId = idx;
    }

    /**
     * Retrieves the numeric id of the given element.
     *
     * @param element Query element.
     * @return Id of the given element.
     */
    public int getId(T element) {
        return idMap.get(element);
    }

    /**
     * Retrieves the element corresponding to the given id.
     *
     * @param id Query id.
     * @return The corresponding element, or <tt>null</tt> if there is no such
     * element.
     */
    public T getElement(int id) {
        return revIdMap.get(id);
    }

    /**
     * Retrieves the set of all indexed elements.
     *
     * @return The set of all indexed elements, in no particular order.
     */
    public Set<T> getElements() {
        return idMap.keySet();
    }

    /**
     * Checks whether this index is empty.
     *
     * @return <tt>True</tt> if this index contains no elements, and <tt>false
     * </tt> otherwise.
     */
    public boolean isEmpty() {
        return idMap.isEmpty();
    }

    /**
     * Retrieves the maximum Id contained in this index.
     *
     * @return The maximum Id contained in this index, or -1 if this index is
     * empty.
     */
    public int maxId() {
        return maxId;
    }

    /**
     * Retrieves the size of this index, as the number of Object - ID mappings.
     *
     * @return The size of this index.
     */
    public int size() {
        return idMap.size();
    }
}

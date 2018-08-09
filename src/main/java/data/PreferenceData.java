package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Container class for unary/binary feedback datasets.
 *
 * Note that neither ratings or frequencies are stored and only positive
 * feedback is considered. That is, this class only models (user, item) pairs
 * for which an observation is available.
 *
 * @author Ignacio Fernández (ignacio.fernandezt@uam.es)
 * @author Iván Cantador (ivan.cantador@uam.es)
 */
public class PreferenceData {

    // User and item indices
    protected final Index<String> userIndex;
    protected final Index<String> itemIndex;

    // User-items and item-users mappings
    protected final Map<String, Set<String>> userItems;
    protected final Map<String, Set<String>> itemUsers;

    // Cached number of observations
    protected int numObservations;

    /**
     * Load a dataset from a file into memory, in which both users and items are
     * represented as strings.
     *
     * See {@link #load(java.lang.String) load()} for more details.
     *
     * @param file File to load.
     * @return A new <code>Dataset</code> object representing the data in the
     * file.
     * @throws IOException
     */
    public static PreferenceData fromFile(String file) throws IOException {
        PreferenceData dataset = new PreferenceData();
        dataset.load(file);
        return dataset;
    }

    /**
     * Create an empty dataset.
     */
    public PreferenceData() {
        userIndex = new Index<>();
        itemIndex = new Index<>();
        userItems = new HashMap<>();
        itemUsers = new HashMap<>();
    }

    /**
     * Loads a unary/binary feedback dataset from a given file.
     *
     * The file should contain an observation per line. Each observation
     * consists on a user identifier, a single tab separator, and an item
     * identifier.
     *
     * @param file Path of the file to read.
     * @throws IOException
     */
    public void load(String file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;

        numObservations = 0;

        while ((line = reader.readLine()) != null) {
            // Ignore lines starting with #
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }

            String[] tok = line.split("\t");

            String user = tok[0];
            String item = tok[1];

            if (!userItems.containsKey(user)) {
                userItems.put(user, new HashSet<>());
            }
            Set<String> items = userItems.get(user);
            items.add(item);

            if (!itemUsers.containsKey(item)) {
                itemUsers.put(item, new HashSet<>());
            }
            Set<String> users = itemUsers.get(item);
            users.add(user);

            numObservations++;
        }

        reader.close();

        userIndex.addElements(userItems.keySet());
        itemIndex.addElements(itemUsers.keySet());
    }

    /**
     * Returns the internal user ID corresponding to the given user.
     *
     * @param user Target user
     * @return The internal numeric ID of the given user.
     */
    public int userId(String user) {
        return userIndex.getId(user);
    }

    /**
     * Returns the internal item ID corresponding to the given item.
     *
     * @param item Target item
     * @return The internal numeric ID of the given item.
     */
    public int itemId(String item) {
        return itemIndex.getId(item);
    }

    /**
     * Returns the user object corresponding to the given internal user ID.
     *
     * @param userId Internal ID of the user
     * @return The user object.
     */
    public String user(int userId) {
        return userIndex.getElement(userId);
    }

    /**
     * Returns the item object corresponding to the given internal item ID.
     *
     * @param itemId Internal ID of the item
     * @return The item object.
     */
    public String item(int itemId) {
        return itemIndex.getElement(itemId);
    }

    /**
     * Returns the maximum internal user ID.
     *
     * @return The maximum internal user ID in this dataset.
     */
    public int maxUserID() {
        return userIndex.maxId();
    }

    /**
     * Returns the maximum internal item ID.
     *
     * @return The maximum internal item ID in this dataset.
     */
    public int maxItemID() {
        return itemIndex.maxId();
    }

    /**
     * Returns a stream of all the user Ids in this dataset, sorted from 0 to
     * the max user Id.
     * <p>
     * <strong>NOTE: </strong>remember that Strems cannot be reused, so another
     * call to this method is required once the stream is consumed.
     *
     * @return A stream with the user IDs in this dataset.
     */
    public IntStream userIds() {
        return IntStream.rangeClosed(0, userIndex.maxId());
    }

    /**
     * Returns the set of all the item Ids in this dataset, sorted from 0 to the
     * max item Id.
     * <p>
     * <strong>NOTE: </strong>remember that Strems cannot be reused, so another
     * call to this method is required once the stream is consumed.
     *
     * @return A stream with the item IDs in this dataset.
     */
    public IntStream itemIds() {
        return IntStream.rangeClosed(0, itemIndex.maxId());
    }

    /**
     * Returns the set of users.
     *
     * @return The set of users in this dataset.
     */
    public Set<String> users() {
        return userItems.keySet();
    }

    /**
     * Returns the set of items.
     *
     * @return The set of items in this dataset.
     */
    public Set<String> items() {
        return itemUsers.keySet();
    }

    /**
     * Returns the set of items preferred by the given user.
     *
     * @param user Target user.
     * @return The set of items preferred by the user.
     */
    public Set<String> userItems(String user) {
        return userItems.get(user);
    }

    /**
     * Returns the set of users that expressed a preference for the given item.
     *
     * @param item Target item.
     * @return The set of users that expressed a preference for the item.
     */
    public Set<String> itemUsers(String item) {
        return itemUsers.get(item);
    }

    /**
     * Tests whether a given user-item pair preference is observed in this
     * dataset.
     *
     * @param user Target user.
     * @param item Target item.
     * @return True iff the user expressed a preference towards the item.
     */
    public boolean existsPreference(String user, String item) {
        Set<String> ui = userItems.get(user);
        return ui != null && ui.contains(item);
    }

    /**
     * Tests whether a given user exists in this dataset.
     *
     * @param user Target user.
     * @return True iff the user exists in this dataset.
     */
    public boolean containsUser(String user) {
        return userItems.containsKey(user);
    }

    /**
     * Tests whether a given item exists in this dataset.
     *
     * @param item Target item.
     * @return True iff the item exists in this dataset.
     */
    public boolean containsItem(String item) {
        return itemUsers.containsKey(item);
    }

    /**
     * Returns the size of this dataset as the number of observations (ratings).
     *
     * @return The number of observations in this dataset.
     */
    public int size() {
        return numObservations;
    }

    /**
     * Merges the given preference data into this instance. Internal ids will be
     * created for new users and items, and the maximum ids and number of
     * observations will be updated.
     *
     * @param other Other preference data to merge.
     */
    public void merge(PreferenceData other) {
        // Merge users
        for (String user : other.users()) {
            // Copy the user's items
            Set<String> thisUserItems = userItems.get(user);
            if (thisUserItems == null) {
                // User is new
                userItems.put(user, thisUserItems = new HashSet<>());
            }
            thisUserItems.addAll(other.userItems(user));
        }

        // Merge items
        for (String item : other.items()) {
            // Copy the item users
            Set<String> thisItemUsers = itemUsers.get(item);
            if (thisItemUsers == null) {
                // Item is new
                itemUsers.put(item, thisItemUsers = new HashSet<>());
            }
            thisItemUsers.addAll(other.itemUsers(item));
        }

        // Update indices
        userIndex.addElements(other.users());
        itemIndex.addElements(other.items());

        // Update number of observations
        numObservations = users().stream()
                .mapToInt(u -> userItems.get(u).size())
                .sum();
    }
}

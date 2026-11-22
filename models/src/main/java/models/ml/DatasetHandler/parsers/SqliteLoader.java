package models.ml.DatasetHandler.parsers;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for loading data from SQLite databases.
 * Reads the first table found in the database and returns it as a list of string arrays.
 */
public final class SqliteLoader {

    private SqliteLoader() {}

    /**
     * Reads the first table from a SQLite database.
     *
     * @param dbPath path to the SQLite database file
     * @return list of rows, first row is header, subsequent rows are data
     * @throws SQLException if database cannot be opened, no tables found, or query fails
     */
    public static List<String[]> readFirstTable(String dbPath) throws SQLException {
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
             Statement stmt = conn.createStatement()) {

            ResultSet table = stmt.executeQuery("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1");
            if (!table.next()) {
                throw new SQLException("No table found");
            }
            String tableName = table.getString("name");

            String query = "SELECT * FROM " + tableName;
            try (ResultSet rs = stmt.executeQuery(query)) {
                ResultSetMetaData md = rs.getMetaData();
                int cols = md.getColumnCount();

                List<String[]> rows = new ArrayList<>();
                String[] header = new String[cols];
                for (int i = 1; i <= cols; i++) {
                    header[i - 1] = md.getColumnName(i);
                }
                rows.add(header);

                while (rs.next()) {
                    String[] row = new String[cols];
                    for (int i = 1; i <= cols; i++) {
                        row[i - 1] = String.valueOf(rs.getObject(i));
                    }
                    rows.add(row);
                }

                return rows;
            }
        }
    }
}
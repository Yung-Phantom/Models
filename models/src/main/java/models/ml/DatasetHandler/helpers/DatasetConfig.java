package models.ml.DatasetHandler.helpers;

public enum DatasetConfig {
    HAS_ID_WITH_HEADER(true, true),
    HAS_ID_NO_HEADER(true, false),
    NO_ID_WITH_HEADER(false, true),
    NO_ID_NO_HEADER(false, false),
    DEFAULT(false, true);
    public final boolean hasID;
    public final boolean hasHeader;

    DatasetConfig(boolean hasID, boolean hasHeader) {
        this.hasID = hasID;
        this.hasHeader = hasHeader;
    }
}

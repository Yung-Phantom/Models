package models.ml.Preprocessing.Image;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable container for image pixel data in [0, 1] range.
 * <p>
 * Pixels are stored in row-major, channel-interleaved order:
 * index = (y * width + x) * channels + c
 * <p>
 * Not thread-safe; immutable after construction.
 */
public final class ImageData {

    private final int width;
    private final int height;
    private final int channels;
    private final double[] pixels;

    private ImageData(int width, int height, int channels, double[] pixels) {
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.pixels = pixels;
    }

    /**
     * Creates an ImageData instance with validation.
     *
     * @param width    image width in pixels; must be > 0
     * @param height   image height in pixels; must be > 0
     * @param channels number of channels; must be 1 (grayscale) or 3 (RGB)
     * @param pixels   pixel values in [0, 1], length must equal width * height * channels
     * @return new ImageData instance
     * @throws IllegalArgumentException if any validation fails
     */
    public static ImageData of(int width, int height, int channels, double[] pixels) {
        if (width <= 0) {
            throw new IllegalArgumentException("width must be > 0, got " + width);
        }
        if (height <= 0) {
            throw new IllegalArgumentException("height must be > 0, got " + height);
        }
        if (channels != 1 && channels != 3) {
            throw new IllegalArgumentException("channels must be 1 or 3, got " + channels);
        }
        if (pixels == null) {
            throw new IllegalArgumentException("pixels array cannot be null");
        }
        long expected = (long) width * height * channels;
        if (pixels.length != expected) {
            throw new IllegalArgumentException(
                    "pixels.length (" + pixels.length + ") must equal width * height * channels (" + expected + ")");
        }
        for (int i = 0; i < pixels.length; i++) {
            double v = pixels[i];
            if (Double.isNaN(v) || Double.isInfinite(v) || v < 0.0 || v > 1.0) {
                throw new IllegalArgumentException(
                        "pixel at index " + i + " must be a finite number in [0, 1], got " + v);
            }
        }
        // Defensive copy
        double[] copy = Arrays.copyOf(pixels, pixels.length);
        return new ImageData(width, height, channels, copy);
    }

    /**
     * Returns the image width in pixels.
     */
    public int getWidth() {
        return width;
    }

    /**
     * Returns the image height in pixels.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Returns the number of channels (1 = grayscale, 3 = RGB).
     */
    public int getChannels() {
        return channels;
    }

    /**
     * Returns a copy of the pixel array.
     * The array is in row-major, channel-interleaved order:
     * index = (y * width + x) * channels + c
     *
     * @return copy of pixels array
     */
    public double[] getPixels() {
        return Arrays.copyOf(pixels, pixels.length);
    }

    /**
     * Returns the pixel value at the given coordinates and channel.
     *
     * @param x        column index [0, width)
     * @param y        row index [0, height)
     * @param c        channel index [0, channels)
     * @return pixel value in [0, 1]
     * @throws IndexOutOfBoundsException if coordinates are out of bounds
     */
    public double getPixel(int x, int y, int c) {
        if (x < 0 || x >= width) {
            throw new IndexOutOfBoundsException("x (" + x + ") out of bounds [0, " + width + ")");
        }
        if (y < 0 || y >= height) {
            throw new IndexOutOfBoundsException("y (" + y + ") out of bounds [0, " + height + ")");
        }
        if (c < 0 || c >= channels) {
            throw new IndexOutOfBoundsException("channel (" + c + ") out of bounds [0, " + channels + ")");
        }
        return pixels[(y * width + x) * channels + c];
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ImageData)) return false;
        ImageData that = (ImageData) o;
        return width == that.width
                && height == that.height
                && channels == that.channels
                && Arrays.equals(pixels, that.pixels);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(width, height, channels);
        result = 31 * result + Arrays.hashCode(pixels);
        return result;
    }

    @Override
    public String toString() {
        return "ImageData[" + width + "x" + height + "x" + channels + "]";
    }
}
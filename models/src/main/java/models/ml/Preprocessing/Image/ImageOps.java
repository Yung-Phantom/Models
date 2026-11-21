package models.ml.Preprocessing.Image;

import java.util.Arrays;

/**
 * Stateless, pure image processing operations on {@link ImageData}.
 * <p>
 * Not instantiable; all methods are static.
 * Not thread-safe; callers must ensure thread confinement.
 */
public final class ImageOps {

    private ImageOps() {}

    /**
     * Converts an image to grayscale using the standard luminance formula.
     * 1-channel images are returned unchanged.
     *
     * @param image input image (1 or 3 channels); must not be null
     * @return 1-channel grayscale image
     * @throws IllegalArgumentException if image is null
     */
    public static ImageData toGrayscale(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        if (image.getChannels() == 1) {
            return image;
        }
        int w = image.getWidth();
        int h = image.getHeight();
        double[] src = image.getPixels();
        double[] dst = new double[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = (y * w + x) * 3;
                double r = src[idx];
                double g = src[idx + 1];
                double b = src[idx + 2];
                // Standard luminance formula
                dst[y * w + x] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
        }
        return ImageData.of(w, h, 1, dst);
    }

    /**
     * Converts an image to 3-channel RGB by replicating the single channel.
     * 3-channel images are returned unchanged.
     *
     * @param image input image (1 or 3 channels); must not be null
     * @return 3-channel RGB image
     * @throws IllegalArgumentException if image is null
     */
    public static ImageData toRgb(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        if (image.getChannels() == 3) {
            return image;
        }
        int w = image.getWidth();
        int h = image.getHeight();
        double[] src = image.getPixels();
        double[] dst = new double[w * h * 3];
        for (int i = 0; i < w * h; i++) {
            double v = src[i];
            dst[i * 3] = v;
            dst[i * 3 + 1] = v;
            dst[i * 3 + 2] = v;
        }
        return ImageData.of(w, h, 3, dst);
    }

    /**
     * Converts an image to the specified channel mode.
     *
     * @param image input image; must not be null
     * @param mode  target channel mode; must not be null
     * @return converted image
     * @throws IllegalArgumentException if image or mode is null
     */
    public static ImageData convert(ImageData image, ChannelMode mode) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        if (mode == null) {
            throw new IllegalArgumentException("mode cannot be null");
        }
        return switch (mode) {
            case GRAY -> toGrayscale(image);
            case RGB -> toRgb(image);
        };
    }

    /**
     * Resizes an image using bilinear interpolation.
     * Same-size images are returned unchanged.
     * Constant images remain constant.
     *
     * @param image      input image (1 or 3 channels); must not be null
     * @param newWidth   target width; must be > 0
     * @param newHeight  target height; must be > 0
     * @return resized image with same channel count
     * @throws IllegalArgumentException if image is null, or newWidth/newHeight <= 0
     */
    public static ImageData resize(ImageData image, int newWidth, int newHeight) {
        if (image == null) {
            throw new IllegalArgumentException("image cannot be null");
        }
        if (newWidth <= 0) {
            throw new IllegalArgumentException("newWidth must be > 0, got " + newWidth);
        }
        if (newHeight <= 0) {
            throw new IllegalArgumentException("newHeight must be > 0, got " + newHeight);
        }
        int w = image.getWidth();
        int h = image.getHeight();
        int c = image.getChannels();
        if (w == newWidth && h == newHeight) {
            return image;
        }
        double[] src = image.getPixels();
        double[] dst = new double[newWidth * newHeight * c];
        // Pixel-center mapping: src = (dst + 0.5) * srcSize / dstSize - 0.5
        for (int y = 0; y < newHeight; y++) {
            double sy = (y + 0.5) * h / (double) newHeight - 0.5;
            int y0 = (int) Math.floor(sy);
            if (y0 < 0) y0 = 0;
            int y1 = Math.min(y0 + 1, h - 1);
            double wy1 = sy - y0;
            double wy0 = 1.0 - wy1;
            for (int x = 0; x < newWidth; x++) {
                double sx = (x + 0.5) * w / (double) newWidth - 0.5;
                int x0 = (int) Math.floor(sx);
                if (x0 < 0) x0 = 0;
                int x1 = Math.min(x0 + 1, w - 1);
                double wx1 = sx - x0;
                double wx0 = 1.0 - wx1;
                for (int ch = 0; ch < c; ch++) {
                    double v00 = src[(y0 * w + x0) * c + ch];
                    double v01 = src[(y0 * w + x1) * c + ch];
                    double v10 = src[(y1 * w + x0) * c + ch];
                    double v11 = src[(y1 * w + x1) * c + ch];
                    double interp = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                    // Clamp to [0, 1] to handle interpolation overshoot
                    if (interp < 0.0) interp = 0.0;
                    else if (interp > 1.0) interp = 1.0;
                    dst[(y * newWidth + x) * c + ch] = interp;
                }
            }
        }
        return ImageData.of(newWidth, newHeight, c, dst);
    }
}
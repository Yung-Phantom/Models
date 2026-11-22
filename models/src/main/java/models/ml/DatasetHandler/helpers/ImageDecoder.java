package models.ml.DatasetHandler.helpers;

import models.ml.Preprocessing.Image.ImageData;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Decodes image files into {@link ImageData} using {@link ImageIO}.
 * <p>
 * Not instantiable; all methods are static.
 */
public final class ImageDecoder {

    private ImageDecoder() {}

    /**
     * Reads an image file and returns it as 3-channel RGB {@link ImageData}
     * with pixel values normalized to [0, 1].
     * <p>
     * Handles indexed color, grayscale, and alpha sources by reading RGB directly
     * from the decoded image. Transparent pixels are composited onto white.
     *
     * @param file path to image file; must not be null
     * @return 3-channel RGB ImageData
     * @throws IllegalArgumentException if file is null, not a regular file,
     *         or cannot be decoded as an image
     */
    public static ImageData decode(Path file) {
        if (file == null) {
            throw new IllegalArgumentException("file path cannot be null");
        }
        if (!Files.isRegularFile(file)) {
            throw new IllegalArgumentException("file does not exist or is not a regular file: " + file);
        }
        try {
            BufferedImage img = ImageIO.read(file.toFile());
            if (img == null) {
                throw new IllegalArgumentException("file is not a decodable image: " + file);
            }
            int w = img.getWidth();
            int h = img.getHeight();

            double[] pixels = new double[w * h * 3];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int rgb = img.getRGB(x, y);
                    // Composite onto white background if transparent
                    int alpha = (rgb >> 24) & 0xFF;
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    if (alpha != 255) {
                        double a = alpha / 255.0;
                        r = (int) Math.round(r * a + 255 * (1 - a));
                        g = (int) Math.round(g * a + 255 * (1 - a));
                        b = (int) Math.round(b * a + 255 * (1 - a));
                    }
                    int idx = (y * w + x) * 3;
                    pixels[idx] = r / 255.0;
                    pixels[idx + 1] = g / 255.0;
                    pixels[idx + 2] = b / 255.0;
                }
            }
            return ImageData.of(w, h, 3, pixels);
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to read image file: " + file, e);
        }
    }
}
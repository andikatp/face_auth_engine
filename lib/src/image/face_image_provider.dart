import 'dart:io';
import 'dart:typed_data';

/// A simple buffer holding a 3-channel RGB image (no alpha channel).
/// Used strictly internally for processing pixels.
class FaceImageBuffer {
  final int width;
  final int height;
  final Uint8List pixels;

  FaceImageBuffer({
    required this.width,
    required this.height,
    required this.pixels,
  }) {
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
        'Pixels length must be exactly width * height * 3 for an RGB image (no alpha).',
      );
    }
  }

  /// Gets the red channel value of the pixel at (x, y)
  int getR(int x, int y) => pixels[(y * width + x) * 3];

  /// Gets the green channel value of the pixel at (x, y)
  int getG(int x, int y) => pixels[(y * width + x) * 3 + 1];

  /// Gets the blue channel value of the pixel at (x, y)
  int getB(int x, int y) => pixels[(y * width + x) * 3 + 2];
}

/// Abstract provider for decoding, cropping, and resizing images.
/// The host app should provide an implementation, for example using `package:image`.
abstract class FaceImageProvider {
  /// Loads an image from a file, fixing any EXIF orientation issues,
  /// and returns it as an RGB buffer.
  Future<FaceImageBuffer> loadImage(File file);

  /// Resizes the image to the specified width and height.
  Future<FaceImageBuffer> resize(FaceImageBuffer image, int width, int height);

  /// Crops a section of the image
  Future<FaceImageBuffer> crop(
    FaceImageBuffer image,
    int x,
    int y,
    int width,
    int height,
  );
}

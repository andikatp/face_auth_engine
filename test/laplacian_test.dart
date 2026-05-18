import 'dart:typed_data';

import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:face_auth_engine/src/liveness/laplacian.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  FaceImageBuffer createTestImage(int width, int height, int Function(int x, int y) pixelValue) {
    final pixels = Uint8List(width * height * 3);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final val = pixelValue(x, y);
        final idx = (y * width + x) * 3;
        pixels[idx] = val;
        pixels[idx + 1] = val;
        pixels[idx + 2] = val;
      }
    }
    return FaceImageBuffer(width: width, height: height, pixels: pixels);
  }

  group('laplacianScore', () {
    test('should return higher score for sharp images', () {
      // Create a sharp image with high contrast edges
      final sharpImage = createTestImage(100, 100, (x, y) => ((x ~/ 10) + (y ~/ 10)) % 2 == 0 ? 255 : 0);

      // Create a blurry image (uniform color)
      final blurryImage = createTestImage(100, 100, (x, y) => 128);

      final sharpScore = laplacianScore(sharpImage, laplacePixelThreshold: 50);
      final blurryScore = laplacianScore(
        blurryImage,
        laplacePixelThreshold: 50,
      );

      expect(sharpScore, greaterThan(blurryScore));
    });

    test('should handle small threshold', () {
      final testImage = createTestImage(50, 50, (x, y) => (x * 5).clamp(0, 255));

      final lowThreshold = laplacianScore(testImage, laplacePixelThreshold: 10);
      final highThreshold = laplacianScore(
        testImage,
        laplacePixelThreshold: 100,
      );

      expect(lowThreshold, greaterThanOrEqualTo(highThreshold));
    });
  });

  group('laplacianVariance', () {
    test('should return higher variance for sharp images', () {
      // Create a sharp image
      final sharpImage = createTestImage(100, 100, (x, y) => ((x ~/ 5) + (y ~/ 5)) % 2 == 0 ? 255 : 0);

      // Create a blurry image
      final blurryImage = createTestImage(100, 100, (x, y) => 128);

      final sharpVar = laplacianVariance(sharpImage);
      final blurryVar = laplacianVariance(blurryImage);

      expect(sharpVar, greaterThan(blurryVar));
      expect(blurryVar, closeTo(0, 0.001)); // Uniform image has zero variance
    });
  });
}

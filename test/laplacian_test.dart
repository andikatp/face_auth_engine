import 'package:face_auth_engine/src/liveness/laplacian.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

void main() {
  group('laplacianScore', () {
    test('should return higher score for sharp images', () {
      // Create a sharp image with high contrast edges
      final sharpImage = img.Image(width: 100, height: 100);
      for (int x = 0; x < 100; x++) {
        for (int y = 0; y < 100; y++) {
          // Checkerboard pattern creates sharp edges
          final value = ((x ~/ 10) + (y ~/ 10)) % 2 == 0 ? 255 : 0;
          sharpImage.setPixelRgb(x, y, value, value, value);
        }
      }

      // Create a blurry image (uniform color)
      final blurryImage = img.Image(width: 100, height: 100);
      for (int x = 0; x < 100; x++) {
        for (int y = 0; y < 100; y++) {
          blurryImage.setPixelRgb(x, y, 128, 128, 128);
        }
      }

      final sharpScore = laplacianScore(sharpImage, laplacePixelThreshold: 50);
      final blurryScore = laplacianScore(
        blurryImage,
        laplacePixelThreshold: 50,
      );

      expect(sharpScore, greaterThan(blurryScore));
    });

    test('should handle small threshold', () {
      final testImage = img.Image(width: 50, height: 50);

      // Gradient image
      for (int x = 0; x < 50; x++) {
        for (int y = 0; y < 50; y++) {
          final value = (x * 5).clamp(0, 255);
          testImage.setPixelRgb(x, y, value, value, value);
        }
      }

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
      final sharpImage = img.Image(width: 100, height: 100);
      for (int x = 0; x < 100; x++) {
        for (int y = 0; y < 100; y++) {
          final value = ((x ~/ 5) + (y ~/ 5)) % 2 == 0 ? 255 : 0;
          sharpImage.setPixelRgb(x, y, value, value, value);
        }
      }

      // Create a blurry image
      final blurryImage = img.Image(width: 100, height: 100);
      for (int x = 0; x < 100; x++) {
        for (int y = 0; y < 100; y++) {
          blurryImage.setPixelRgb(x, y, 128, 128, 128);
        }
      }

      final sharpVar = laplacianVariance(sharpImage);
      final blurryVar = laplacianVariance(blurryImage);

      expect(sharpVar, greaterThan(blurryVar));
      expect(blurryVar, closeTo(0, 0.001)); // Uniform image has zero variance
    });
  });
}

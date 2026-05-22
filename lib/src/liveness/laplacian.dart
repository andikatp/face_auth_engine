import 'dart:typed_data';

import 'package:face_auth_engine/src/image/face_image_provider.dart';

/// Calculates mean luma (brightness) of the image.
/// Returns a value in [0, 255]: 0 = totally black, 255 = totally white.
/// Values below ~40 are considered low-light in typical selfie scenarios.
double brightnessScore(FaceImageBuffer image) {
  double sum = 0;
  final bytes = image.pixels;
  final pixelCount = image.width * image.height;
  for (var i = 0, p = 0; i < pixelCount; i++, p += 3) {
    // ITU-R BT.601 luma weights (integer approximation)
    sum += (77 * bytes[p] + 150 * bytes[p + 1] + 29 * bytes[p + 2]) >> 8;
  }
  return sum / pixelCount;
}

/// Calculate Laplacian score for blur detection
/// Higher score = sharper image
int laplacianScore(
  FaceImageBuffer resized, {
  required int laplacePixelThreshold,
}) {
  final w = resized.width;
  final h = resized.height;
  final gray = Uint8List(w * h);
  final bytes = resized.pixels;

  // Convert to grayscale
  for (var i = 0, p = 0; i < gray.length; i++, p += 3) {
    final r = bytes[p];
    final g = bytes[p + 1];
    final b = bytes[p + 2];
    gray[i] = (77 * r + 150 * g + 29 * b) >> 8;
  }

  var score = 0;

  for (var r = 1; r < h - 1; r++) {
    final rowOffset = r * w;
    for (var c = 1; c < w - 1; c++) {
      final idx = rowOffset + c;
      final center = gray[idx];

      final north = gray[idx - w];
      final south = gray[idx + w];
      final west = gray[idx - 1];
      final east = gray[idx + 1];

      final conv = north + south + west + east - (center << 2);

      if (conv > laplacePixelThreshold || conv < -laplacePixelThreshold) {
        score++;
      }
    }
  }

  return score;
}

/// Calculate Laplacian variance for blur detection
/// This is more robust than pixel count method
/// Returns variance value (higher = sharper)
double laplacianVariance(FaceImageBuffer resized) {
  final w = resized.width;
  final h = resized.height;
  final gray = Uint8List(w * h);
  final bytes = resized.pixels;

  // Convert to grayscale
  for (var i = 0, p = 0; i < gray.length; i++, p += 3) {
    final r = bytes[p];
    final g = bytes[p + 1];
    final b = bytes[p + 2];
    gray[i] = (77 * r + 150 * g + 29 * b) >> 8;
  }

  // Calculate Laplacian and accumulate for variance
  double sum = 0;
  double sumSq = 0;
  var count = 0;

  for (var r = 1; r < h - 1; r++) {
    final rowOffset = r * w;
    for (var c = 1; c < w - 1; c++) {
      final idx = rowOffset + c;
      final center = gray[idx];

      final north = gray[idx - w];
      final south = gray[idx + w];
      final west = gray[idx - 1];
      final east = gray[idx + 1];

      final conv = (north + south + west + east - (center << 2)).toDouble();
      sum += conv;
      sumSq += conv * conv;
      count++;
    }
  }

  if (count == 0) return 0;

  final mean = sum / count;
  final variance = (sumSq / count) - (mean * mean);

  return variance;
}

import 'dart:typed_data';

import '../image/face_image_provider.dart';

/// Calculate Laplacian score for blur detection
/// Higher score = sharper image
int laplacianScore(FaceImageBuffer resized, {required int laplacePixelThreshold}) {
  final w = resized.width;
  final h = resized.height;
  final gray = Uint8List(w * h);
  final bytes = resized.pixels;

  // Convert to grayscale
  for (int i = 0, p = 0; i < gray.length; i++, p += 3) {
    final r = bytes[p];
    final g = bytes[p + 1];
    final b = bytes[p + 2];
    gray[i] = ((77 * r + 150 * g + 29 * b) >> 8);
  }

  int score = 0;

  for (int r = 1; r < h - 1; r++) {
    final rowOffset = r * w;
    for (int c = 1; c < w - 1; c++) {
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
  for (int i = 0, p = 0; i < gray.length; i++, p += 3) {
    final r = bytes[p];
    final g = bytes[p + 1];
    final b = bytes[p + 2];
    gray[i] = ((77 * r + 150 * g + 29 * b) >> 8);
  }

  // Calculate Laplacian and accumulate for variance
  double sum = 0;
  double sumSq = 0;
  int count = 0;

  for (int r = 1; r < h - 1; r++) {
    final rowOffset = r * w;
    for (int c = 1; c < w - 1; c++) {
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

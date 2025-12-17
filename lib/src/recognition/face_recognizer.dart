import 'dart:math' as math;
import 'dart:typed_data';

/// Stateless face recognition using L2 distance.
class FaceRecognizer {
  /// Verify if two embeddings belong to the same person.
  /// Returns true if distance is less than threshold.
  bool verify(
    Float32List embedding1,
    Float32List embedding2,
    double threshold,
  ) {
    final distance = _euclideanDistance(embedding1, embedding2);
    return distance < threshold;
  }

  /// Calculate Euclidean (L2) distance between two embeddings.
  double _euclideanDistance(Float32List a, Float32List b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      final d = a[i] - b[i];
      sum += d * d;
    }
    return math.sqrt(sum);
  }
}

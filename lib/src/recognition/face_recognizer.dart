import 'dart:math' as math;
import 'dart:typed_data';

import '../models/face_embedding.dart';

/// Stateless face recognition using L2 distance.
class FaceRecognizer {
  /// Recognize a face from a query embedding against enrolled faces.
  /// Returns the person ID if a match is found, null otherwise.
  String? recognize(
    Float32List query,
    List<FaceEmbedding> enrolled,
    double threshold,
  ) {
    if (enrolled.isEmpty) {
      return null;
    }

    String? bestMatch;
    double minDistance = double.infinity;

    // Compare query against all enrolled embeddings
    for (final face in enrolled) {
      final distance = _euclideanDistance(query, face.embedding);

      if (distance < minDistance && distance < threshold) {
        minDistance = distance;
        bestMatch = face.personId;
      }
    }

    return bestMatch;
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

  /// Average multiple embeddings and L2-normalize the result.
  /// Useful for combining multiple samples of the same person.
  Float32List averageEmbeddings(List<Float32List> embeddings) {
    if (embeddings.isEmpty) {
      throw ArgumentError('Cannot average empty list of embeddings');
    }

    final len = embeddings.first.length;
    final avg = List.filled(len, 0.0);

    // Sum all embeddings
    for (final embedding in embeddings) {
      for (int i = 0; i < len; i++) {
        avg[i] += embedding[i];
      }
    }

    // Average
    for (int i = 0; i < len; i++) {
      avg[i] /= embeddings.length;
    }

    // L2 normalize
    final norm = math.sqrt(avg.fold(0.0, (sum, val) => sum + val * val));
    final normalized = avg.map((val) => val / norm).toList();

    return Float32List.fromList(normalized);
  }
}

import 'dart:math' as math;
import 'dart:typed_data';

import '../face_config.dart';

/// Manages enrollment of face samples.
/// Collects multiple samples per person and validates consistency.
class EnrollmentManager {
  final FaceConfig config;
  final Map<String, List<Float32List>> _samples = {};

  EnrollmentManager({required this.config});

  /// Add a sample for a person.
  /// Throws if the sample is too different from previous samples.
  void addSample(String personId, Float32List embedding) {
    _samples.putIfAbsent(personId, () => []);

    // Validate consistency with previous samples
    if (_samples[personId]!.isNotEmpty) {
      final last = _samples[personId]!.last;
      final dist = _euclideanDistance(last, embedding);

      // If distance is too large, reject the sample
      if (dist > 0.9) {
        throw Exception(
          'Face pose too different from previous samples. '
          'Distance: ${dist.toStringAsFixed(3)}. Try again.',
        );
      }
    }

    _samples[personId]!.add(embedding);

    // Use sliding window to keep only required number of samples
    if (_samples[personId]!.length > config.requiredEnrollmentSamples) {
      _samples[personId]!.removeAt(0);
    }
  }

  /// Check if enrollment is complete for a person.
  bool isEnrollmentComplete(String personId) {
    return _samples[personId]?.length == config.requiredEnrollmentSamples;
  }

  /// Get current sample count for a person.
  int getSampleCount(String personId) {
    return _samples[personId]?.length ?? 0;
  }

  /// Build final averaged and normalized embedding for a person.
  /// Returns null if enrollment is not complete.
  Float32List? buildFinalEmbedding(String personId) {
    if (!isEnrollmentComplete(personId)) {
      return null;
    }

    final samples = _samples[personId]!;
    return _averageAndNormalize(samples);
  }

  /// Get list of all enrolled person IDs.
  List<String> getEnrolledPersons() {
    return _samples.keys.where((id) => isEnrollmentComplete(id)).toList();
  }

  /// Clear enrollment data for a specific person.
  void clearPerson(String personId) {
    _samples.remove(personId);
  }

  /// Clear all enrollment data.
  void clear() {
    _samples.clear();
  }

  /// Average multiple embeddings and L2-normalize the result.
  Float32List _averageAndNormalize(List<Float32List> embeddings) {
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

  /// Calculate Euclidean distance between two embeddings.
  double _euclideanDistance(Float32List a, Float32List b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      final d = a[i] - b[i];
      sum += d * d;
    }
    return math.sqrt(sum);
  }
}

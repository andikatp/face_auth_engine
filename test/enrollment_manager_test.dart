import 'dart:typed_data';

import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:face_auth_engine/src/enrollment/enrollment_manager.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('EnrollmentManager', () {
    late EnrollmentManager manager;
    late FaceConfig config;

    setUp(() {
      config = const FaceConfig(requiredEnrollmentSamples: 3);
      manager = EnrollmentManager(config: config);
    });

    test('should add samples successfully', () {
      final emb1 = _createNormalizedEmbedding([0.1, 0.2, 0.3]);
      final emb2 = _createNormalizedEmbedding([0.15, 0.18, 0.32]);

      expect(() => manager.addSample('person1', emb1), returnsNormally);
      expect(manager.getSampleCount('person1'), 1);

      expect(() => manager.addSample('person1', emb2), returnsNormally);
      expect(manager.getSampleCount('person1'), 2);
    });

    test('should reject samples that are too different', () {
      final emb1 = _createNormalizedEmbedding([1.0, 0.0, 0.0]);
      final emb2 = _createNormalizedEmbedding([0.0, 1.0, 0.0]); // Far away

      manager.addSample('person1', emb1);

      expect(
        () => manager.addSample('person1', emb2),
        throwsA(isA<Exception>()),
      );
    });

    test('should track enrollment completion', () {
      final emb = _createNormalizedEmbedding([0.1, 0.2, 0.3]);

      expect(manager.isEnrollmentComplete('person1'), isFalse);

      manager.addSample('person1', emb);
      expect(manager.isEnrollmentComplete('person1'), isFalse);

      manager.addSample('person1', emb);
      expect(manager.isEnrollmentComplete('person1'), isFalse);

      manager.addSample('person1', emb);
      expect(manager.isEnrollmentComplete('person1'), isTrue);
    });

    test('should build final embedding when complete', () {
      // Use very similar embeddings (small perturbations)
      final emb1 = _createNormalizedEmbedding([0.6, 0.0, 0.8]);
      final emb2 = _createNormalizedEmbedding([0.61, 0.01, 0.79]);
      final emb3 = _createNormalizedEmbedding([0.59, 0.02, 0.81]);

      manager.addSample('person1', emb1);
      manager.addSample('person1', emb2);
      manager.addSample('person1', emb3);

      final finalEmbedding = manager.buildFinalEmbedding('person1');

      expect(finalEmbedding, isNotNull);
      expect(finalEmbedding!.length, 3);

      // Verify L2 normalization (norm should be close to 1.0)
      final norm = _calculateNorm(finalEmbedding);
      expect(norm, closeTo(1.0, 0.001));
    });

    test('should return null for incomplete enrollment', () {
      final emb = _createNormalizedEmbedding([0.1, 0.2, 0.3]);

      manager.addSample('person1', emb);
      final finalEmbedding = manager.buildFinalEmbedding('person1');

      expect(finalEmbedding, isNull);
    });

    test('should use sliding window for excess samples', () {
      final emb = _createNormalizedEmbedding([0.1, 0.2, 0.3]);

      // Add 5 samples (config requires 3)
      for (int i = 0; i < 5; i++) {
        manager.addSample('person1', emb);
      }

      // Should keep only 3 samples
      expect(manager.getSampleCount('person1'), 3);
    });

    test('should track multiple persons separately', () {
      final emb1 = _createNormalizedEmbedding([0.1, 0.2, 0.3]);
      final emb2 = _createNormalizedEmbedding([0.4, 0.5, 0.6]);

      manager.addSample('person1', emb1);
      manager.addSample('person2', emb2);

      expect(manager.getSampleCount('person1'), 1);
      expect(manager.getSampleCount('person2'), 1);
      expect(manager.getEnrolledPersons(), isEmpty);
    });

    test('should list all enrolled persons', () {
      final emb = _createNormalizedEmbedding([0.1, 0.2, 0.3]);

      // Complete enrollment for person1
      for (int i = 0; i < 3; i++) {
        manager.addSample('person1', emb);
      }

      // Partial enrollment for person2
      manager.addSample('person2', emb);

      final enrolled = manager.getEnrolledPersons();
      expect(enrolled, contains('person1'));
      expect(enrolled, isNot(contains('person2')));
    });

    test('should clear specific person data', () {
      final emb = _createNormalizedEmbedding([0.1, 0.2, 0.3]);

      manager.addSample('person1', emb);
      manager.addSample('person2', emb);

      manager.clearPerson('person1');

      expect(manager.getSampleCount('person1'), 0);
      expect(manager.getSampleCount('person2'), 1);
    });

    test('should clear all enrollment data', () {
      final emb = _createNormalizedEmbedding([0.1, 0.2, 0.3]);

      manager.addSample('person1', emb);
      manager.addSample('person2', emb);

      manager.clear();

      expect(manager.getSampleCount('person1'), 0);
      expect(manager.getSampleCount('person2'), 0);
      expect(manager.getEnrolledPersons(), isEmpty);
    });
  });
}

// Helper to create L2-normalized embedding
Float32List _createNormalizedEmbedding(List<double> values) {
  final norm = _calculateNorm(Float32List.fromList(values));
  return Float32List.fromList(values.map((v) => v / norm).toList());
}

double _calculateNorm(Float32List embedding) {
  double sum = 0.0;
  for (final val in embedding) {
    sum += val * val;
  }
  return sum > 0 ? embedding.fold(0.0, (sum, val) => sum + val * val) : 0.0;
}

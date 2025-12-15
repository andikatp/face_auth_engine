import 'dart:math' as math;
import 'dart:typed_data';

import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:face_auth_engine/src/recognition/face_recognizer.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('FaceRecognizer', () {
    late FaceRecognizer recognizer;

    setUp(() {
      recognizer = FaceRecognizer();
    });

    test('should return null for empty enrolled list', () {
      final query = _createNormalizedEmbedding([0.1, 0.2, 0.3]);
      final result = recognizer.recognize(query, [], 1.0);

      expect(result, isNull);
    });

    test('should recognize identical embedding', () {
      final emb = _createNormalizedEmbedding([0.6, 0.0, 0.8]);
      final enrolled = [FaceEmbedding(id: 'person1', embedding: emb)];

      final result = recognizer.recognize(emb, enrolled, 1.0);

      expect(result, 'person1');
    });

    test('should recognize very similar embedding', () {
      final emb1 = _createNormalizedEmbedding([0.6, 0.0, 0.8]);
      final emb2 = _createNormalizedEmbedding([0.61, 0.01, 0.79]);
      final enrolled = [FaceEmbedding(id: 'person1', embedding: emb1)];

      final result = recognizer.recognize(emb2, enrolled, 1.0);

      expect(result, 'person1');
    });

    test('should return null for dissimilar embedding', () {
      final emb1 = _createNormalizedEmbedding([1.0, 0.0, 0.0]);
      final emb2 = _createNormalizedEmbedding([0.0, 1.0, 0.0]);
      final enrolled = [FaceEmbedding(id: 'person1', embedding: emb1)];

      final result = recognizer.recognize(emb2, enrolled, 1.0);

      expect(result, isNull);
    });

    test('should select closest match among multiple persons', () {
      final query = _createNormalizedEmbedding([0.6, 0.0, 0.8]);
      final emb1 = _createNormalizedEmbedding([0.7, 0.0, 0.71]); // Further
      final emb2 = _createNormalizedEmbedding([0.61, 0.01, 0.79]); // Closer
      final enrolled = [
        FaceEmbedding(id: 'person1', embedding: emb1),
        FaceEmbedding(id: 'person2', embedding: emb2),
      ];

      final result = recognizer.recognize(query, enrolled, 1.0);

      expect(result, 'person2');
    });

    test('should respect threshold parameter', () {
      final emb1 = _createNormalizedEmbedding([0.6, 0.0, 0.8]);
      final emb2 = _createNormalizedEmbedding([0.55, 0.1, 0.75]);
      final enrolled = [FaceEmbedding(id: 'person1', embedding: emb1)];

      // With loose threshold
      final result1 = recognizer.recognize(emb2, enrolled, 1.0);
      expect(result1, 'person1');

      // With strict threshold
      final result2 = recognizer.recognize(emb2, enrolled, 0.05);
      expect(result2, isNull);
    });

    test('should average multiple embeddings correctly', () {
      final emb1 = _createNormalizedEmbedding([0.6, 0.0, 0.8]);
      final emb2 = _createNormalizedEmbedding([0.8, 0.0, 0.6]);
      final embeddings = [emb1, emb2];

      final averaged = recognizer.averageEmbeddings(embeddings);

      expect(averaged.length, 3);

      // Verify L2 normalization
      final norm = _calculateNorm(averaged);
      expect(norm, closeTo(1.0, 0.001));

      // Verify it's between the two inputs
      expect(averaged[0], greaterThan(0.6));
      expect(averaged[0], lessThan(0.8));
    });

    test('should throw on averaging empty list', () {
      expect(
        () => recognizer.averageEmbeddings([]),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('should calculate L2 distance correctly', () {
      // For normalized vectors pointing in same direction, distance = 0
      final emb1 = _createNormalizedEmbedding([1.0, 0.0, 0.0]);
      final emb2 = _createNormalizedEmbedding([1.0, 0.0, 0.0]);
      final enrolled = [FaceEmbedding(id: 'test', embedding: emb1)];

      final result = recognizer.recognize(emb2, enrolled, 0.001);
      expect(result, 'test'); // Should match with very strict threshold
    });

    test('should handle 192-dimensional embeddings', () {
      var emb1 = Float32List(192);
      var emb2 = Float32List(192);

      // Fill with pseudo-random normalized values
      for (int i = 0; i < 192; i++) {
        emb1[i] = math.sin(i * 0.1);
        emb2[i] = math.sin(i * 0.1);
      }

      // Normalize
      emb1 = _normalize(emb1);
      emb2 = _normalize(emb2);

      final enrolled = [FaceEmbedding(id: 'person1', embedding: emb1)];
      final result = recognizer.recognize(emb2, enrolled, 1.0);

      expect(result, 'person1');
    });
  });
}

Float32List _createNormalizedEmbedding(List<double> values) {
  return _normalize(Float32List.fromList(values));
}

Float32List _normalize(Float32List embedding) {
  final norm = _calculateNorm(embedding);
  if (norm == 0) return embedding;
  return Float32List.fromList(embedding.map((v) => v / norm).toList());
}

double _calculateNorm(Float32List embedding) {
  return math.sqrt(embedding.fold(0.0, (sum, val) => sum + val * val));
}

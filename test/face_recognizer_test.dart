import 'dart:math' as math;
import 'dart:typed_data';

import 'package:face_auth_engine/src/recognition/face_recognizer.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('FaceRecognizer', () {
    late FaceRecognizer recognizer;

    setUp(() {
      recognizer = FaceRecognizer();
    });

    test('verify should return true for identical embeddings', () {
      final emb = _createNormalizedEmbedding([0.6, 0.0, 0.8]);
      final result = recognizer.verify(emb, emb, 1.0);
      expect(result, isTrue);
    });

    test('verify should return true for very similar embeddings', () {
      final emb1 = _createNormalizedEmbedding([0.6, 0.0, 0.8]);
      final emb2 = _createNormalizedEmbedding([0.61, 0.01, 0.79]);
      final result = recognizer.verify(emb1, emb2, 1.0);
      expect(result, isTrue);
    });

    test('verify should return false for dissimilar embeddings', () {
      final emb1 = _createNormalizedEmbedding([1.0, 0.0, 0.0]);
      final emb2 = _createNormalizedEmbedding([0.0, 1.0, 0.0]);
      final result = recognizer.verify(emb1, emb2, 1.0);
      expect(result, isFalse);
    });

    test('verify should respect threshold parameter', () {
      final emb1 = _createNormalizedEmbedding([0.6, 0.0, 0.8]);
      final emb2 = _createNormalizedEmbedding([0.55, 0.1, 0.75]);

      // With loose threshold
      expect(recognizer.verify(emb1, emb2, 1.0), isTrue);

      // With strict threshold
      expect(recognizer.verify(emb1, emb2, 0.05), isFalse);
    });

    test('should calculate L2 distance correctly', () {
      // Distance between (1,0,0) and (1,0,0) is 0
      final emb1 = _createNormalizedEmbedding([1.0, 0.0, 0.0]);
      final emb2 = _createNormalizedEmbedding([1.0, 0.0, 0.0]);

      expect(recognizer.verify(emb1, emb2, 0.001), isTrue);
    });

    test('should handle 192-dimensional embeddings', () {
      var emb1 = Float32List(192);
      var emb2 = Float32List(192);

      for (int i = 0; i < 192; i++) {
        emb1[i] = math.sin(i * 0.1);
        emb2[i] = math.sin(i * 0.1);
      }

      emb1 = _normalize(emb1);
      emb2 = _normalize(emb2);

      expect(recognizer.verify(emb1, emb2, 1.0), isTrue);
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

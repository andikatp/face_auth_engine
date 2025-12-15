import 'dart:typed_data';

import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('FaceEmbedding', () {
    test('should create FaceEmbedding with id and embedding', () {
      final embedding = Float32List.fromList([0.1, 0.2, 0.3]);
      final face = FaceEmbedding(
        personId: 'test_person',
        embedding: embedding,
        version: '1.0',
      );

      expect(face.personId, 'test_person');
      expect(face.embedding, embedding);
      expect(face.version, '1.0');
    });

    test('should serialize to JSON correctly', () {
      final embedding = Float32List.fromList([0.1, 0.2, 0.3]);
      final face = FaceEmbedding(
        personId: 'test_person',
        embedding: embedding,
        version: '1.0',
      );

      final json = face.toJson();

      expect(json['personId'], 'test_person');
      expect((json['embedding'] as List).length, 3);
      expect((json['embedding'] as List)[0], closeTo(0.1, 0.0001));
      expect((json['embedding'] as List)[1], closeTo(0.2, 0.0001));
      expect((json['embedding'] as List)[2], closeTo(0.3, 0.0001));
    });

    test('should deserialize from JSON correctly', () {
      final json = {
        'personId': 'test_person',
        'embedding': [0.1, 0.2, 0.3],
        'version': '1.0',
      };

      final face = FaceEmbedding.fromJson(json);

      expect(face.personId, 'test_person');
      expect(face.embedding.length, 3);
      expect(face.embedding[0], closeTo(0.1, 0.0001));
      expect(face.embedding[1], closeTo(0.2, 0.0001));
      expect(face.embedding[2], closeTo(0.3, 0.0001));
      expect(face.version, '1.0');
    });

    test('should serialize and deserialize via JSON string', () {
      final embedding = Float32List.fromList([0.1, 0.2, 0.3]);
      final original = FaceEmbedding(
        personId: 'test_person',
        embedding: embedding,
        version: '1.0',
      );

      final jsonString = original.toJsonString();
      final decoded = FaceEmbedding.fromJsonString(jsonString);

      expect(decoded.personId, original.personId);
      expect(decoded.embedding.length, original.embedding.length);
    });

    test('should implement equality correctly', () {
      final emb1 = Float32List.fromList([0.1, 0.2, 0.3]);
      final emb2 = Float32List.fromList([0.1, 0.2, 0.3]);
      final emb3 = Float32List.fromList([0.1, 0.2, 0.4]);

      final face1 = FaceEmbedding(
        personId: 'person1',
        embedding: emb1,
        version: '1.0',
      );
      final face2 = FaceEmbedding(
        personId: 'person1',
        embedding: emb2,
        version: '1.0',
      );
      final face3 = FaceEmbedding(
        personId: 'person1',
        embedding: emb3,
        version: '1.0',
      );
      final face4 = FaceEmbedding(
        personId: 'person2',
        embedding: emb1,
        version: '1.0',
      );

      expect(face1, equals(face2)); // Same ID and embedding
      expect(face1, isNot(equals(face3))); // Different embedding
      expect(face1, isNot(equals(face4))); // Different ID
    });

    test('should have consistent toString', () {
      final embedding = Float32List(192); // MobileFaceNet size
      final face = FaceEmbedding(
        personId: 'test',
        embedding: embedding,
        version: '1.0',
      );

      expect(face.toString(), contains('test'));
      expect(face.toString(), contains('192D'));
      expect(face.toString(), contains('1.0'));
    });
  });
}

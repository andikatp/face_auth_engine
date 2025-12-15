import 'dart:convert';
import 'dart:typed_data';

/// Represents a face embedding for a person.
/// This is the primary data model for the package.
/// Apps are responsible for persisting this data.
class FaceEmbedding {
  /// Unique identifier for the person (e.g., name, user ID)
  final String id;

  /// 192-dimensional L2-normalized embedding vector
  final Float32List embedding;

  const FaceEmbedding({required this.id, required this.embedding});

  /// Convert to JSON for external persistence
  Map<String, dynamic> toJson() {
    return {'id': id, 'embedding': embedding.toList()};
  }

  /// Create from JSON
  factory FaceEmbedding.fromJson(Map<String, dynamic> json) {
    return FaceEmbedding(
      id: json['id'] as String,
      embedding: Float32List.fromList(
        (json['embedding'] as List).cast<double>(),
      ),
    );
  }

  /// Encode to JSON string
  String toJsonString() => json.encode(toJson());

  /// Decode from JSON string
  factory FaceEmbedding.fromJsonString(String source) {
    return FaceEmbedding.fromJson(json.decode(source) as Map<String, dynamic>);
  }

  @override
  String toString() =>
      'FaceEmbedding(id: $id, embedding: ${embedding.length}D)';

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! FaceEmbedding) return false;
    return id == other.id && _embeddingsEqual(embedding, other.embedding);
  }

  @override
  int get hashCode => id.hashCode ^ embedding.length.hashCode;

  bool _embeddingsEqual(Float32List a, Float32List b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if ((a[i] - b[i]).abs() > 1e-6) return false;
    }
    return true;
  }
}

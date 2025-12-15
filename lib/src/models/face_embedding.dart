import 'dart:convert';
import 'dart:typed_data';

/// Represents a face embedding for a person.
/// This is the primary data model for the package.
/// Apps are responsible for persisting this data.
class FaceEmbedding {
  /// Unique identifier for the person (e.g., name, user ID)
  final String personId;

  /// 192-dimensional L2-normalized embedding vector
  final Float32List embedding;

  /// Version of the embedding model
  final String version;

  const FaceEmbedding({
    required this.personId,
    required this.embedding,
    required this.version,
  });

  /// Convert to JSON for external persistence
  Map<String, dynamic> toJson() {
    return {
      'personId': personId,
      'embedding': embedding.toList(),
      'version': version,
    };
  }

  /// Create from JSON
  factory FaceEmbedding.fromJson(Map<String, dynamic> json) {
    return FaceEmbedding(
      personId: json['personId'] as String,
      embedding: Float32List.fromList(
        (json['embedding'] as List).cast<double>(),
      ),
      version: json['version'] as String,
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
      'FaceEmbedding(personId: $personId, embedding: ${embedding.length}D, version: $version)';

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! FaceEmbedding) return false;
    return personId == other.personId &&
        _embeddingsEqual(embedding, other.embedding) &&
        version == other.version;
  }

  @override
  int get hashCode =>
      personId.hashCode ^ embedding.length.hashCode ^ version.hashCode;

  bool _embeddingsEqual(Float32List a, Float32List b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if ((a[i] - b[i]).abs() > 1e-6) return false;
    }
    return true;
  }
}

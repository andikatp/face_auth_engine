/// Result of liveness detection analysis
class LivenessResult {
  /// True if live, false if spoof
  final bool isLive;

  /// Liveness probability (0.0 = definitely spoof, 1.0 = definitely live)
  final double score;

  /// Laplacian variance score (higher = sharper image)
  final int laplacian;

  /// Processing duration
  final Duration duration;

  const LivenessResult({
    required this.isLive,
    required this.score,
    required this.laplacian,
    required this.duration,
  });

  /// Spoof probability (convenience getter, inverse of score)
  double get spoofProb => 1.0 - score;

  @override
  String toString() {
    return 'LivenessResult('
        'isLive: $isLive, '
        'score: ${score.toStringAsFixed(3)}, '
        'laplacian: $laplacian, '
        'duration: ${duration.inMilliseconds}ms'
        ')';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other.runtimeType != runtimeType) return false;
    return other is LivenessResult &&
        other.isLive == isLive &&
        other.score == score &&
        other.laplacian == laplacian &&
        other.duration == duration;
  }

  @override
  int get hashCode => Object.hash(isLive, score, laplacian, duration);
}

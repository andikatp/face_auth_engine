import 'package:meta/meta.dart';

enum LivenessRejectionReason {
  /// Not rejected — either live or a normal spoof classification.
  none,

  /// Image is too dark to analyze reliably.
  /// Surface a user-facing message such as:
  /// "Poor lighting – please move to a brighter area."
  lowLight,

  /// Image is too blurry (Laplacian gate failed).
  /// Surface: "Image blurry – please hold still."
  blurry,

  /// Image is too bright/washed out.
  /// Surface: "Too much light – please move to a shaded area."
  overExposed,

  /// Model classified the face image as a spoof/attack.
  spoof,
}

/// Result of liveness detection analysis.
@immutable
class LivenessResult {
  const LivenessResult({
    required this.isLive,
    required this.score,
    required this.laplacian,
    required this.duration,
    this.rejectionReason = LivenessRejectionReason.none,
    this.brightness,
  });

  /// True if live, false if spoof or inconclusive.
  final bool isLive;

  /// Liveness probability (0.0 = definitely spoof, 1.0 = definitely live).
  /// Will be 0.0 for low-light or blurry rejections,
  /// not meaningful in those cases.
  final double score;

  /// Laplacian sharpness score (higher = sharper image).
  /// Null if the Laplacian gate was bypassed.
  final int? laplacian;

  /// Processing duration.
  final Duration duration;

  /// Why the result was rejected (or [LivenessRejectionReason.none] if not).
  final LivenessRejectionReason rejectionReason;

  /// Mean luma of the face crop [0–255]. Null when not computed.
  final double? brightness;

  /// Spoof probability (convenience getter, inverse of score).
  double get spoofProb => 1.0 - score;

  @override
  String toString() {
    return 'LivenessResult('
        'isLive: $isLive, '
        'score: ${score.toStringAsFixed(3)}, '
        'laplacian: ${laplacian ?? 'n/a'}, '
        'brightness: ${brightness?.toStringAsFixed(1) ?? 'n/a'}, '
        'rejectionReason: $rejectionReason, '
        'duration: ${duration.inMilliseconds}ms '
        ')';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is LivenessResult &&
        other.isLive == isLive &&
        other.score == score &&
        other.laplacian == laplacian &&
        other.duration == duration &&
        other.rejectionReason == rejectionReason &&
        other.brightness == brightness;
  }

  @override
  int get hashCode => Object.hash(
        isLive,
        score,
        laplacian,
        duration,
        rejectionReason,
        brightness,
      );
}

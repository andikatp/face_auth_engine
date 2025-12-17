/// Configuration for face authentication engine.
/// All values are immutable after construction.
class FaceConfig {
  /// Recognition threshold for L2-normalized embeddings.
  /// Lower values are more strict.
  /// Default: 1.0
  final double recognitionThreshold;

  /// Minimum face size (width/height) in pixels.
  /// Default: 80
  final int minFaceSize;

  /// Maximum allowed head roll angle in degrees.
  /// Default: 15.0
  final double maxRollAngle;

  const FaceConfig({
    this.recognitionThreshold = 1.0,
    this.minFaceSize = 80,
    this.maxRollAngle = 15.0,
  });

  /// Default configuration
  static const FaceConfig defaultConfig = FaceConfig();

  @override
  String toString() {
    return 'FaceConfig('
        'threshold: $recognitionThreshold, '
        'minSize: $minFaceSize, '
        'maxRoll: $maxRollAngle)';
  }
}

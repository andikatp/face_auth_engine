import 'dart:io';
import 'dart:math';
import 'dart:ui';

/// Agnostic interface for face detection.
/// Implement this in your app to provide face detection (e.g. using ML Kit).
abstract class FaceDetectorProvider {
  /// Detects a single face in the image and returns landmarks and bounding box
  /// Should throw an Exception if:
  /// - No face is detected
  /// - Multiple faces are detected
  /// - Required landmarks are missing
  Future<FaceDetectionResult> detectFace(File imageFile);

  void dispose();
}

/// Represents the points needed for face alignment and liveness detection.
class FaceLandmark {
  final Point<int> position;

  FaceLandmark(this.position);
}

/// Result of face detection.
class FaceDetectionResult {
  /// Must contain exactly 5 landmarks in this exact order:
  /// [leftEye, rightEye, noseBase, leftMouth, rightMouth]
  final List<FaceLandmark> landmarks;
  final Rect boundingBox;

  FaceDetectionResult({required this.landmarks, required this.boundingBox});
}

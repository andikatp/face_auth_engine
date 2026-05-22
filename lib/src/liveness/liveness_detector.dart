import 'dart:io';

import 'package:face_auth_engine/src/detection/face_detector_provider.dart';
import 'package:face_auth_engine/src/image/face_image_provider.dart';
import 'package:face_auth_engine/src/liveness/liveness_engine.dart';
import 'package:face_auth_engine/src/liveness/liveness_options.dart';
import 'package:face_auth_engine/src/liveness/liveness_result.dart';
import 'package:face_auth_engine/src/liveness/tflite_runner.dart';

/// Liveness detector for anti-spoofing face verification.
///
/// Analyzes face images to determine if they are from a live person
/// or a spoof attempt (photo, video, mask, etc.)
///
/// Usage:
/// ```dart
/// final detector = await LivenessDetector.create();
///
/// // Option A: Auto-detect and crop face from file
/// final result = await detector.detectLiveness(imageFile);
///
/// // Option B: Manually pass cropped face image
/// final result = await detector.analyze(faceImageBuffer);
///
/// print('Is Live: ${result.isLive}, Score: ${result.score}');
/// await detector.dispose();
/// ```
class LivenessDetector {
  const LivenessDetector._(
    this._engine,
    this._runner,
    this.faceDetector,
    this.imageProvider,
  );
  final LivenessEngine _engine;
  final TFLiteRunner _runner;
  final FaceDetectorProvider faceDetector;
  final FaceImageProvider imageProvider;

  /// Create a new liveness detector with specified options
  static Future<LivenessDetector> create({
    required FaceDetectorProvider faceDetector,
    required FaceImageProvider imageProvider,
    LivenessOptions options = const LivenessOptions(),
  }) async {
    final runner = await TFLiteRunner.create(
      useGpu: options.useGpu,
      threads: options.cpuThreads,
    );

    // Warm up the model for stable inference
    await runner.warmUp(times: options.warmUpIterations);

    final engine = LivenessEngine(runner, options, imageProvider);

    return LivenessDetector._(engine, runner, faceDetector, imageProvider);
  }

  /// Analyze a full frame image file.
  ///
  /// This method will:
  /// 1. Detect a face in the image (throws if no face or multiple faces)
  /// 2. Crop the face from the image
  /// 3. Run liveness analysis on the cropped face
  Future<LivenessResult> detectLiveness(File imageFile) async {
    // 1. Detect face (throws if no face or multiple faces)
    final faceResult = await faceDetector.detectFace(imageFile);

    // 2. Decode full image (only if face detected)
    final fullImage = await imageProvider.loadImage(imageFile);

    // 3. Crop face safely using bounding box from detection
    final box = faceResult.boundingBox;

    // Use rounding and clamp to image boundaries to prevent errors
    final left = box.left.round().clamp(0, fullImage.width - 1);
    final top = box.top.round().clamp(0, fullImage.height - 1);
    final right = box.right.round().clamp(0, fullImage.width);
    final bottom = box.bottom.round().clamp(0, fullImage.height);

    final width = (right - left).clamp(1, fullImage.width);
    final height = (bottom - top).clamp(1, fullImage.height);

    final croppedFace = await imageProvider.crop(
      fullImage,
      left,
      top,
      width,
      height,
    );

    // 4. Analyze
    return analyze(croppedFace);
  }

  /// Analyze a face image for liveness
  ///
  /// [face] should be a cropped face image (face detection
  /// should be done beforehand)
  ///
  /// Returns [LivenessResult] with:
  /// - `isLive`: true if detected as live person
  /// - `score`: liveness probability (0.0 = spoof, 1.0 = live)
  /// - `laplacian`: blur detection score
  /// - `duration`: processing time
  Future<LivenessResult> analyze(FaceImageBuffer face) => _engine.analyze(face);

  /// Dispose resources
  Future<void> dispose() async {
    await _runner.dispose();
  }
}

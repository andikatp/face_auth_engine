import 'dart:io';

import 'package:image/image.dart' as imglib;

import '../detection/face_detector_helper.dart';
import 'liveness_engine.dart';
import 'liveness_options.dart';
import 'liveness_result.dart';
import 'tflite_runner.dart';

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
/// final result = await detector.analyze(faceImage);
///
/// print('Is Live: ${result.isLive}, Score: ${result.score}');
/// await detector.dispose();
/// ```
class LivenessDetector {
  final LivenessEngine _engine;
  final TFLiteRunner _runner;
  final FaceDetectorHelper _faceDetector;

  const LivenessDetector._(this._engine, this._runner, this._faceDetector);

  /// Create a new liveness detector with specified options
  static Future<LivenessDetector> create({
    LivenessOptions options = const LivenessOptions(),
  }) async {
    final runner = await TFLiteRunner.create(
      useGpu: options.useGpu,
      threads: options.cpuThreads,
    );

    // Warm up the model for stable inference
    await runner.warmUp(times: options.warmUpIterations);

    final engine = LivenessEngine(runner, options);
    final pd = FaceDetectorHelper();

    return LivenessDetector._(engine, runner, pd);
  }

  /// Analyze a full frame image file.
  ///
  /// This method will:
  /// 1. Detect a face in the image (throws if no face or multiple faces)
  /// 2. Crop the face from the image
  /// 3. Run liveness analysis on the cropped face
  Future<LivenessResult> detectLiveness(File imageFile) async {
    // 1. Detect face
    final faceResult = await _faceDetector.detectFace(imageFile);

    // 2. Decode full image
    final bytes = await imageFile.readAsBytes();
    var fullImage = imglib.decodeImage(bytes);
    if (fullImage == null) throw Exception("Could not decode image file");

    // Ensure upright orientation so bounding box matches pixels
    fullImage = imglib.bakeOrientation(fullImage);

    // 3. Crop face using bounding box from detection
    final box = faceResult.boundingBox;
    final croppedFace = imglib.copyCrop(
      fullImage,
      x: box.left.toInt(),
      y: box.top.toInt(),
      width: box.width.toInt(),
      height: box.height.toInt(),
    );

    // 4. Analyze
    return analyze(croppedFace);
  }

  /// Analyze a face image for liveness
  ///
  /// [face] should be a cropped face image (face detection should be done beforehand)
  ///
  /// Returns [LivenessResult] with:
  /// - `isLive`: true if detected as live person
  /// - `score`: liveness probability (0.0 = spoof, 1.0 = live)
  /// - `laplacian`: blur detection score
  /// - `duration`: processing time
  Future<LivenessResult> analyze(imglib.Image face) => _engine.analyze(face);

  /// Dispose resources
  Future<void> dispose() async {
    await _runner.dispose();
    _faceDetector.dispose();
  }
}

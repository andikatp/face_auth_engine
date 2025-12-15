import 'dart:developer';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'alignment/face_alignment_helper.dart';
import 'detection/face_detector_helper.dart';
import 'enrollment/enrollment_manager.dart';
import 'face_config.dart';
import 'models/face_embedding.dart';
import 'recognition/face_recognizer.dart';

/// Main facade for face authentication engine.
/// Provides face detection, alignment, embedding extraction,
/// enrollment, and recognition capabilities.
///
/// This class is stateless regarding face data - all persistence
/// is delegated to the consuming application.
class FaceAuthEngine {
  final FaceConfig config;
  final String _modelPath = 'assets/models/mobilefacenet.tflite';

  Interpreter? _interpreter;
  FaceDetectorHelper? _faceDetector;
  late EnrollmentManager _enrollmentManager;
  late FaceRecognizer _recognizer;
  List<FaceEmbedding> _importedEmbeddings = [];

  FaceAuthEngine({FaceConfig? config})
    : config = config ?? FaceConfig.defaultConfig {
    _enrollmentManager = EnrollmentManager(config: this.config);
    _recognizer = FaceRecognizer();
    _initializeAsync();
  }

  /// Initialize TFLite model and face detector.
  Future<void> _initializeAsync() async {
    await _loadModel();
    _faceDetector = FaceDetectorHelper();
  }

  /// Load the TensorFlow Lite model.
  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(_modelPath);
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      log('MobileFaceNet model loaded successfully');
      log('Input shape: $inputShape');
      log('Output shape: $outputShape');
    } catch (e) {
      log('Error loading model: $e');
      rethrow;
    }
  }

  /// Extract face embedding from an image file.
  ///
  /// Steps:
  /// 1. Detect face and landmarks
  /// 2. Validate face quality
  /// 3. Align face to canonical pose
  /// 4. Normalize pixel values
  /// 5. Run through MobileFaceNet
  /// 6. L2-normalize output embedding
  ///
  /// Throws if:
  /// - No face detected
  /// - Multiple faces detected
  /// - Face quality too low
  /// - Model not initialized
  Future<Float32List> extractEmbedding(File imageFile) async {
    // Ensure model is loaded
    if (_interpreter == null) {
      log('Waiting for model to load...');
      await _loadModel();
    }

    _faceDetector ??= FaceDetectorHelper();

    // Read and decode image
    final imageBytes = await imageFile.readAsBytes();
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      throw Exception('Failed to decode image');
    }

    // Step 1: Detect face and extract landmarks
    final faceResult = await _faceDetector!.detectFace(imageFile);

    // Step 2: Validate face quality
    if (!_isFaceQualityAcceptable(faceResult)) {
      throw Exception('Face quality too low. Please face the camera.');
    }

    // Step 3: Align face using detected landmarks
    log('Aligning face using detected landmarks...');
    final alignedFace = FaceAlignmentHelper.alignFace(
      originalImage,
      faceResult.landmarks,
    );

    // Step 4: Normalize pixel values to [-1, 1] for MobileFaceNet
    final inputTensor = _preprocessImage(alignedFace);

    // Step 5: Run through model
    final embedding = _runModel(inputTensor);

    // Step 6: L2-normalize embedding
    final normalizedEmbedding = _l2Normalize(embedding);

    log(
      'Generated embedding, first 5 values: '
      '${normalizedEmbedding.sublist(0, 5)}',
    );

    return normalizedEmbedding;
  }

  /// Enroll a face sample for a person.
  /// Multiple samples are required (configured in FaceConfig).
  void enrollSample(String personId, Float32List embedding) {
    _enrollmentManager.addSample(personId, embedding);
  }

  /// Check if enrollment is complete for a person.
  bool isEnrollmentComplete(String personId) {
    return _enrollmentManager.isEnrollmentComplete(personId);
  }

  /// Get current sample count for a person.
  int getEnrollmentSampleCount(String personId) {
    return _enrollmentManager.getSampleCount(personId);
  }

  /// Build final averaged and normalized embedding for a person.
  /// Returns null if enrollment is not complete.
  Float32List? buildFinalEmbedding(String personId) {
    return _enrollmentManager.buildFinalEmbedding(personId);
  }

  /// Export all completed enrollments as FaceEmbedding objects.
  /// These can be persisted by the app and imported later.
  List<FaceEmbedding> exportAveragedEmbeddings() {
    final enrolled = _enrollmentManager.getEnrolledPersons();
    final embeddings = <FaceEmbedding>[];

    for (final personId in enrolled) {
      final embedding = _enrollmentManager.buildFinalEmbedding(personId);
      if (embedding != null) {
        embeddings.add(
          FaceEmbedding(
            personId: personId,
            embedding: embedding,
            version: '1.0',
          ),
        );
      }
    }

    return embeddings;
  }

  /// Import face embeddings for recognition.
  /// These embeddings will be used for subsequent recognize() calls.
  void importEmbeddings(List<FaceEmbedding> embeddings) {
    _importedEmbeddings = embeddings;
  }

  /// Recognize a face from a query embedding.
  /// Returns the person ID if a match is found, null otherwise.
  ///
  /// Uses the embeddings imported via importEmbeddings().
  String? recognize(Float32List query) {
    return _recognizer.recognize(
      query,
      _importedEmbeddings,
      config.recognitionThreshold,
    );
  }

  /// Get list of enrolled person IDs (from enrollment manager).
  List<String> getEnrolledPersons() {
    return _enrollmentManager.getEnrolledPersons();
  }

  /// Clear enrollment data for a specific person.
  void clearPerson(String personId) {
    _enrollmentManager.clearPerson(personId);
  }

  /// Clear all enrollment data.
  void clear() {
    _enrollmentManager.clear();
    _importedEmbeddings.clear();
  }

  /// Dispose resources.
  void dispose() {
    _interpreter?.close();
    _faceDetector?.dispose();
  }

  /// Preprocess aligned face image for MobileFaceNet.
  /// Normalizes pixel values to [-1, 1].
  Float32List _preprocessImage(img.Image alignedFace) {
    final inputImage = Float32List(112 * 112 * 3);
    int pixelIndex = 0;

    for (int y = 0; y < 112; y++) {
      for (int x = 0; x < 112; x++) {
        final pixel = alignedFace.getPixel(x, y);
        // Normalize to [-1, 1]: (pixel - 127.5) / 128.0
        inputImage[pixelIndex++] = (pixel.r.toInt() - 127.5) / 128.0;
        inputImage[pixelIndex++] = (pixel.g.toInt() - 127.5) / 128.0;
        inputImage[pixelIndex++] = (pixel.b.toInt() - 127.5) / 128.0;
      }
    }

    return inputImage;
  }

  /// Run MobileFaceNet model on preprocessed input.
  Float32List _runModel(Float32List inputImage) {
    if (_interpreter == null) {
      throw Exception('Model not initialized');
    }

    // Reshape for TFLite: [1, 112, 112, 3]
    final input = inputImage.reshape([1, 112, 112, 3]);

    // Prepare output: [1, 192]
    final output = List.filled(1 * 192, 0.0).reshape([1, 192]);

    // Run inference
    _interpreter!.run(input, output);

    // Flatten output
    final outputList = Float32List.fromList(
      output.expand<double>((e) => e).toList(),
    );

    return outputList;
  }

  /// L2-normalize an embedding vector.
  Float32List _l2Normalize(Float32List embedding) {
    final norm = math.sqrt(embedding.fold(0.0, (sum, val) => sum + val * val));
    return Float32List.fromList(embedding.map((val) => val / norm).toList());
  }

  /// Check if detected face meets quality requirements.
  bool _isFaceQualityAcceptable(FaceDetectionResult faceResult) {
    final boundingBox = faceResult.boundingBox;
    final landmarks = faceResult.landmarks;

    // Face size check
    if (boundingBox.width < config.minFaceSize ||
        boundingBox.height < config.minFaceSize) {
      log('Face too small: ${boundingBox.width}x${boundingBox.height}');
      return false;
    }

    // Eye-line tilt check (head roll approximation)
    final leftEye = landmarks[0].position; // First landmark is left eye
    final rightEye = landmarks[1].position; // Second is right eye

    final dx = rightEye.x - leftEye.x;
    final dy = rightEye.y - leftEye.y;
    final rollAngle = math.atan2(dy, dx) * 180 / math.pi;

    if (rollAngle.abs() > config.maxRollAngle) {
      log('Head roll too large: ${rollAngle.toStringAsFixed(1)}°');
      return false;
    }

    return true;
  }
}

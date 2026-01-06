import 'dart:developer' as developer;
import 'dart:io';
import 'dart:ui';

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

class FaceDetectionResult {
  final List<FaceLandmark> landmarks;
  final Rect boundingBox;

  FaceDetectionResult({required this.landmarks, required this.boundingBox});
}

class FaceDetectorHelper {
  late FaceDetector _faceDetector;

  FaceDetectorHelper() {
    _initializeDetector();
  }

  void _initializeDetector() {
    final options = FaceDetectorOptions(
      enableLandmarks: true,
      enableClassification: false,
      enableTracking: false,
      performanceMode: FaceDetectorMode.accurate,
    );
    _faceDetector = FaceDetector(options: options);
  }

  /// Detects a single face in the image and returns landmarks and bounding box
  /// Throws an exception if:
  /// - No face is detected
  /// - Multiple faces are detected
  /// - Required landmarks are missing
  Future<FaceDetectionResult> detectFace(File imageFile) async {
    final inputImage = InputImage.fromFile(imageFile);
    List<Face> faces = await _faceDetector.processImage(inputImage);

    // FIX: iOS images often have orientation issues (EXIF rotation) that ML Kit
    // might miss or handle poorly if the image is raw. If no face detected,
    // try to "bake" the orientation and retry.
    if (faces.isEmpty) {
      developer.log(
        'No face detected initially. Retrying with orientation fix...',
      );
      try {
        final bytes = await imageFile.readAsBytes();
        var image = img.decodeImage(bytes);

        if (image != null) {
          // Bake orientation ensures the pixels are physically rotated
          // according to EXIF, and EXIF is reset.
          image = img.bakeOrientation(image);

          final tempDir = Directory.systemTemp;
          final tempFile = File(
            '${tempDir.path}/temp_face_detect_${DateTime.now().millisecondsSinceEpoch}.jpg',
          );

          await tempFile.writeAsBytes(img.encodeJpg(image));

          final inputImageRetry = InputImage.fromFile(tempFile);
          faces = await _faceDetector.processImage(inputImageRetry);

          developer.log('Retry result: ${faces.length} faces found.');

          // Cleanup temp file
          if (await tempFile.exists()) {
            await tempFile.delete();
          }
        }
      } catch (e) {
        developer.log('Error during orientation fix retry: $e');
        // Ignore and fall through to original empty check
      }
    }

    if (faces.isEmpty) {
      throw Exception(
        'No face detected. Please ensure:\n'
        '• Your face is clearly visible\n'
        '• Good lighting conditions\n'
        '• Face is frontal (not side profile)',
      );
    }

    if (faces.length > 1) {
      throw Exception(
        'Multiple faces detected (${faces.length} faces).\n'
        'Please ensure only one person is in the image.',
      );
    }

    final face = faces.first;

    // Extract required 5-point landmarks
    final leftEye = face.landmarks[FaceLandmarkType.leftEye];
    final rightEye = face.landmarks[FaceLandmarkType.rightEye];
    final noseTip = face.landmarks[FaceLandmarkType.noseBase];
    final leftMouth = face.landmarks[FaceLandmarkType.leftMouth];
    final rightMouth = face.landmarks[FaceLandmarkType.rightMouth];

    // Validate that all required landmarks are detected
    final missingLandmarks = <String>[];
    if (leftEye == null) missingLandmarks.add('left eye');
    if (rightEye == null) missingLandmarks.add('right eye');
    if (noseTip == null) missingLandmarks.add('nose');
    if (leftMouth == null) missingLandmarks.add('left mouth');
    if (rightMouth == null) missingLandmarks.add('right mouth');

    if (missingLandmarks.isNotEmpty) {
      throw Exception(
        'Face landmarks not detected: ${missingLandmarks.join(", ")}.\n'
        'Please use a clear frontal face photo.',
      );
    }

    developer.log('Face detected successfully with all landmarks');
    developer.log('Bounding box: ${face.boundingBox}');
    developer.log('Left eye: ${leftEye!.position}');
    developer.log('Right eye: ${rightEye!.position}');
    developer.log('Nose: ${noseTip!.position}');
    developer.log('Left mouth: ${leftMouth!.position}');
    developer.log('Right mouth: ${rightMouth!.position}');

    return FaceDetectionResult(
      landmarks: [leftEye, rightEye, noseTip, leftMouth, rightMouth],
      boundingBox: face.boundingBox,
    );
  }

  void dispose() {
    _faceDetector.close();
  }
}

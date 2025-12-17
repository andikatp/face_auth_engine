import 'dart:developer';
import 'dart:io';

import 'package:face_auth_engine/face_auth_engine.dart';

/// Minimal example demonstrating Face Auth Engine usage.
///
/// This example shows:
/// 1. Converting an image to an embedding (stateless)
/// 2. Verifying a person against a known embedding
void main() async {
  log('=== Face Auth Engine Example ===\n');

  // 1. Initialize the engine
  final engine = FaceAuthEngine(config: FaceConfig(recognitionThreshold: 1.0));

  try {
    // 2. Extract embedding from an image file path
    log('--- Extraction Phase ---');
    // Replace with a real path to test
    final imagePath = 'path/to/person_photo.jpg';

    // Check if file exists in a real scenario
    final file = File(imagePath);
    if (!await file.exists()) {
      log('Example file not found at $imagePath. Skipping actual extraction.');
    } else {
      final embedding = await engine.convertToEmbedded(imagePath);
      log(
        'Extracted embedding: ${embedding.take(5)}... (len: ${embedding.length})',
      );

      // 3. Verify Person
      log('\n--- Verification Phase ---');
      final isSame = await engine.isThePersonTheSame(
        'path/to/another_photo.jpg',
        embedding,
      );
      log('Is the same person? $isSame');
    }

    log('\n--- Bulk Extraction Phase ---');
    // Demonstrate converting list of paths
    final paths = ['path/to/photo1.jpg', 'path/to/photo2.jpg'];
    // In a real app, ensure these exist
    if (await File(paths[0]).exists()) {
      final allEmbeddings = await engine.convertFromListToEmbedded(paths);
      log('Extracted ${allEmbeddings.length} embeddings');
    }
  } catch (e) {
    log('Error (expected if files map to nothing): $e');
  } finally {
    engine.dispose();
  }
}

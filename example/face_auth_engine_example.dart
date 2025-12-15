import 'dart:convert';
import 'dart:developer';
import 'dart:io';

import 'package:face_auth_engine/face_auth_engine.dart';

/// Minimal example demonstrating Face Auth Engine usage.
///
/// This example shows:
/// 1. Enrolling a person with multiple samples
/// 2. Exporting embeddings to JSON (simulated persistence)
/// 3. Importing embeddings from JSON
/// 4. Recognizing a face
void main() async {
  log('=== Face Auth Engine Example ===\n');

  // 1. Initialize the engine
  final engine = FaceAuthEngine(
    config: FaceConfig(recognitionThreshold: 1.0, requiredEnrollmentSamples: 5),
  );

  try {
    // 2. ENROLLMENT PHASE
    log('--- Enrollment Phase ---');
    await enrollPerson(engine, 'john_doe');

    // 3. EXPORT EMBEDDINGS (Simulated persistence)
    log('\n--- Export Phase ---');
    final embeddings = engine.exportAveragedEmbeddings();
    final jsonData = jsonEncode(embeddings.map((e) => e.toJson()).toList());
    log('Exported ${embeddings.length} embeddings to JSON');
    log('JSON preview: ${jsonData.substring(0, 100)}...\n');

    // 4. CLEAR ENGINE (Simulate app restart)
    log('--- Clearing engine (simulating app restart) ---');
    engine.clear();

    // 5. IMPORT EMBEDDINGS
    log('\n--- Import Phase ---');
    final importedEmbeddings = (jsonDecode(jsonData) as List)
        .map((json) => FaceEmbedding.fromJson(json))
        .toList();
    engine.importEmbeddings(importedEmbeddings);
    log('Imported ${importedEmbeddings.length} embeddings from JSON\n');

    // 6. RECOGNITION PHASE
    log('--- Recognition Phase ---');
    await recognizePerson(engine);
  } catch (e) {
    log('Error: $e');
  } finally {
    engine.dispose();
  }
}

/// Simulate enrolling a person with multiple samples.
/// In a real app, you would pick actual images.
Future<void> enrollPerson(FaceAuthEngine engine, String personId) async {
  log('Enrolling person: $personId');
  log('Required samples: ${engine.config.requiredEnrollmentSamples}\n');

  // In a real app, you would:
  // 1. Pick images using image_picker
  // 2. Extract embeddings from each image
  // 3. Add samples until enrollment is complete
  //
  // Example:
  // for (int i = 0; i < 5; i++) {
  //   final imageFile = await ImagePicker().pickImage(source: ImageSource.camera);
  //   if (imageFile != null) {
  //     final embedding = await engine.extractEmbedding(File(imageFile.path));
  //     engine.enrollSample(personId, embedding);
  //     log('Sample ${i + 1}/5 enrolled');
  //   }
  // }

  log('ℹ️  In a real app, you would:');
  log('   1. Pick 5 images of the person (camera/gallery)');
  log('   2. Extract embedding: await engine.extractEmbedding(imageFile)');
  log('   3. Add sample: engine.enrollSample(personId, embedding)');
  log('   4. Check completion: engine.isEnrollmentComplete(personId)');
  log('   5. Build final: engine.buildFinalEmbedding(personId)\n');

  log('✓ Enrollment simulation complete');
}

/// Simulate recognizing a person.
/// In a real app, you would pick an actual image.
Future<void> recognizePerson(FaceAuthEngine engine) async {
  log('Attempting to recognize a face...\n');

  // In a real app, you would:
  // 1. Pick an image
  // 2. Extract embedding
  // 3. Call recognize
  //
  // Example:
  // final imageFile = await ImagePicker().pickImage(source: ImageSource.camera);
  // if (imageFile != null) {
  //   final embedding = await engine.extractEmbedding(File(imageFile.path));
  //   final personId = engine.recognize(embedding);
  //   if (personId != null) {
  //     log('✓ Recognized: $personId');
  //   } else {
  //     log('✗ Unknown person');
  //   }
  // }

  log('ℹ️  In a real app, you would:');
  log('   1. Pick an image to recognize');
  log('   2. Extract embedding: await engine.extractEmbedding(imageFile)');
  log('   3. Recognize: final personId = engine.recognize(embedding)');
  log('   4. Handle result (personId is null if unknown)\n');

  log('✓ Recognition simulation complete');
}

/// Example: How to persist embeddings to a file
Future<void> saveEmbeddingsToFile(List<FaceEmbedding> embeddings) async {
  final file = File('face_embeddings.json');
  final jsonData = jsonEncode(embeddings.map((e) => e.toJson()).toList());
  await file.writeAsString(jsonData);
  log('✓ Saved embeddings to ${file.path}');
}

/// Example: How to load embeddings from a file
Future<List<FaceEmbedding>> loadEmbeddingsFromFile() async {
  final file = File('face_embeddings.json');
  final jsonData = await file.readAsString();
  final embeddings = (jsonDecode(jsonData) as List)
      .map((json) => FaceEmbedding.fromJson(json))
      .toList();
  log('✓ Loaded ${embeddings.length} embeddings from ${file.path}');
  return embeddings;
}

import 'dart:io';

import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:image/image.dart' as img;

/// Basic implementation of Image Provider using the standard package:image
class ImageProviderImpl implements FaceImageProvider {
  @override
  Future<FaceImageBuffer> loadImage(File file) async {
    final bytes = await file.readAsBytes();
    img.Image? decoded = img.decodeImage(bytes);
    if (decoded == null) throw Exception('Failed to decode image');

    // Fix orientation
    decoded = img.bakeOrientation(decoded);

    return _toBuffer(decoded);
  }

  @override
  Future<FaceImageBuffer> resize(FaceImageBuffer image, int width, int height) async {
    final originalImage = _fromBuffer(image);
    final resized = img.copyResize(originalImage, width: width, height: height, interpolation: img.Interpolation.linear);
    return _toBuffer(resized);
  }

  @override
  Future<FaceImageBuffer> crop(FaceImageBuffer image, int x, int y, int width, int height) async {
    final originalImage = _fromBuffer(image);
    final cropped = img.copyCrop(originalImage, x: x, y: y, width: width, height: height);
    return _toBuffer(cropped);
  }

  FaceImageBuffer _toBuffer(img.Image image) {
    final bytes = image.getBytes(order: img.ChannelOrder.rgb);
    return FaceImageBuffer(
      width: image.width,
      height: image.height,
      pixels: bytes,
    );
  }

  img.Image _fromBuffer(FaceImageBuffer buffer) {
    return img.Image.fromBytes(
      width: buffer.width,
      height: buffer.height,
      bytes: buffer.pixels.buffer,
      order: img.ChannelOrder.rgb,
    );
  }
}

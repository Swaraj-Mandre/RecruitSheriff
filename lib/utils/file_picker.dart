// lib/utils/file_picker.dart

import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';

class PickedFileData {
  final Uint8List bytes;
  final String name;

  PickedFileData(this.bytes, this.name);
}

/// Works for both web and desktop.
/// Returns file bytes and filename for upload.
Future<PickedFileData?> pickAnyFile() async {
  try {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      withData: true, // required for web
    );

    if (result != null && result.files.single.bytes != null) {
      return PickedFileData(result.files.single.bytes!, result.files.single.name);
    } else {
      return null;
    }
  } catch (e) {
    print("File picker error: $e");
    return null;
  }
}

// lib/services/api_service.dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';

class ApiService {
  // ✅ Base URL depends on platform:
  // Chrome / macOS / iOS → http://127.0.0.1:8080
  // Android Emulator → http://10.0.2.2:8080
  final String baseUrl = "http://127.0.0.1:8080";

  Future<Map<String, dynamic>> analyzeMultipleResumes(
      List<PlatformFile> files, String jobDescription) async {
    try {
      final uri = Uri.parse('$baseUrl/api/resume');
      var request = http.MultipartRequest('POST', uri);

      // 🧾 Add the job description
      request.fields['jobDescription'] = jobDescription;

      // 📁 Add all resume files
      for (var file in files) {
        // Safely read file bytes
        final fileBytes = file.bytes ?? await File(file.path!).readAsBytes();
        request.files.add(http.MultipartFile.fromBytes(
          'resumeFiles',
          fileBytes,
          filename: file.name,
        ));
      }

      // 🚀 Send request
      var response = await request.send();
      var responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return jsonDecode(responseBody);
      } else {
        throw Exception(
            "Server returned ${response.statusCode}: ${responseBody}");
      }
    } catch (e) {
      print("❌ Error in API call: $e");
      rethrow;
    }
  }
}

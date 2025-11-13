// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../services/api_service.dart' as service; // ✅ Alias to avoid import conflicts
import 'package:rescruit_ai/screens/result_screen.dart' ;


class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController jobController = TextEditingController();
  List<PlatformFile> pickedFiles = [];
  bool loading = false;

  // 📁 Pick multiple resumes
  Future<void> handleFilePick() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        allowMultiple: true,
        type: FileType.custom,
        allowedExtensions: ['pdf', 'docx', 'txt'],
        withData: true, // helps with Chrome/iOS
      );

      if (result != null && result.files.isNotEmpty) {
        setState(() => pickedFiles = result.files);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("${pickedFiles.length} resumes selected"),
            backgroundColor: Colors.green,
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("No files selected")),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error picking files: $e")),
      );
    }
  }

  // 🧠 Analyze all resumes
  Future<void> analyzeResumes() async {
    if (pickedFiles.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please select resumes first")),
      );
      return;
    }

    if (jobController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please enter a job description")),
      );
      return;
    }

    setState(() => loading = true);

    try {
      final api = service.ApiService(); // ✅ uses alias safely
      final response =
      await api.analyzeMultipleResumes(pickedFiles, jobController.text);

      if (!mounted) return;

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(response: response),
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text("Error analyzing resumes: $e"),
          backgroundColor: Colors.redAccent,
        ),
      );
    } finally {
      setState(() => loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        title: const Text("AI Resume Analyzer"),
        centerTitle: true,
        elevation: 1,
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
          child: Container(
            constraints: const BoxConstraints(maxWidth: 600),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  "Upload Multiple Resumes for ATS Evaluation",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  "AI will compare each resume with the job description, score them, and generate interview questions.",
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.grey.shade700, height: 1.4),
                ),
                const SizedBox(height: 24),

                // 🧾 Job Description Input
                const Text(
                  "Job Description",
                  style: TextStyle(fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: jobController,
                  maxLines: 4,
                  decoration: InputDecoration(
                    hintText: "Paste or type job description here...",
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
                const SizedBox(height: 20),

                // 📁 File Picker Button
                ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    backgroundColor: Colors.blueAccent,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  onPressed: handleFilePick,
                  icon: const Icon(Icons.upload_file),
                  label: const Text(
                    "Select Resumes",
                    style: TextStyle(fontSize: 16),
                  ),
                ),
                const SizedBox(height: 10),

                // 📋 Show Selected File Names
                if (pickedFiles.isNotEmpty)
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.blue.shade50,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: pickedFiles
                          .map((f) => Padding(
                        padding:
                        const EdgeInsets.symmetric(vertical: 2.0),
                        child: Text("• ${f.name}"),
                      ))
                          .toList(),
                    ),
                  ),

                const SizedBox(height: 30),

                // 🔄 Analyze Button or Loader
                loading
                    ? Column(
                  children: [
                    const CircularProgressIndicator(),
                    const SizedBox(height: 12),
                    Text(
                      "Analyzing ${pickedFiles.length} resumes...",
                      style: TextStyle(
                          fontSize: 15, color: Colors.grey.shade600),
                    ),
                  ],
                )
                    : ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    backgroundColor: Colors.black,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  onPressed: analyzeResumes,
                  child: const Text(
                    "Analyze All Resumes",
                    style: TextStyle(fontSize: 17),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

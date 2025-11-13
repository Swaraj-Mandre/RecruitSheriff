// lib/models/resume_response.dart

class ResumeResponse {
  final String name;
  final String age;
  final String appliedFor;
  final String description;
  final int score;
  final List<String> questions;

  ResumeResponse({
    required this.name,
    required this.age,
    required this.appliedFor,
    required this.description,
    required this.score,
    required this.questions,
  });

  factory ResumeResponse.fromJson(Map<String, dynamic> json) {
    final structured = json['structuredResume'] ?? {};
    final scoreData = json['score'] ?? {};
    final questionList = json['interviewQuestions'] ?? [];

    return ResumeResponse(
      name: structured['name'] ?? 'Unknown',
      age: structured['age'] ?? 'Not specified',
      appliedFor: structured['appliedFor'] ?? 'Unspecified Role',
      description: structured['description'] ?? 'No summary provided.',
      score: (scoreData['overall_score'] ?? 0).toInt(),
      questions: List<String>.from(
        questionList.map((q) => q['question'] ?? ''),
      ),
    );
  }
}

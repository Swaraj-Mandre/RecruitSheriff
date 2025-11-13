import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> response;
  const ResultScreen({super.key, required this.response});

  @override
  Widget build(BuildContext context) {
    // ✅ Safely parse response data
    final results = response['results'] as List<dynamic>? ?? [];

    // ✅ Prepare chart data
    final chartData = results
        .map((r) => _ChartData(
      r['file'] ?? 'Unknown',
      (r['score'] ?? 0).toDouble(),
    ))
        .toList();

    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        title: const Text("AI Analysis Results"),
        centerTitle: true,
        elevation: 1,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const SizedBox(height: 10),
            const Text(
              "ATS Score Comparison",
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),

            // 🥧 Pie Chart
            if (chartData.isNotEmpty)
              SfCircularChart(
                legend: const Legend(isVisible: true),
                series: <CircularSeries>[
                  PieSeries<_ChartData, String>(
                    dataSource: chartData,
                    xValueMapper: (_ChartData data, _) => data.file,
                    yValueMapper: (_ChartData data, _) => data.score,
                    dataLabelSettings:
                    const DataLabelSettings(isVisible: true),
                  ),
                ],
              )
            else
              const Padding(
                padding: EdgeInsets.all(16),
                child: Text("No analysis results available."),
              ),

            const SizedBox(height: 30),

            // 🧾 Candidate cards
            ...results.map((r) {
              final candidate = r['file'] ?? 'Unnamed Candidate';
              final score = r['score'] ?? 0;
              final questions =
              (r['interviewQuestions'] as List<dynamic>? ?? [])
                  .whereType<String>()
                  .toList();

              return Card(
                margin: const EdgeInsets.only(bottom: 20),
                elevation: 3,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        candidate,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        "ATS Score: $score / 100",
                        style: TextStyle(
                          fontSize: 15,
                          color: score >= 80
                              ? Colors.green
                              : score >= 50
                              ? Colors.orange
                              : Colors.redAccent,
                        ),
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        "AI-Suggested Interview Questions:",
                        style: TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 15,
                        ),
                      ),
                      const SizedBox(height: 8),
                      if (questions.isNotEmpty)
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: questions
                              .map(
                                (q) => Padding(
                              padding:
                              const EdgeInsets.symmetric(vertical: 2.0),
                              child: Text("• $q"),
                            ),
                          )
                              .toList(),
                        )
                      else
                        const Text("No questions generated."),
                    ],
                  ),
                ),
              );
            }),
          ],
        ),
      ),
    );
  }
}

// 📊 Helper class for chart data
class _ChartData {
  final String file;
  final double score;
  _ChartData(this.file, this.score);
}

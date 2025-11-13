import 'package:flutter/material.dart';

class ResumeCard extends StatelessWidget {
  final String name;
  final String score;

  const ResumeCard({super.key, required this.name, required this.score});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(8),
      child: ListTile(
        leading: const Icon(Icons.person),
        title: Text(name),
        subtitle: Text("Score: $score"),
      ),
    );
  }
}

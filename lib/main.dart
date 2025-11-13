import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const ResumeAIApp());
}

class ResumeAIApp extends StatelessWidget {
  const ResumeAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Resume AI',
      theme: ThemeData(
        primarySwatch: Colors.indigo,
        useMaterial3: true,
      ),
      debugShowCheckedModeBanner: false,
      home: const HomeScreen(),
    );
  }
}

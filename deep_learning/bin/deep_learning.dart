import 'dart:math';

import 'package:deep_learning/activation_functions.dart';
import 'package:deep_learning/deep_learning.dart' as deep_learning;

class Inputs {
  
  /// Valor do inpt
  final double value;

  /// Valor do peso
  final double weight;

  const Inputs({required this.value, required this.weight});

}
void main(List<String> arguments) {

  // Valores verdadeiros
  List<double> yTrue = [1,0,1,0];

  // Erros calculados
  List<double> yPred = [0.300, 0.020, 0.890, 0.320];

  print("mean absolute error ${mae(yTrue, yPred)}");
  print("mean squared error ${mse(yTrue, yPred)}");
  print(("root mean squared error: ${rmse(yTrue, yPred)}"));
  print("taxa de acerto (%): ${accuracy(yTrue, yPred)}");
}

double perceptron(List<Inputs> inputs){

  // Retorna o valor da soma
  return inputs.map((i) => i.value * i.weight).reduce((a, b) => a+b);
  
}

double sum(double value, double weight){
  return value * weight;
}

double mae(List<double> yTrue, List<double> yPred) {
  double sum = 0;

  for (int i = 0; i < yTrue.length; i++) {
    sum += (yTrue[i] - yPred[i]).abs();
  }

  return sum / yTrue.length;
}

double mse(List<double> yTrue, List<double> yPred) {
  double sum = 0;

  for (int i = 0; i < yTrue.length; i++) {
    double error = yTrue[i] - yPred[i];
    sum += error * error;
  }

  return sum / yTrue.length;
}

double rmse(List<double> yTrue, List<double> yPred) {
  double sum = 0;

  for (int i = 0; i < yTrue.length; i++) {
    double error = yTrue[i] - yPred[i];
    sum += error * error;
  }

  return sqrt(sum / yTrue.length);
}

double accuracy(List<double> yTrue, List<double> yPred) {
  int correct = 0;

  for (int i = 0; i < yTrue.length; i++) {
    if (yTrue[i] == yPred[i]) {
      correct++;
    }
  }

  return (correct / yTrue.length) * 100;
}
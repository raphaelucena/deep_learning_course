import 'dart:math';

/// Step function
int step(double sum) {
  return sum >= 1 ? 1 : 0;
}

/// Sigmoid
double sigmoid(double sum) {
  return 1 / (1 + exp(-sum));
}

/// Hyperbolic
double tahn(double sum){
  return (exp(sum) - exp(-sum)) /  (exp(sum) + exp(-sum));
}


/// ReLU
double relu(double sum) {
  return sum >= 0 ? sum : 0;
}

/// Softmax
List<double> softmax(List<double> x){

  // Calcula todos os exponentes
  final ex = x.map((e) => exp(e)).toList();

  // Suma todos os valores da lista
  final sum = ex.reduce((a, b) => a+b);

  // Divide cada elemento pela soma
  return ex.map((e) => e / sum).toList();
}

/// Linear
double linear(double sum) {
  return sum;
}

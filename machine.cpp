#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

class LinearRegression {
public:
    std::vector<double> weights;
    double bias;

    LinearRegression(int num_features) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0, 1);
        for (int i = 0; i < num_features; i++) {
            weights.push_back(dis(gen));
        }
        bias = dis(gen);
    }

    double predict(const std::vector<double>& input) {
        double sum = std::inner_product(input.begin(), input.end(), weights.begin(), 0.0);
        return sum + bias;
    }

    void train(const std::vector<std::vector<double>>& input, const std::vector<double>& target, double learning_rate, int num_iters) {
        for (int i = 0; i < num_iters; i++) {
            std::vector<double> predictions;
            for (int j = 0; j < input.size(); j++) {
                predictions.push_back(predict(input[j]));
            }
            std::vector<double> error = subtract_vectors(target, predictions);

            for (int j = 0; j < weights.size(); j++) {
                double gradient = std::inner_product(error.begin(), error.end(), input[j].begin(), 0.0) / input.size();
                weights[j] += gradient * learning_rate;
            }
            double b_gradient = std::accumulate(error.begin(), error.end(), 0.0) / input.size();
            bias += b_gradient * learning_rate;
        }
    }

private:
    std::vector<double> subtract_vectors(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> result;
        for (int i = 0; i < a.size(); i++) {
            result.push_back(a[i] - b[i]);
        }
        return result;
    }
};

int main() {
    std::vector<std::vector<double>> input = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    std::vector<double> target = {2, 3, 4, 5};
    LinearRegression model(2);
    model.train(input, target, 0.1, 1000);
    std::vector<double> test_input = {5, 6};
    std::cout << "Prediction: " << model.predict(test_input) << std::endl;
    return 0;
}

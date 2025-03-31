#pragma once

#include <vector>
#include <memory>
#include <random>
#include <string>
#include <map>
#include <Eigen/Dense>

class MLModel {
public:
    // Model types
    enum class ModelType {
        LinearRegression,
        RandomForest,
        GradientBoosting,
        NeuralNetwork,
        Ensemble
    };

    // Model configuration
    struct ModelConfig {
        ModelType type;
        size_t inputFeatures;
        size_t hiddenLayers;
        size_t neuronsPerLayer;
        double learningRate;
        size_t maxIterations;
        double regularization;
        bool useDropout;
        double dropoutRate;
        bool useBatchNormalization;
        std::string activationFunction;
    };

    // Training metrics
    struct TrainingMetrics {
        double trainLoss;
        double validationLoss;
        double trainAccuracy;
        double validationAccuracy;
        std::vector<double> lossHistory;
        std::vector<double> accuracyHistory;
        std::map<std::string, double> featureImportance;
    };

    // Feature importance
    struct FeatureImportance {
        std::string featureName;
        double importance;
        double confidence;
    };

    MLModel(const ModelConfig& config);
    
    // Core ML functions
    void Train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd Predict(const Eigen::MatrixXd& X) const;
    void Update(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    
    // Model management
    void SaveModel(const std::string& path) const;
    void LoadModel(const std::string& path);
    void ResetModel();
    
    // Feature engineering
    Eigen::MatrixXd PreprocessFeatures(const Eigen::MatrixXd& X) const;
    void UpdateFeatureScaling(const Eigen::MatrixXd& X);
    
    // Model evaluation
    TrainingMetrics GetTrainingMetrics() const;
    std::vector<FeatureImportance> GetFeatureImportance() const;
    double GetModelScore() const;
    
    // Hyperparameter optimization
    void OptimizeHyperparameters(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    ModelConfig GetOptimalConfig() const;
    
    // Ensemble methods
    void AddModel(std::unique_ptr<MLModel> model);
    void RemoveModel(size_t index);
    void UpdateWeights(const Eigen::VectorXd& y_true);
    
private:
    // Model components
    struct ModelComponents {
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        std::vector<Eigen::MatrixXd> batchNormParams;
        std::vector<Eigen::MatrixXd> dropoutMasks;
    };

    // Internal training helpers
    void InitializeModel();
    void ForwardPass(const Eigen::MatrixXd& X);
    void BackwardPass(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    void UpdateParameters();
    void ValidateModel(const Eigen::MatrixXd& X_val, const Eigen::VectorXd& y_val);
    
    // Optimization helpers
    void GradientDescent();
    void AdamOptimizer();
    void RMSpropOptimizer();
    
    // Regularization helpers
    void ApplyL1Regularization();
    void ApplyL2Regularization();
    void ApplyDropout();
    
    // Feature engineering helpers
    void NormalizeFeatures(Eigen::MatrixXd& X) const;
    void StandardizeFeatures(Eigen::MatrixXd& X) const;
    void HandleMissingValues(Eigen::MatrixXd& X) const;
    void RemoveOutliers(Eigen::MatrixXd& X) const;
    
    // Model evaluation helpers
    double CalculateLoss(const Eigen::VectorXd& y_pred, const Eigen::VectorXd& y_true) const;
    double CalculateAccuracy(const Eigen::VectorXd& y_pred, const Eigen::VectorXd& y_true) const;
    void UpdateMetrics(const Eigen::VectorXd& y_pred, const Eigen::VectorXd& y_true);
    
    // Hyperparameter optimization helpers
    void GridSearch(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    void RandomSearch(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    void BayesianOptimization(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    
    // Internal state
    ModelConfig config_;
    ModelComponents components_;
    TrainingMetrics metrics_;
    
    Eigen::MatrixXd featureScaling_;
    Eigen::VectorXd featureMeans_;
    Eigen::VectorXd featureStdDevs_;
    
    std::vector<std::unique_ptr<MLModel>> ensembleModels_;
    Eigen::VectorXd ensembleWeights_;
    
    std::mt19937 rng_;
    
    // Model parameters
    const double EPSILON = 1e-8;
    const double BETA1 = 0.9;
    const double BETA2 = 0.999;
    const size_t BATCH_SIZE = 32;
    const size_t VALIDATION_SPLIT = 0.2;
    
    // Feature engineering parameters
    const double OUTLIER_THRESHOLD = 3.0;
    const double MISSING_VALUE_THRESHOLD = 0.5;
    
    // Optimization parameters
    const size_t MAX_OPTIMIZATION_ITERATIONS = 100;
    const double LEARNING_RATE_DECAY = 0.95;
    const double MIN_LEARNING_RATE = 1e-6;
}; 
#pragma once

#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <cmath>
#include <random>
#include "Usings.h"
#include "MLModel.h"

class MarketImpactAnalyzer {
public:
    // Market impact metrics
    struct ImpactMetrics {
        double temporaryImpact;
        double permanentImpact;
        double totalImpact;
        double marketRegime;
        double liquidityFactor;
        double costFactor;
        double timeFactor;
        double volumeFactor;
        double spreadFactor;
        std::vector<double> historicalImpacts;
        MLModel::TrainingMetrics mlMetrics;
    };

    // Market regime classification
    enum class MarketRegime {
        Normal,
        Volatile,
        Illiquid,
        HighSpread,
        LowVolume,
        HighVolume,
        Unknown
    };

    // Impact model parameters
    struct ImpactParameters {
        double alpha;        // Temporary impact coefficient
        double beta;         // Permanent impact coefficient
        double gamma;        // Decay rate
        double delta;        // Volume sensitivity
        double epsilon;      // Time sensitivity
        double zeta;        // Spread sensitivity
        double eta;         // Liquidity sensitivity
        double theta;       // Market regime sensitivity
    };

    // Impact analysis results
    struct ImpactAnalysis {
        double expectedImpact;
        double confidenceInterval;
        double optimalExecutionTime;
        double optimalOrderSize;
        std::vector<double> impactDecomposition;
        std::vector<double> costDecomposition;
        std::vector<double> riskDecomposition;
        std::vector<MLModel::FeatureImportance> featureImportance;
    };

    MarketImpactAnalyzer();
    
    // Core impact analysis functions
    ImpactAnalysis AnalyzeImpact(const OrderPointer& order);
    void UpdateMarketData(const Price& currentPrice, double volume, 
                         double spread, double volatility);
    void UpdateExecutionData(const OrderPointer& order, double actualImpact);
    
    // Market regime analysis functions
    MarketRegime ClassifyMarketRegime() const;
    void UpdateMarketRegime();
    double CalculateRegimeFactor() const;
    
    // Impact model functions
    void CalibrateModel();
    void UpdateModel();
    double PredictImpact(const OrderPointer& order) const;
    double PredictTemporaryImpact(const OrderPointer& order) const;
    double PredictPermanentImpact(const OrderPointer& order) const;
    
    // Optimization functions
    void OptimizeExecutionStrategy(const OrderPointer& order);
    void OptimizeOrderSplitting(const OrderPointer& order);
    void OptimizeTiming(const OrderPointer& order);
    
    // Cost analysis functions
    double CalculateExecutionCost(const OrderPointer& order) const;
    double CalculateOpportunityCost(const OrderPointer& order) const;
    double CalculateMarketImpactCost(const OrderPointer& order) const;
    
    // Risk analysis functions
    double CalculateImpactRisk(const OrderPointer& order) const;
    double CalculateTimingRisk(const OrderPointer& order) const;
    double CalculateLiquidityRisk(const OrderPointer& order) const;
    
    // Reporting functions
    ImpactMetrics GetCurrentMetrics() const;
    std::vector<double> GetHistoricalImpacts() const;
    std::map<std::string, double> GetImpactDecomposition() const;
    std::vector<std::pair<std::string, double>> GetCostDecomposition() const;
    
private:
    // Internal impact calculation helpers
    double CalculateTemporaryImpact(const OrderPointer& order) const;
    double CalculatePermanentImpact(const OrderPointer& order) const;
    double CalculateTotalImpact(const OrderPointer& order) const;
    double CalculateLiquidityFactor() const;
    double CalculateCostFactor() const;
    double CalculateTimeFactor() const;
    double CalculateVolumeFactor() const;
    double CalculateSpreadFactor() const;
    
    // Market regime analysis helpers
    void AnalyzeVolatility();
    void AnalyzeLiquidity();
    void AnalyzeSpread();
    void AnalyzeVolume();
    void UpdateRegimeParameters();
    
    // Model calibration helpers
    void ExtractFeatures();
    void NormalizeFeatures();
    void TrainTemporaryImpactModel();
    void TrainPermanentImpactModel();
    void ValidateModels();
    
    // Optimization helpers
    void OptimizeParameters();
    void OptimizeExecutionTime();
    void OptimizeOrderSize();
    void OptimizeSplitting();
    
    // ML model helpers
    Eigen::MatrixXd CreateFeatureMatrix() const;
    Eigen::VectorXd CreateTargetVector() const;
    void InitializeMLModels();
    void UpdateMLModels();
    
    // Internal state
    ImpactMetrics metrics_;
    ImpactParameters parameters_;
    MarketRegime currentRegime_;
    
    std::vector<Price> priceHistory_;
    std::vector<double> volumeHistory_;
    std::vector<double> spreadHistory_;
    std::vector<double> volatilityHistory_;
    std::vector<double> impactHistory_;
    
    std::chrono::system_clock::time_point lastUpdate_;
    std::mt19937 rng_;
    
    // ML models
    std::unique_ptr<MLModel> temporaryImpactModel_;
    std::unique_ptr<MLModel> permanentImpactModel_;
    
    // Model parameters
    const size_t MIN_TRAINING_SAMPLES = 1000;
    const size_t VALIDATION_SPLIT = 0.2;
    const double LEARNING_RATE = 0.01;
    const size_t MAX_ITERATIONS = 1000;
    
    // Market regime thresholds
    const double VOLATILITY_THRESHOLD = 0.02;
    const double LIQUIDITY_THRESHOLD = 0.1;
    const double SPREAD_THRESHOLD = 0.01;
    const double VOLUME_THRESHOLD = 1000;
    
    // Impact model coefficients
    const double ALPHA_INITIAL = 0.1;
    const double BETA_INITIAL = 0.05;
    const double GAMMA_INITIAL = 0.5;
    const double DELTA_INITIAL = 0.3;
    const double EPSILON_INITIAL = 0.2;
    const double ZETA_INITIAL = 0.15;
    const double ETA_INITIAL = 0.25;
    const double THETA_INITIAL = 0.1;
}; 
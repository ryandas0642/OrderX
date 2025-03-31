#include "MarketImpactAnalyzer.h"
#include "Order.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>

MarketImpactAnalyzer::MarketImpactAnalyzer()
    : currentRegime_(MarketRegime::Normal)
    , rng_(std::random_device{}())
    , lastUpdate_(std::chrono::system_clock::now())
{
    // Initialize impact metrics
    metrics_.temporaryImpact = 0.0;
    metrics_.permanentImpact = 0.0;
    metrics_.totalImpact = 0.0;
    metrics_.marketRegime = 0.0;
    metrics_.liquidityFactor = 1.0;
    metrics_.costFactor = 1.0;
    metrics_.timeFactor = 1.0;
    metrics_.volumeFactor = 1.0;
    metrics_.spreadFactor = 1.0;
    
    // Initialize impact parameters
    parameters_.alpha = ALPHA_INITIAL;
    parameters_.beta = BETA_INITIAL;
    parameters_.gamma = GAMMA_INITIAL;
    parameters_.delta = DELTA_INITIAL;
    parameters_.epsilon = EPSILON_INITIAL;
    parameters_.zeta = ZETA_INITIAL;
    parameters_.eta = ETA_INITIAL;
    parameters_.theta = THETA_INITIAL;
    
    // Initialize ML models
    InitializeMLModels();
}

void MarketImpactAnalyzer::InitializeMLModels() {
    // Configure temporary impact model
    MLModel::ModelConfig tempConfig;
    tempConfig.modelType = MLModel::ModelType::NeuralNetwork;
    tempConfig.inputFeatures = {"price", "volume", "spread", "volatility", "time", "size"};
    tempConfig.hiddenLayers = {64, 32, 16};
    tempConfig.learningRate = LEARNING_RATE;
    tempConfig.maxIterations = MAX_ITERATIONS;
    tempConfig.regularization = 0.01;
    tempConfig.dropoutRate = 0.2;
    tempConfig.activationFunction = "relu";
    
    // Configure permanent impact model
    MLModel::ModelConfig permConfig;
    permConfig.modelType = MLModel::ModelType::GradientBoosting;
    permConfig.inputFeatures = {"price", "volume", "spread", "volatility", "time", "size", "temporary_impact"};
    permConfig.learningRate = LEARNING_RATE;
    permConfig.maxIterations = MAX_ITERATIONS;
    permConfig.regularization = 0.01;
    
    // Create models
    temporaryImpactModel_ = std::make_unique<MLModel>(tempConfig);
    permanentImpactModel_ = std::make_unique<MLModel>(permConfig);
}

MarketImpactAnalyzer::ImpactAnalysis MarketImpactAnalyzer::AnalyzeImpact(
    const OrderPointer& order) {
    ImpactAnalysis analysis;
    
    // Update market data
    UpdateMarketData(order->price, order->quantity, 
                    order->spread, order->volatility);
    
    // Calculate impact components
    analysis.expectedImpact = PredictImpact(order);
    analysis.confidenceInterval = CalculateImpactRisk(order);
    analysis.optimalExecutionTime = CalculateTimeFactor();
    analysis.optimalOrderSize = order->quantity * CalculateVolumeFactor();
    
    // Calculate decompositions
    analysis.impactDecomposition = {
        CalculateTemporaryImpact(order),
        CalculatePermanentImpact(order),
        CalculateTotalImpact(order)
    };
    
    analysis.costDecomposition = {
        CalculateExecutionCost(order),
        CalculateOpportunityCost(order),
        CalculateMarketImpactCost(order)
    };
    
    analysis.riskDecomposition = {
        CalculateImpactRisk(order),
        CalculateTimingRisk(order),
        CalculateLiquidityRisk(order)
    };
    
    // Get feature importance from ML models
    analysis.featureImportance = {
        temporaryImpactModel_->GetFeatureImportance(),
        permanentImpactModel_->GetFeatureImportance()
    };
    
    return analysis;
}

void MarketImpactAnalyzer::UpdateMarketData(const Price& currentPrice, double volume,
                                           double spread, double volatility) {
    // Update historical data
    priceHistory_.push_back(currentPrice);
    volumeHistory_.push_back(volume);
    spreadHistory_.push_back(spread);
    volatilityHistory_.push_back(volatility);
    
    // Keep only recent history
    const size_t maxHistory = 1000;
    if (priceHistory_.size() > maxHistory) {
        priceHistory_.erase(priceHistory_.begin());
        volumeHistory_.erase(volumeHistory_.begin());
        spreadHistory_.erase(spreadHistory_.begin());
        volatilityHistory_.erase(volatilityHistory_.begin());
    }
    
    // Update market regime
    UpdateMarketRegime();
    
    // Update ML models if enough data
    if (priceHistory_.size() >= MIN_TRAINING_SAMPLES) {
        UpdateMLModels();
    }
    
    lastUpdate_ = std::chrono::system_clock::now();
}

void MarketImpactAnalyzer::UpdateExecutionData(const OrderPointer& order, 
                                              double actualImpact) {
    impactHistory_.push_back(actualImpact);
    if (impactHistory_.size() > MIN_TRAINING_SAMPLES) {
        impactHistory_.erase(impactHistory_.begin());
    }
    
    // Update model with new data
    UpdateModel();
}

MarketImpactAnalyzer::MarketRegime MarketImpactAnalyzer::ClassifyMarketRegime() const {
    if (volatilityHistory_.empty()) return MarketRegime::Unknown;
    
    double avgVolatility = std::accumulate(volatilityHistory_.begin(),
                                         volatilityHistory_.end(), 0.0) /
                          volatilityHistory_.size();
    
    double avgSpread = std::accumulate(spreadHistory_.begin(),
                                     spreadHistory_.end(), 0.0) /
                      spreadHistory_.size();
    
    double avgVolume = std::accumulate(volumeHistory_.begin(),
                                     volumeHistory_.end(), 0.0) /
                      volumeHistory_.size();
    
    if (avgVolatility > VOLATILITY_THRESHOLD) {
        return MarketRegime::Volatile;
    }
    
    if (avgSpread > SPREAD_THRESHOLD) {
        return MarketRegime::HighSpread;
    }
    
    if (avgVolume < VOLUME_THRESHOLD) {
        return MarketRegime::LowVolume;
    }
    
    if (avgVolume > VOLUME_THRESHOLD * 10) {
        return MarketRegime::HighVolume;
    }
    
    return MarketRegime::Normal;
}

void MarketImpactAnalyzer::UpdateMarketRegime() {
    currentRegime_ = ClassifyMarketRegime();
    metrics_.marketRegime = static_cast<double>(currentRegime_);
}

double MarketImpactAnalyzer::CalculateRegimeFactor() const {
    switch (currentRegime_) {
        case MarketRegime::Normal:
            return 1.0;
        case MarketRegime::Volatile:
            return 1.5;
        case MarketRegime::Illiquid:
            return 2.0;
        case MarketRegime::HighSpread:
            return 1.8;
        case MarketRegime::LowVolume:
            return 1.6;
        case MarketRegime::HighVolume:
            return 0.8;
        default:
            return 1.0;
    }
}

void MarketImpactAnalyzer::CalibrateModel() {
    if (impactHistory_.size() < MIN_TRAINING_SAMPLES) return;
    
    ExtractFeatures();
    NormalizeFeatures();
    
    // Split data into training and validation sets
    size_t splitIndex = static_cast<size_t>(impactHistory_.size() * (1 - VALIDATION_SPLIT));
    
    // Train models
    TrainTemporaryImpactModel();
    TrainPermanentImpactModel();
    
    // Validate models
    ValidateModels();
}

void MarketImpactAnalyzer::UpdateModel() {
    // Add new data points to training set
    ExtractFeatures();
    
    // Retrain models with updated data
    CalibrateModel();
}

double MarketImpactAnalyzer::PredictImpact(const OrderPointer& order) const {
    // Create feature vector for prediction
    Eigen::VectorXd features(6);
    features(0) = order->price.value;
    features(1) = order->quantity;
    features(2) = order->spread;
    features(3) = order->volatility;
    features(4) = 1.0; // current time
    features(5) = order->quantity / std::accumulate(volumeHistory_.begin(), 
                                                  volumeHistory_.end(), 0.0);
    
    // Get temporary impact prediction
    double tempImpact = temporaryImpactModel_->Predict(features);
    
    // Create features for permanent impact prediction
    Eigen::VectorXd permFeatures(7);
    permFeatures.head(6) = features;
    permFeatures(6) = tempImpact;
    
    // Get permanent impact prediction
    double permImpact = permanentImpactModel_->Predict(permFeatures);
    
    return tempImpact + permImpact;
}

void MarketImpactAnalyzer::OptimizeExecutionStrategy(const OrderPointer& order) {
    OptimizeParameters();
    OptimizeExecutionTime();
    OptimizeOrderSize();
    OptimizeSplitting();
}

void MarketImpactAnalyzer::OptimizeOrderSplitting(const OrderPointer& order) {
    // Implementation would optimize order splitting strategy
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::OptimizeTiming(const OrderPointer& order) {
    // Implementation would optimize execution timing
    // This is a placeholder for actual implementation
}

double MarketImpactAnalyzer::CalculateExecutionCost(const OrderPointer& order) const {
    double size = order->GetRemainingQuantity();
    double price = order->GetPrice();
    double spread = spreadHistory_.empty() ? 0 : spreadHistory_.back();
    
    return size * price * spread / 2;
}

double MarketImpactAnalyzer::CalculateOpportunityCost(const OrderPointer& order) const {
    double size = order->GetRemainingQuantity();
    double price = order->GetPrice();
    double volatility = volatilityHistory_.empty() ? 0 : volatilityHistory_.back();
    
    return size * price * volatility;
}

double MarketImpactAnalyzer::CalculateMarketImpactCost(const OrderPointer& order) const {
    return PredictImpact(order);
}

double MarketImpactAnalyzer::CalculateImpactRisk(const OrderPointer& order) const {
    double impact = PredictImpact(order);
    double volatility = volatilityHistory_.empty() ? 0 : volatilityHistory_.back();
    
    return impact * volatility;
}

double MarketImpactAnalyzer::CalculateTimingRisk(const OrderPointer& order) const {
    double volatility = volatilityHistory_.empty() ? 0 : volatilityHistory_.back();
    double timeFactor = CalculateTimeFactor();
    
    return volatility * timeFactor;
}

double MarketImpactAnalyzer::CalculateLiquidityRisk(const OrderPointer& order) const {
    double size = order->GetRemainingQuantity();
    double volume = volumeHistory_.empty() ? 0 : volumeHistory_.back();
    
    return size / volume;
}

MarketImpactAnalyzer::ImpactMetrics MarketImpactAnalyzer::GetCurrentMetrics() const {
    return metrics_;
}

std::vector<double> MarketImpactAnalyzer::GetHistoricalImpacts() const {
    return impactHistory_;
}

std::map<std::string, double> MarketImpactAnalyzer::GetImpactDecomposition() const {
    std::map<std::string, double> decomposition;
    decomposition["Temporary Impact"] = metrics_.temporaryImpact;
    decomposition["Permanent Impact"] = metrics_.permanentImpact;
    decomposition["Liquidity Factor"] = metrics_.liquidityFactor;
    decomposition["Cost Factor"] = metrics_.costFactor;
    decomposition["Time Factor"] = metrics_.timeFactor;
    decomposition["Volume Factor"] = metrics_.volumeFactor;
    decomposition["Spread Factor"] = metrics_.spreadFactor;
    return decomposition;
}

std::vector<std::pair<std::string, double>> MarketImpactAnalyzer::GetCostDecomposition() const {
    std::vector<std::pair<std::string, double>> decomposition;
    decomposition.emplace_back("Execution Cost", metrics_.costFactor);
    decomposition.emplace_back("Market Impact", metrics_.temporaryImpact + metrics_.permanentImpact);
    decomposition.emplace_back("Opportunity Cost", metrics_.timeFactor);
    return decomposition;
}

double MarketImpactAnalyzer::CalculateTemporaryImpact(const OrderPointer& order) const {
    double size = order->GetRemainingQuantity();
    double price = order->GetPrice();
    double volume = volumeHistory_.empty() ? 0 : volumeHistory_.back();
    double spread = spreadHistory_.empty() ? 0 : spreadHistory_.back();
    
    return parameters_.alpha * std::sqrt(size / volume) * price * spread;
}

double MarketImpactAnalyzer::CalculatePermanentImpact(const OrderPointer& order) const {
    double size = order->GetRemainingQuantity();
    double price = order->GetPrice();
    double volume = volumeHistory_.empty() ? 0 : volumeHistory_.back();
    
    return parameters_.beta * (size / volume) * price;
}

double MarketImpactAnalyzer::CalculateTotalImpact(const OrderPointer& order) const {
    return CalculateTemporaryImpact(order) + CalculatePermanentImpact(order);
}

double MarketImpactAnalyzer::CalculateLiquidityFactor() const {
    if (volumeHistory_.empty()) return 0;
    
    double avgVolume = std::accumulate(volumeHistory_.begin(),
                                     volumeHistory_.end(), 0.0) /
                      volumeHistory_.size();
    
    double currentVolume = volumeHistory_.back();
    
    return currentVolume / avgVolume;
}

double MarketImpactAnalyzer::CalculateCostFactor() const {
    if (spreadHistory_.empty()) return 0;
    
    double avgSpread = std::accumulate(spreadHistory_.begin(),
                                     spreadHistory_.end(), 0.0) /
                      spreadHistory_.size();
    
    double currentSpread = spreadHistory_.back();
    
    return currentSpread / avgSpread;
}

double MarketImpactAnalyzer::CalculateTimeFactor() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&time);
    
    // Higher impact during market open and close
    double hour = tm->tm_hour + tm->tm_min / 60.0;
    if (hour < 9.5 || hour > 16.0) {
        return 1.5;
    }
    
    return 1.0;
}

double MarketImpactAnalyzer::CalculateVolumeFactor() const {
    if (volumeHistory_.empty()) return 0;
    
    double avgVolume = std::accumulate(volumeHistory_.begin(),
                                     volumeHistory_.end(), 0.0) /
                      volumeHistory_.size();
    
    double currentVolume = volumeHistory_.back();
    
    return currentVolume / avgVolume;
}

double MarketImpactAnalyzer::CalculateSpreadFactor() const {
    if (spreadHistory_.empty()) return 0;
    
    double avgSpread = std::accumulate(spreadHistory_.begin(),
                                     spreadHistory_.end(), 0.0) /
                      spreadHistory_.size();
    
    double currentSpread = spreadHistory_.back();
    
    return currentSpread / avgSpread;
}

void MarketImpactAnalyzer::AnalyzeVolatility() {
    // Implementation would analyze market volatility
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::AnalyzeLiquidity() {
    // Implementation would analyze market liquidity
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::AnalyzeSpread() {
    // Implementation would analyze market spread
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::AnalyzeVolume() {
    // Implementation would analyze market volume
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::UpdateRegimeParameters() {
    // Implementation would update regime-specific parameters
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::ExtractFeatures() {
    // Implementation would extract features from market data
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::NormalizeFeatures() {
    // Implementation would normalize feature vectors
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::TrainTemporaryImpactModel() {
    // Implementation would train temporary impact prediction model
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::TrainPermanentImpactModel() {
    // Implementation would train permanent impact prediction model
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::ValidateModels() {
    // Implementation would validate trained models
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::OptimizeParameters() {
    // Implementation would optimize model parameters
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::OptimizeExecutionTime() {
    // Implementation would optimize execution timing
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::OptimizeOrderSize() {
    // Implementation would optimize order size
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::OptimizeSplitting() {
    // Implementation would optimize order splitting
    // This is a placeholder for actual implementation
}

void MarketImpactAnalyzer::UpdateMLModels() {
    // Create feature matrix and target vector
    Eigen::MatrixXd features = CreateFeatureMatrix();
    Eigen::VectorXd targets = CreateTargetVector();
    
    // Train temporary impact model
    temporaryImpactModel_->Train(features, targets);
    
    // Create features for permanent impact model
    Eigen::MatrixXd permFeatures = features;
    Eigen::VectorXd tempImpacts = temporaryImpactModel_->Predict(features);
    permFeatures.conservativeResize(permFeatures.rows(), permFeatures.cols() + 1);
    permFeatures.col(permFeatures.cols() - 1) = tempImpacts;
    
    // Train permanent impact model
    permanentImpactModel_->Train(permFeatures, targets);
    
    // Update metrics
    metrics_.mlMetrics = temporaryImpactModel_->GetTrainingMetrics();
}

Eigen::MatrixXd MarketImpactAnalyzer::CreateFeatureMatrix() const {
    const size_t nSamples = priceHistory_.size();
    const size_t nFeatures = 6; // price, volume, spread, volatility, time, size
    
    Eigen::MatrixXd features(nSamples, nFeatures);
    
    for (size_t i = 0; i < nSamples; ++i) {
        features(i, 0) = priceHistory_[i].value;
        features(i, 1) = volumeHistory_[i];
        features(i, 2) = spreadHistory_[i];
        features(i, 3) = volatilityHistory_[i];
        features(i, 4) = static_cast<double>(i) / nSamples; // normalized time
        features(i, 5) = volumeHistory_[i] / std::accumulate(volumeHistory_.begin(), 
                                                           volumeHistory_.end(), 0.0);
    }
    
    return features;
}

Eigen::VectorXd MarketImpactAnalyzer::CreateTargetVector() const {
    const size_t nSamples = impactHistory_.size();
    Eigen::VectorXd targets(nSamples);
    
    for (size_t i = 0; i < nSamples; ++i) {
        targets(i) = impactHistory_[i];
    }
    
    return targets;
} 
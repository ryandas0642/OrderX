#include "RiskManager.h"
#include "Order.h"
#include "Trade.h"
#include <algorithm>
#include <numeric>
#include <cmath>

RiskManager::RiskManager(const RiskLimits& limits)
    : limits_(limits)
    , rng_(std::random_device{}())
{
    // Initialize risk metrics
    metrics_.currentPositionValue = 0;
    metrics_.currentDrawdown = 0;
    metrics_.dailyPnL = 0;
    metrics_.currentExposure = 0;
    metrics_.currentLeverage = 0;
    metrics_.currentConcentration = 0;
    metrics_.currentCorrelation = 0;
    metrics_.currentVolatility = 0;
    metrics_.var95 = 0;
    metrics_.cvar95 = 0;
    
    // Initialize risk factor weights
    weights_.marketRisk = 0.3;
    weights_.creditRisk = 0.2;
    weights_.liquidityRisk = 0.15;
    weights_.operationalRisk = 0.1;
    weights_.counterpartyRisk = 0.1;
    weights_.concentrationRisk = 0.05;
    weights_.correlationRisk = 0.05;
    weights_.tailRisk = 0.02;
    weights_.systemicRisk = 0.02;
    weights_.regulatoryRisk = 0.01;
}

bool RiskManager::CheckOrderRisk(const OrderPointer& order) {
    // Check order size against limits
    if (order->GetRemainingQuantity() > limits_.maxOrderSize) {
        return false;
    }
    
    // Check position value after order
    Price newPositionValue = metrics_.currentPositionValue + 
        (order->GetSide() == Side::Buy ? 1 : -1) * 
        order->GetRemainingQuantity() * order->GetPrice();
    
    if (std::abs(newPositionValue) > limits_.maxPositionValue) {
        return false;
    }
    
    // Check leverage
    double newLeverage = std::abs(newPositionValue) / 
        (metrics_.currentPositionValue + order->GetRemainingQuantity() * order->GetPrice());
    
    if (newLeverage > limits_.maxLeverage) {
        return false;
    }
    
    return true;
}

void RiskManager::UpdatePosition(const Trade& trade) {
    // Update position value
    metrics_.currentPositionValue += 
        (trade.GetSide() == Side::Buy ? 1 : -1) * 
        trade.GetQuantity() * trade.GetPrice();
    
    // Update daily PnL
    metrics_.dailyPnL += 
        (trade.GetSide() == Side::Buy ? 1 : -1) * 
        trade.GetQuantity() * (trade.GetPrice() - trade.GetAveragePrice());
    
    // Update historical PnL
    pnlHistory_.push_back(metrics_.dailyPnL);
    if (pnlHistory_.size() > HISTORICAL_WINDOW) {
        pnlHistory_.erase(pnlHistory_.begin());
    }
    
    // Update drawdown
    Price peak = *std::max_element(pnlHistory_.begin(), pnlHistory_.end());
    metrics_.currentDrawdown = peak - metrics_.dailyPnL;
    
    // Update exposure
    metrics_.currentExposure = std::abs(metrics_.currentPositionValue);
    
    // Update concentration
    metrics_.currentConcentration = metrics_.currentPositionValue / 
        (limits_.maxPositionValue + 1e-10);
    
    lastUpdate_ = std::chrono::system_clock::now();
}

void RiskManager::UpdateMarketData(const Price& currentPrice) {
    priceHistory_.push_back(currentPrice);
    if (priceHistory_.size() > HISTORICAL_WINDOW) {
        priceHistory_.erase(priceHistory_.begin());
    }
    
    // Calculate volatility
    if (priceHistory_.size() > 1) {
        std::vector<Price> returns;
        for (size_t i = 1; i < priceHistory_.size(); ++i) {
            returns.push_back(std::log(priceHistory_[i] / priceHistory_[i-1]));
        }
        
        double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double variance = 0;
        for (double r : returns) {
            double diff = r - mean;
            variance += diff * diff;
        }
        variance /= returns.size();
        
        metrics_.currentVolatility = std::sqrt(variance * 252); // Annualized volatility
    }
}

bool RiskManager::IsRiskLimitBreached() const {
    return metrics_.currentDrawdown > limits_.maxDrawdown ||
           metrics_.dailyPnL < -limits_.maxDailyLoss ||
           metrics_.currentExposure > limits_.maxExposure ||
           metrics_.currentLeverage > limits_.maxLeverage ||
           metrics_.currentConcentration > limits_.maxConcentration ||
           metrics_.currentVolatility > limits_.maxVolatility;
}

void RiskManager::CalculateVaR() {
    if (pnlHistory_.empty()) return;
    
    std::vector<Price> sortedPnL = pnlHistory_;
    std::sort(sortedPnL.begin(), sortedPnL.end());
    
    size_t varIndex = static_cast<size_t>((1 - VAR_CONFIDENCE_LEVEL) * sortedPnL.size());
    metrics_.var95 = -sortedPnL[varIndex]; // Convert to positive value
    
    // Calculate CVaR
    double sum = 0;
    size_t count = 0;
    for (size_t i = 0; i < varIndex; ++i) {
        sum += sortedPnL[i];
        ++count;
    }
    metrics_.cvar95 = count > 0 ? -sum / count : 0;
}

void RiskManager::CalculateMonteCarloSimulation() {
    GeneratePriceScenarios();
    CalculateScenarioPnL();
    AggregateScenarioResults();
}

void RiskManager::GeneratePriceScenarios() {
    priceScenarios_.clear();
    priceScenarios_.resize(MONTE_CARLO_SCENARIOS);
    
    std::normal_distribution<double> normal(0, 1);
    
    for (auto& scenario : priceScenarios_) {
        scenario.resize(HISTORICAL_WINDOW);
        Price currentPrice = priceHistory_.back();
        
        for (size_t i = 0; i < HISTORICAL_WINDOW; ++i) {
            double random = normal(rng_);
            currentPrice *= std::exp(random * metrics_.currentVolatility / std::sqrt(252));
            scenario[i] = currentPrice;
        }
    }
}

void RiskManager::CalculateScenarioPnL() {
    scenarioPnL_.clear();
    scenarioPnL_.resize(MONTE_CARLO_SCENARIOS);
    
    for (size_t i = 0; i < MONTE_CARLO_SCENARIOS; ++i) {
        Price scenarioPnL = 0;
        for (size_t j = 0; j < HISTORICAL_WINDOW; ++j) {
            scenarioPnL += metrics_.currentPositionValue * 
                (priceScenarios_[i][j] - priceHistory_.back()) / priceHistory_.back();
        }
        scenarioPnL_[i] = scenarioPnL;
    }
}

void RiskManager::AggregateScenarioResults() {
    std::sort(scenarioPnL_.begin(), scenarioPnL_.end());
    
    size_t varIndex = static_cast<size_t>((1 - VAR_CONFIDENCE_LEVEL) * scenarioPnL_.size());
    metrics_.var95 = -scenarioPnL_[varIndex];
    
    double sum = 0;
    size_t count = 0;
    for (size_t i = 0; i < varIndex; ++i) {
        sum += scenarioPnL_[i];
        ++count;
    }
    metrics_.cvar95 = count > 0 ? -sum / count : 0;
}

void RiskManager::CalculateHistoricalSimulation() {
    UpdateHistoricalData();
    SortHistoricalScenarios();
    CalculateHistoricalVaR();
}

void RiskManager::UpdateHistoricalData() {
    // Implementation would update historical price and PnL data
    // This is a placeholder for actual implementation
}

void RiskManager::SortHistoricalScenarios() {
    std::sort(pnlHistory_.begin(), pnlHistory_.end());
}

void RiskManager::CalculateHistoricalVaR() {
    if (pnlHistory_.empty()) return;
    
    size_t varIndex = static_cast<size_t>((1 - VAR_CONFIDENCE_LEVEL) * pnlHistory_.size());
    metrics_.var95 = -pnlHistory_[varIndex];
    
    double sum = 0;
    size_t count = 0;
    for (size_t i = 0; i < varIndex; ++i) {
        sum += pnlHistory_[i];
        ++count;
    }
    metrics_.cvar95 = count > 0 ? -sum / count : 0;
}

void RiskManager::CalculateRiskDecomposition() {
    DecomposeMarketRisk();
    DecomposeCreditRisk();
    DecomposeLiquidityRisk();
    DecomposeOperationalRisk();
    DecomposeCounterpartyRisk();
}

double RiskManager::CalculateMarketRisk() {
    return metrics_.currentVolatility * metrics_.currentPositionValue;
}

double RiskManager::CalculateCreditRisk() {
    return metrics_.currentExposure * 0.02; // Assuming 2% default probability
}

double RiskManager::CalculateLiquidityRisk() {
    return metrics_.currentPositionValue * (1.0 - metrics_.currentConcentration);
}

double RiskManager::CalculateOperationalRisk() {
    return metrics_.currentPositionValue * 0.01; // Assuming 1% operational risk
}

double RiskManager::CalculateCounterpartyRisk() {
    return metrics_.currentExposure * 0.015; // Assuming 1.5% counterparty risk
}

double RiskManager::CalculateConcentrationRisk() {
    return metrics_.currentPositionValue * metrics_.currentConcentration;
}

double RiskManager::CalculateCorrelationRisk() {
    return metrics_.currentPositionValue * metrics_.currentCorrelation;
}

double RiskManager::CalculateTailRisk() {
    return metrics_.cvar95 - metrics_.var95;
}

double RiskManager::CalculateSystemicRisk() {
    return metrics_.currentPositionValue * 0.03; // Assuming 3% systemic risk
}

double RiskManager::CalculateRegulatoryRisk() {
    return metrics_.currentPositionValue * 0.01; // Assuming 1% regulatory risk
}

void RiskManager::DecomposeMarketRisk() {
    // Implementation would decompose market risk into its components
    // This is a placeholder for actual implementation
}

void RiskManager::DecomposeCreditRisk() {
    // Implementation would decompose credit risk into its components
    // This is a placeholder for actual implementation
}

void RiskManager::DecomposeLiquidityRisk() {
    // Implementation would decompose liquidity risk into its components
    // This is a placeholder for actual implementation
}

void RiskManager::DecomposeOperationalRisk() {
    // Implementation would decompose operational risk into its components
    // This is a placeholder for actual implementation
}

void RiskManager::DecomposeCounterpartyRisk() {
    // Implementation would decompose counterparty risk into its components
    // This is a placeholder for actual implementation
}

RiskMetrics RiskManager::GetCurrentRiskMetrics() const {
    return metrics_;
}

std::vector<Price> RiskManager::GetHistoricalPnL() const {
    return pnlHistory_;
}

std::map<std::string, double> RiskManager::GetRiskFactorContributions() const {
    std::map<std::string, double> contributions;
    contributions["Market Risk"] = CalculateMarketRisk();
    contributions["Credit Risk"] = CalculateCreditRisk();
    contributions["Liquidity Risk"] = CalculateLiquidityRisk();
    contributions["Operational Risk"] = CalculateOperationalRisk();
    contributions["Counterparty Risk"] = CalculateCounterpartyRisk();
    contributions["Concentration Risk"] = CalculateConcentrationRisk();
    contributions["Correlation Risk"] = CalculateCorrelationRisk();
    contributions["Tail Risk"] = CalculateTailRisk();
    contributions["Systemic Risk"] = CalculateSystemicRisk();
    contributions["Regulatory Risk"] = CalculateRegulatoryRisk();
    return contributions;
}

std::vector<std::pair<std::string, double>> RiskManager::GetStressTestResults() const {
    std::vector<std::pair<std::string, double>> results;
    results.emplace_back("VaR (95%)", metrics_.var95);
    results.emplace_back("CVaR (95%)", metrics_.cvar95);
    results.emplace_back("Current Drawdown", metrics_.currentDrawdown);
    results.emplace_back("Daily PnL", metrics_.dailyPnL);
    results.emplace_back("Position Value", metrics_.currentPositionValue);
    results.emplace_back("Exposure", metrics_.currentExposure);
    results.emplace_back("Leverage", metrics_.currentLeverage);
    results.emplace_back("Concentration", metrics_.currentConcentration);
    results.emplace_back("Volatility", metrics_.currentVolatility);
    return results;
}

void RiskManager::AdjustPositionLimits() {
    // Implementation would adjust position limits based on risk metrics
    // This is a placeholder for actual implementation
}

void RiskManager::AdjustRiskParameters() {
    // Implementation would adjust risk parameters based on market conditions
    // This is a placeholder for actual implementation
}

void RiskManager::AdjustHedgingStrategy() {
    // Implementation would adjust hedging strategy based on risk metrics
    // This is a placeholder for actual implementation
} 
#pragma once

#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <cmath>
#include <random>
#include "Usings.h"

class RiskManager {
public:
    // Risk limits and thresholds
    struct RiskLimits {
        Price maxPositionValue;
        Price maxDrawdown;
        Price maxDailyLoss;
        Price maxOrderSize;
        Price maxExposure;
        double maxLeverage;
        double maxConcentration;
        double maxCorrelation;
        double maxVolatility;
    };

    // Risk metrics
    struct RiskMetrics {
        Price currentPositionValue;
        Price currentDrawdown;
        Price dailyPnL;
        Price currentExposure;
        double currentLeverage;
        double currentConcentration;
        double currentCorrelation;
        double currentVolatility;
        double var95;
        double cvar95;
        std::vector<Price> historicalPnL;
    };

    // Risk factor weights for VaR calculation
    struct RiskFactorWeights {
        double marketRisk;
        double creditRisk;
        double liquidityRisk;
        double operationalRisk;
        double counterpartyRisk;
        double concentrationRisk;
        double correlationRisk;
        double tailRisk;
        double systemicRisk;
        double regulatoryRisk;
    };

    RiskManager(const RiskLimits& limits);
    
    // Core risk management functions
    bool CheckOrderRisk(const OrderPointer& order);
    void UpdatePosition(const Trade& trade);
    void UpdateMarketData(const Price& currentPrice);
    bool IsRiskLimitBreached() const;
    
    // Risk calculation functions
    void CalculateVaR();
    void CalculateCVaR();
    void CalculateStressTest();
    void CalculateMonteCarloSimulation();
    void CalculateHistoricalSimulation();
    void CalculateRiskDecomposition();
    
    // Risk monitoring functions
    void MonitorMarketRisk();
    void MonitorCreditRisk();
    void MonitorLiquidityRisk();
    void MonitorOperationalRisk();
    void MonitorCounterpartyRisk();
    void MonitorConcentrationRisk();
    void MonitorCorrelationRisk();
    void MonitorTailRisk();
    void MonitorSystemicRisk();
    void MonitorRegulatoryRisk();
    
    // Risk reporting functions
    RiskMetrics GetCurrentRiskMetrics() const;
    std::vector<Price> GetHistoricalPnL() const;
    std::map<std::string, double> GetRiskFactorContributions() const;
    std::vector<std::pair<std::string, double>> GetStressTestResults() const;
    
    // Risk adjustment functions
    void AdjustPositionLimits();
    void AdjustRiskParameters();
    void AdjustHedgingStrategy();
    
private:
    // Internal risk calculation helpers
    double CalculateMarketRisk();
    double CalculateCreditRisk();
    double CalculateLiquidityRisk();
    double CalculateOperationalRisk();
    double CalculateCounterpartyRisk();
    double CalculateConcentrationRisk();
    double CalculateCorrelationRisk();
    double CalculateTailRisk();
    double CalculateSystemicRisk();
    double CalculateRegulatoryRisk();
    
    // Monte Carlo simulation helpers
    void GeneratePriceScenarios();
    void CalculateScenarioPnL();
    void AggregateScenarioResults();
    
    // Historical simulation helpers
    void UpdateHistoricalData();
    void SortHistoricalScenarios();
    void CalculateHistoricalVaR();
    
    // Risk decomposition helpers
    void DecomposeMarketRisk();
    void DecomposeCreditRisk();
    void DecomposeLiquidityRisk();
    void DecomposeOperationalRisk();
    void DecomposeCounterpartyRisk();
    
    // Internal state
    RiskLimits limits_;
    RiskMetrics metrics_;
    RiskFactorWeights weights_;
    
    std::vector<Price> priceHistory_;
    std::vector<Price> pnlHistory_;
    std::vector<std::vector<Price>> priceScenarios_;
    std::vector<Price> scenarioPnL_;
    
    std::chrono::system_clock::time_point lastUpdate_;
    std::mt19937 rng_;
    
    // Risk thresholds
    const double VAR_CONFIDENCE_LEVEL = 0.95;
    const size_t HISTORICAL_WINDOW = 252;  // 1 year of trading days
    const size_t MONTE_CARLO_SCENARIOS = 10000;
    
    // Risk decomposition thresholds
    const double CONCENTRATION_THRESHOLD = 0.2;  // 20% concentration limit
    const double CORRELATION_THRESHOLD = 0.7;    // 70% correlation limit
    const double LEVERAGE_THRESHOLD = 2.0;       // 2x leverage limit
}; 
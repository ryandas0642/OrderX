#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <chrono>
#include "Usings.h"
#include "MLModel.h"

class SmartOrderRouter {
public:
    // Venue information
    struct VenueInfo {
        std::string name;
        double latency;
        double fees;
        double liquidity;
        double spread;
        double volume;
        double reliability;
        std::vector<double> historicalPerformance;
        MLModel::TrainingMetrics mlMetrics;
    };

    // Routing decision
    struct RoutingDecision {
        std::vector<std::string> targetVenues;
        std::vector<double> venueWeights;
        std::vector<double> orderSizes;
        double expectedCost;
        double expectedSlippage;
        double confidence;
        std::vector<double> venueRisks;
    };

    // Performance metrics
    struct PerformanceMetrics {
        double totalVolume;
        double totalCost;
        double averageSlippage;
        double fillRate;
        double averageLatency;
        std::vector<double> venuePerformance;
        std::vector<double> strategyPerformance;
    };

    SmartOrderRouter();
    
    // Core routing functions
    RoutingDecision RouteOrder(const OrderPointer& order);
    void UpdateVenueInfo(const std::string& venue, const VenueInfo& info);
    void UpdateExecutionData(const std::string& venue, const OrderPointer& order, 
                           double actualCost, double actualSlippage);
    
    // Venue management
    void AddVenue(const std::string& name, const VenueInfo& info);
    void RemoveVenue(const std::string& name);
    void UpdateVenueParameters(const std::string& name);
    
    // Strategy management
    void SetRoutingStrategy(const std::string& strategy);
    void UpdateStrategyParameters();
    void OptimizeStrategy();
    
    // Risk management
    void SetRiskLimits(const std::map<std::string, double>& limits);
    void UpdateRiskMetrics();
    bool CheckRiskLimits(const RoutingDecision& decision) const;
    
    // Performance monitoring
    PerformanceMetrics GetPerformanceMetrics() const;
    std::vector<double> GetVenuePerformance(const std::string& venue) const;
    std::map<std::string, double> GetStrategyPerformance() const;
    
private:
    // Internal routing helpers
    RoutingDecision CalculateOptimalRouting(const OrderPointer& order);
    std::vector<double> CalculateVenueWeights(const OrderPointer& order);
    std::vector<double> CalculateOrderSizes(const OrderPointer& order, 
                                          const std::vector<double>& weights);
    double CalculateExpectedCost(const RoutingDecision& decision) const;
    double CalculateExpectedSlippage(const RoutingDecision& decision) const;
    
    // Venue analysis helpers
    void AnalyzeVenueLiquidity(const std::string& venue);
    void AnalyzeVenueLatency(const std::string& venue);
    void AnalyzeVenueCosts(const std::string& venue);
    void AnalyzeVenueReliability(const std::string& venue);
    
    // Strategy helpers
    void InitializeStrategy();
    void UpdateStrategyState();
    void OptimizeStrategyParameters();
    
    // Risk analysis helpers
    void CalculateVenueRisks(RoutingDecision& decision);
    void UpdateRiskMetrics(const RoutingDecision& decision);
    bool CheckVenueRiskLimits(const std::string& venue, double exposure) const;
    
    // Performance analysis helpers
    void UpdatePerformanceMetrics(const RoutingDecision& decision, 
                                const std::map<std::string, double>& results);
    void CalculateStrategyPerformance();
    void UpdateVenuePerformance(const std::string& venue, double performance);
    
    // ML model helpers
    void InitializeMLModels();
    void UpdateMLModels();
    double PredictVenuePerformance(const std::string& venue, 
                                 const OrderPointer& order) const;
    
    // Internal state
    std::map<std::string, VenueInfo> venues_;
    std::map<std::string, double> riskLimits_;
    std::map<std::string, std::vector<double>> strategyPerformance_;
    PerformanceMetrics metrics_;
    
    std::string currentStrategy_;
    std::chrono::system_clock::time_point lastUpdate_;
    
    // ML models
    std::map<std::string, std::unique_ptr<MLModel>> venueModels_;
    std::unique_ptr<MLModel> strategyModel_;
    
    // Configuration parameters
    const size_t MIN_TRAINING_SAMPLES = 1000;
    const double MAX_VENUE_EXPOSURE = 0.3;  // Maximum exposure per venue
    const double MIN_VENUE_WEIGHT = 0.1;    // Minimum weight per venue
    const double MAX_SLIPPAGE = 0.02;       // Maximum allowed slippage
    const double MIN_FILL_RATE = 0.95;      // Minimum required fill rate
    
    // Strategy parameters
    const double LIQUIDITY_WEIGHT = 0.4;
    const double COST_WEIGHT = 0.3;
    const double LATENCY_WEIGHT = 0.2;
    const double RELIABILITY_WEIGHT = 0.1;
}; 
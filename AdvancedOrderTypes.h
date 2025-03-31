#pragma once

#include <memory>
#include <variant>
#include <optional>
#include <chrono>
#include "Order.h"
#include "Usings.h"
#include <vector>
#include <string>
#include "MLModel.h"

// Forward declarations
class Orderbook;

// Base class for all advanced order types
class AdvancedOrder : public Order {
public:
    AdvancedOrder(OrderType orderType, OrderId orderId, Side side, Price price, Quantity quantity)
        : Order(orderType, orderId, side, price, quantity)
        , status_(OrderStatus::Active)
        , creationTime_(std::chrono::system_clock::now())
    {}

    virtual ~AdvancedOrder() = default;

    // Core virtual methods that all advanced orders must implement
    virtual bool ShouldExecute(const Orderbook& orderbook) const = 0;
    virtual void OnTriggered(Orderbook& orderbook) = 0;
    virtual void OnCancelled() = 0;
    
    // Common functionality
    OrderStatus GetStatus() const { return status_; }
    std::chrono::system_clock::time_point GetCreationTime() const { return creationTime_; }
    
protected:
    OrderStatus status_;
    std::chrono::system_clock::time_point creationTime_;
};

// Stop Loss Order
class StopLossOrder : public AdvancedOrder {
public:
    StopLossOrder(OrderId orderId, Side side, Price triggerPrice, Quantity quantity)
        : AdvancedOrder(OrderType::StopLoss, orderId, side, triggerPrice, quantity)
        , triggerPrice_(triggerPrice)
    {}

    bool ShouldExecute(const Orderbook& orderbook) const override;
    void OnTriggered(Orderbook& orderbook) override;
    void OnCancelled() override { status_ = OrderStatus::Cancelled; }

private:
    Price triggerPrice_;
};

// Take Profit Order
class TakeProfitOrder : public AdvancedOrder {
public:
    TakeProfitOrder(OrderId orderId, Side side, Price triggerPrice, Quantity quantity)
        : AdvancedOrder(OrderType::TakeProfit, orderId, side, triggerPrice, quantity)
        , triggerPrice_(triggerPrice)
    {}

    bool ShouldExecute(const Orderbook& orderbook) const override;
    void OnTriggered(Orderbook& orderbook) override;
    void OnCancelled() override { status_ = OrderStatus::Cancelled; }

private:
    Price triggerPrice_;
};

// Trailing Stop Order
class TrailingStopOrder : public AdvancedOrder {
public:
    TrailingStopOrder(OrderId orderId, Side side, Price initialTriggerPrice, 
                     Price trailingDistance, Quantity quantity)
        : AdvancedOrder(OrderType::TrailingStop, orderId, side, initialTriggerPrice, quantity)
        , triggerPrice_(initialTriggerPrice)
        , trailingDistance_(trailingDistance)
        , highestPrice_(side == Side::Buy ? Price(0) : std::numeric_limits<Price>::max())
    {}

    bool ShouldExecute(const Orderbook& orderbook) const override;
    void OnTriggered(Orderbook& orderbook) override;
    void OnCancelled() override { status_ = OrderStatus::Cancelled; }
    void UpdateTriggerPrice(Price currentPrice);

private:
    Price triggerPrice_;
    Price trailingDistance_;
    Price highestPrice_;
};

// OCO (One-Cancels-Other) Order
class OCOOrder : public AdvancedOrder {
public:
    OCOOrder(OrderId orderId, Side side, Price stopPrice, Price limitPrice, 
             Quantity quantity)
        : AdvancedOrder(OrderType::OCO, orderId, side, stopPrice, quantity)
        , stopPrice_(stopPrice)
        , limitPrice_(limitPrice)
        , stopOrder_(std::make_shared<StopLossOrder>(orderId, side, stopPrice, quantity))
        , limitOrder_(std::make_shared<Order>(OrderType::Limit, orderId, side, limitPrice, quantity))
    {}

    bool ShouldExecute(const Orderbook& orderbook) const override;
    void OnTriggered(Orderbook& orderbook) override;
    void OnCancelled() override {
        status_ = OrderStatus::Cancelled;
        stopOrder_->OnCancelled();
        limitOrder_->OnCancelled();
    }

private:
    Price stopPrice_;
    Price limitPrice_;
    std::shared_ptr<StopLossOrder> stopOrder_;
    std::shared_ptr<Order> limitOrder_;
};

// Pegged Order
class PeggedOrder : public AdvancedOrder {
public:
    enum class PegType {
        Primary,    // Peg to primary market
        MidPrice,   // Peg to mid price
        BestBid,    // Peg to best bid
        BestAsk,    // Peg to best ask
        Custom      // Peg with custom offset
    };

    PeggedOrder(OrderId orderId, Side side, PegType pegType, 
                Price offset, Quantity quantity)
        : AdvancedOrder(OrderType::Pegged, orderId, side, Price(0), quantity)
        , pegType_(pegType)
        , offset_(offset)
    {}

    bool ShouldExecute(const Orderbook& orderbook) const override;
    void OnTriggered(Orderbook& orderbook) override;
    void OnCancelled() override { status_ = OrderStatus::Cancelled; }
    Price CalculatePegPrice(const Orderbook& orderbook) const;

private:
    PegType pegType_;
    Price offset_;
};

// Market-on-Close Order
class MarketOnCloseOrder : public AdvancedOrder {
public:
    MarketOnCloseOrder(OrderId orderId, Side side, Quantity quantity)
        : AdvancedOrder(OrderType::MarketOnClose, orderId, side, Price(0), quantity)
    {}

    bool ShouldExecute(const Orderbook& orderbook) const override;
    void OnTriggered(Orderbook& orderbook) override;
    void OnCancelled() override { status_ = OrderStatus::Cancelled; }
};

// Spread Order
class SpreadOrder : public AdvancedOrder {
public:
    SpreadOrder(OrderId orderId, Side side, Price spread, Quantity quantity)
        : AdvancedOrder(OrderType::Spread, orderId, side, Price(0), quantity)
        , spread_(spread)
    {}

    bool ShouldExecute(const Orderbook& orderbook) const override;
    void OnTriggered(Orderbook& orderbook) override;
    void OnCancelled() override { status_ = OrderStatus::Cancelled; }

private:
    Price spread_;
};

// Straddle Order
class StraddleOrder : public AdvancedOrder {
public:
    StraddleOrder(OrderId orderId, Side side, Price strikePrice, 
                  Price callStrike, Price putStrike, Quantity quantity)
        : AdvancedOrder(OrderType::Straddle, orderId, side, strikePrice, quantity)
        , strikePrice_(strikePrice)
        , callStrike_(callStrike)
        , putStrike_(putStrike)
    {}

    bool ShouldExecute(const Orderbook& orderbook) const override;
    void OnTriggered(Orderbook& orderbook) override;
    void OnCancelled() override { status_ = OrderStatus::Cancelled; }

private:
    Price strikePrice_;
    Price callStrike_;
    Price putStrike_;
};

// Type alias for advanced order pointer
using AdvancedOrderPointer = std::shared_ptr<AdvancedOrder>;

class AdvancedOrderTypes {
public:
    // Order type enumeration
    enum class OrderType {
        Market,
        Limit,
        Iceberg,
        TWAP,
        VWAP,
        SmartSplit,
        Adaptive
    };

    // Order parameters
    struct OrderParameters {
        double totalQuantity;
        double price;
        double minQuantity;
        double maxQuantity;
        double timeInterval;
        double volumeTarget;
        double priceDeviation;
        double urgency;
        std::vector<double> historicalVolumes;
        std::vector<double> historicalPrices;
    };

    // Order state
    struct OrderState {
        double remainingQuantity;
        double averagePrice;
        double executionTime;
        double marketImpact;
        double slippage;
        std::vector<double> executionPrices;
        std::vector<double> executionQuantities;
        std::vector<std::chrono::system_clock::time_point> executionTimes;
        MLModel::TrainingMetrics mlMetrics;
    };

    // Order configuration
    struct OrderConfig {
        OrderType type;
        OrderParameters params;
        OrderState state;
        std::string strategy;
        std::vector<std::string> targetVenues;
        std::vector<double> venueWeights;
        bool isActive;
        std::chrono::system_clock::time_point startTime;
        std::chrono::system_clock::time_point endTime;
    };

    AdvancedOrderTypes();
    
    // Order creation and management
    OrderPointer CreateOrder(const OrderConfig& config);
    void UpdateOrder(const OrderPointer& order, const OrderState& state);
    void CancelOrder(const OrderPointer& order);
    void ModifyOrder(const OrderPointer& order, const OrderParameters& params);
    
    // Order execution
    void ExecuteOrder(const OrderPointer& order);
    void SplitOrder(const OrderPointer& order);
    void AdaptOrder(const OrderPointer& order);
    
    // Order monitoring
    OrderState GetOrderState(const OrderPointer& order) const;
    std::vector<OrderState> GetOrderHistory(const OrderPointer& order) const;
    double GetOrderProgress(const OrderPointer& order) const;
    
    // Strategy management
    void SetExecutionStrategy(const OrderPointer& order, const std::string& strategy);
    void UpdateStrategyParameters(const OrderPointer& order);
    void OptimizeStrategy(const OrderPointer& order);
    
    // Performance analysis
    double CalculateExecutionQuality(const OrderPointer& order) const;
    double CalculateMarketImpact(const OrderPointer& order) const;
    double CalculateSlippage(const OrderPointer& order) const;
    double CalculateCost(const OrderPointer& order) const;
    
private:
    // Order execution helpers
    void ExecuteIcebergOrder(const OrderPointer& order);
    void ExecuteTWAPOrder(const OrderPointer& order);
    void ExecuteVWAPOrder(const OrderPointer& order);
    void ExecuteSmartSplitOrder(const OrderPointer& order);
    void ExecuteAdaptiveOrder(const OrderPointer& order);
    
    // Order splitting helpers
    std::vector<double> CalculateSplitSizes(const OrderPointer& order);
    std::vector<double> CalculateSplitPrices(const OrderPointer& order);
    std::vector<std::chrono::system_clock::time_point> CalculateSplitTimes(
        const OrderPointer& order);
    
    // Order adaptation helpers
    void AdaptOrderSize(const OrderPointer& order);
    void AdaptOrderPrice(const OrderPointer& order);
    void AdaptOrderTiming(const OrderPointer& order);
    void AdaptOrderVenues(const OrderPointer& order);
    
    // Strategy helpers
    void InitializeStrategy(const OrderPointer& order);
    void UpdateStrategyState(const OrderPointer& order);
    void OptimizeStrategyParameters(const OrderPointer& order);
    
    // Performance analysis helpers
    void CalculateExecutionMetrics(const OrderPointer& order);
    void UpdateHistoricalData(const OrderPointer& order);
    void AnalyzeMarketImpact(const OrderPointer& order);
    void AnalyzeSlippage(const OrderPointer& order);
    
    // ML model helpers
    void InitializeMLModels();
    void UpdateMLModels(const OrderPointer& order);
    double PredictExecutionQuality(const OrderPointer& order) const;
    double PredictMarketImpact(const OrderPointer& order) const;
    
    // Internal state
    std::map<OrderPointer, OrderConfig> orders_;
    std::map<OrderPointer, std::vector<OrderState>> orderHistory_;
    std::map<std::string, std::unique_ptr<MLModel>> mlModels_;
    
    // Configuration parameters
    const size_t MIN_HISTORY_SAMPLES = 1000;
    const double MAX_PRICE_DEVIATION = 0.02;
    const double MIN_EXECUTION_QUALITY = 0.8;
    const double MAX_MARKET_IMPACT = 0.01;
    const double MAX_SLIPPAGE = 0.005;
    
    // Strategy parameters
    const double SIZE_WEIGHT = 0.3;
    const double PRICE_WEIGHT = 0.3;
    const double TIME_WEIGHT = 0.2;
    const double IMPACT_WEIGHT = 0.2;
    
    // ML model parameters
    const size_t MIN_TRAINING_SAMPLES = 1000;
    const double LEARNING_RATE = 0.01;
    const size_t MAX_ITERATIONS = 1000;
    const double REGULARIZATION = 0.01;
}; 
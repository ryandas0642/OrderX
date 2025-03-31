#include "AdvancedOrderTypes.h"
#include "Orderbook.h"
#include <algorithm>
#include <cmath>

// Stop Loss Order Implementation
bool StopLossOrder::ShouldExecute(const Orderbook& orderbook) const {
    if (GetSide() == Side::Buy) {
        return orderbook.GetBestAsk() <= triggerPrice_;
    } else {
        return orderbook.GetBestBid() >= triggerPrice_;
    }
}

void StopLossOrder::OnTriggered(Orderbook& orderbook) {
    // Convert to market order when triggered
    auto marketOrder = std::make_shared<Order>(
        OrderType::Market,
        GetOrderId(),
        GetSide(),
        GetRemainingQuantity()
    );
    orderbook.AddOrder(marketOrder);
    status_ = OrderStatus::Triggered;
}

// Take Profit Order Implementation
bool TakeProfitOrder::ShouldExecute(const Orderbook& orderbook) const {
    if (GetSide() == Side::Buy) {
        return orderbook.GetBestBid() >= triggerPrice_;
    } else {
        return orderbook.GetBestAsk() <= triggerPrice_;
    }
}

void TakeProfitOrder::OnTriggered(Orderbook& orderbook) {
    // Convert to limit order at trigger price
    auto limitOrder = std::make_shared<Order>(
        OrderType::Limit,
        GetOrderId(),
        GetSide(),
        triggerPrice_,
        GetRemainingQuantity()
    );
    orderbook.AddOrder(limitOrder);
    status_ = OrderStatus::Triggered;
}

// Trailing Stop Order Implementation
bool TrailingStopOrder::ShouldExecute(const Orderbook& orderbook) const {
    Price currentPrice = GetSide() == Side::Buy ? 
        orderbook.GetBestBid() : orderbook.GetBestAsk();
    
    UpdateTriggerPrice(currentPrice);
    
    if (GetSide() == Side::Buy) {
        return currentPrice <= triggerPrice_;
    } else {
        return currentPrice >= triggerPrice_;
    }
}

void TrailingStopOrder::UpdateTriggerPrice(Price currentPrice) {
    if (GetSide() == Side::Buy) {
        highestPrice_ = std::max(highestPrice_, currentPrice);
        triggerPrice_ = highestPrice_ - trailingDistance_;
    } else {
        highestPrice_ = std::min(highestPrice_, currentPrice);
        triggerPrice_ = highestPrice_ + trailingDistance_;
    }
}

void TrailingStopOrder::OnTriggered(Orderbook& orderbook) {
    auto marketOrder = std::make_shared<Order>(
        OrderType::Market,
        GetOrderId(),
        GetSide(),
        GetRemainingQuantity()
    );
    orderbook.AddOrder(marketOrder);
    status_ = OrderStatus::Triggered;
}

// OCO Order Implementation
bool OCOOrder::ShouldExecute(const Orderbook& orderbook) const {
    return stopOrder_->ShouldExecute(orderbook) || 
           (GetSide() == Side::Buy ? 
            orderbook.GetBestAsk() <= limitPrice_ : 
            orderbook.GetBestBid() >= limitPrice_);
}

void OCOOrder::OnTriggered(Orderbook& orderbook) {
    if (stopOrder_->ShouldExecute(orderbook)) {
        stopOrder_->OnTriggered(orderbook);
        limitOrder_->OnCancelled();
    } else {
        limitOrder_->Fill(GetRemainingQuantity());
        stopOrder_->OnCancelled();
    }
    status_ = OrderStatus::Triggered;
}

// Pegged Order Implementation
bool PeggedOrder::ShouldExecute(const Orderbook& orderbook) const {
    Price pegPrice = CalculatePegPrice(orderbook);
    if (pegPrice == Price(0)) return false;
    
    if (GetSide() == Side::Buy) {
        return orderbook.GetBestAsk() <= pegPrice;
    } else {
        return orderbook.GetBestBid() >= pegPrice;
    }
}

Price PeggedOrder::CalculatePegPrice(const Orderbook& orderbook) const {
    switch (pegType_) {
        case PegType::Primary:
            return orderbook.GetPrimaryMarketPrice();
        case PegType::MidPrice:
            return (orderbook.GetBestBid() + orderbook.GetBestAsk()) / 2;
        case PegType::BestBid:
            return orderbook.GetBestBid();
        case PegType::BestAsk:
            return orderbook.GetBestAsk();
        case PegType::Custom:
            return GetSide() == Side::Buy ? 
                   orderbook.GetBestBid() + offset_ : 
                   orderbook.GetBestAsk() - offset_;
        default:
            return Price(0);
    }
}

void PeggedOrder::OnTriggered(Orderbook& orderbook) {
    Price pegPrice = CalculatePegPrice(orderbook);
    if (pegPrice == Price(0)) return;

    auto limitOrder = std::make_shared<Order>(
        OrderType::Limit,
        GetOrderId(),
        GetSide(),
        pegPrice,
        GetRemainingQuantity()
    );
    orderbook.AddOrder(limitOrder);
    status_ = OrderStatus::Triggered;
}

// Market-on-Close Order Implementation
bool MarketOnCloseOrder::ShouldExecute(const Orderbook& orderbook) const {
    return orderbook.IsMarketClosing();
}

void MarketOnCloseOrder::OnTriggered(Orderbook& orderbook) {
    auto marketOrder = std::make_shared<Order>(
        OrderType::Market,
        GetOrderId(),
        GetSide(),
        GetRemainingQuantity()
    );
    orderbook.AddOrder(marketOrder);
    status_ = OrderStatus::Triggered;
}

// Spread Order Implementation
bool SpreadOrder::ShouldExecute(const Orderbook& orderbook) const {
    Price currentSpread = orderbook.GetBestAsk() - orderbook.GetBestBid();
    return GetSide() == Side::Buy ? 
           currentSpread <= spread_ : 
           currentSpread >= spread_;
}

void SpreadOrder::OnTriggered(Orderbook& orderbook) {
    auto marketOrder = std::make_shared<Order>(
        OrderType::Market,
        GetOrderId(),
        GetSide(),
        GetRemainingQuantity()
    );
    orderbook.AddOrder(marketOrder);
    status_ = OrderStatus::Triggered;
}

// Straddle Order Implementation
bool StraddleOrder::ShouldExecute(const Orderbook& orderbook) const {
    Price currentPrice = orderbook.GetMidPrice();
    Price callSpread = std::abs(currentPrice - callStrike_);
    Price putSpread = std::abs(currentPrice - putStrike_);
    
    return callSpread <= strikePrice_ || putSpread <= strikePrice_;
}

void StraddleOrder::OnTriggered(Orderbook& orderbook) {
    Price currentPrice = orderbook.GetMidPrice();
    Price callSpread = std::abs(currentPrice - callStrike_);
    Price putSpread = std::abs(currentPrice - putStrike_);
    
    if (callSpread <= strikePrice_) {
        auto callOrder = std::make_shared<Order>(
            OrderType::Limit,
            GetOrderId(),
            GetSide(),
            callStrike_,
            GetRemainingQuantity()
        );
        orderbook.AddOrder(callOrder);
    }
    
    if (putSpread <= strikePrice_) {
        auto putOrder = std::make_shared<Order>(
            OrderType::Limit,
            GetOrderId(),
            GetSide(),
            putStrike_,
            GetRemainingQuantity()
        );
        orderbook.AddOrder(putOrder);
    }
    
    status_ = OrderStatus::Triggered;
} 
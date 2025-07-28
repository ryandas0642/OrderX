# Advanced Orderbook System

A high-performance electronic trading system featuring sophisticated order management, advanced order types, and data collection for predictive modeling.

## Overview

This system implements a state-of-the-art orderbook with advanced features designed for high-frequency trading and market making. It provides robust order management, sophisticated order types, and collects data to be used for predictive modeling.

## Key Features

### 1. Advanced Order Types
- **Smart Order Types**:
  - Iceberg Orders: Large orders split into smaller visible chunks
  - TWAP Orders: Time-weighted average price execution
  - VWAP Orders: Volume-weighted average price execution
  - Smart Split Orders: Intelligent order splitting based on market conditions
  - Adaptive Orders: Self-adjusting orders that adapt to market conditions

- **Conditional Orders**:
  - Stop Loss Orders: Automatically execute at specified price levels
  - Take Profit Orders: Secure profits at target price levels
  - Trailing Stop Orders: Dynamic stop loss that follows price movement
  - OCO Orders: One-Cancels-Other orders for complex trading strategies
  - Pegged Orders: Orders that track market prices with custom offsets

### 2. Smart Order Router
- **Multi-Venue Execution**:
  - Intelligent order splitting across multiple venues
  - Dynamic venue selection based on multiple factors
  - Risk-aware venue allocation
  - Real-time venue performance tracking

### 3. Market Impact Analysis
- **Impact Modeling**:
  - Temporary and permanent impact analysis
  - Market regime classification
  - Impact prediction using ML models
  - Cost and risk analysis

- **Execution Optimization**:
  - Optimal execution time calculation
  - Smart order splitting
  - Impact-aware execution strategies
  - Real-time adaptation to market conditions

### 4. Risk Management
- **Position Management**:
  - Position limits and exposure tracking
  - Real-time P&L monitoring
  - Value-at-Risk (VaR) calculations
  - Stress testing capabilities

- **Risk Controls**:
  - Price deviation limits
  - Size constraints
  - Time-based controls
  - Impact monitoring
  - Slippage tracking

### 5. Data Collection for Predictive Modeling
- **Comprehensive Data Logging**:
  - Detailed records of all order actions (add, cancel, modify, fill)
  - High-resolution market data snapshots (order book depth, trades)
  - Execution data, including venue, price, and latency
- **Feature Engineering**:
  - Extraction of meaningful features from raw data
  - Calculation of various market indicators (e.g., volatility, liquidity)
- **Model Training and Evaluation**:
  - Framework for training and testing predictive models
  - Backtesting capabilities to evaluate model performance

## Technical Details

### Architecture
- C++17 implementation for maximum performance
- Thread-safe design for concurrent operations
- Lock-free data structures where applicable
- Efficient memory management
- Real-time processing capabilities

### Performance Optimization
- SIMD instructions for calculations
- GPU acceleration for ML models
- Optimized data structures
- Efficient memory allocation
- Low-latency execution

### Machine Learning Integration
- Data collection for training predictive models (e.g., market impact, price prediction)
- Framework for feature engineering and model evaluation

## Building and Running

### Prerequisites
- C++17 compatible compiler
- CMake 3.15 or higher
- Eigen library for linear algebra
- CUDA toolkit (optional, for GPU acceleration)

### Build Instructions
```bash
mkdir build
cd build
cmake ..
make
```

### Running Tests
```bash
./orderbook_test
```

## Usage Examples

### Creating an Advanced Order
```cpp
AdvancedOrderTypes::OrderConfig config;
config.type = AdvancedOrderTypes::OrderType::Iceberg;
config.params.totalQuantity = 1000;
config.params.price = 100.0;
config.params.minQuantity = 100;
config.params.maxQuantity = 200;
config.params.timeInterval = 3600; // 1 hour
config.params.volumeTarget = 10000;
config.params.priceDeviation = 0.01;
config.params.urgency = 0.5;

auto order = advancedOrderTypes.CreateOrder(config);
```

### Smart Order Routing
```cpp
SmartOrderRouter::RoutingDecision decision = router.RouteOrder(order);
for (size_t i = 0; i < decision.targetVenues.size(); ++i) {
    std::cout << "Venue: " << decision.targetVenues[i] 
              << ", Size: " << decision.orderSizes[i] 
              << ", Weight: " << decision.venueWeights[i] << std::endl;
}
```

## Performance Metrics

The system is designed to achieve:
- Sub-millisecond order processing
- Microsecond-level market data handling
- Efficient memory usage
- Scalable architecture

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Eigen library for linear algebra operations
- Inspired by modern electronic trading systems
- Built with performance and reliability in mind
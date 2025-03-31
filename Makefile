CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I.
LDFLAGS = 

# Main program sources
MAIN_SOURCES = main.cpp Orderbook.cpp
MAIN_HEADERS = Orderbook.h Order.h OrderModify.h OrderType.h Side.h Trade.h TradeInfo.h LevelInfo.h OrderbookLevelInfos.h Constants.h Usings.h
MAIN_TARGET = orderbook

# Test program sources
TEST_SOURCES = OrderbookTest/test.cpp OrderbookTest/pch.cpp Orderbook.cpp
TEST_HEADERS = $(MAIN_HEADERS) OrderbookTest/pch.h
TEST_TARGET = orderbook_test

# Object files
MAIN_OBJS = $(MAIN_SOURCES:.cpp=.o)
TEST_OBJS = $(TEST_SOURCES:.cpp=.o)

all: $(MAIN_TARGET) $(TEST_TARGET)

$(MAIN_TARGET): $(MAIN_OBJS)
	$(CXX) $(MAIN_OBJS) -o $(MAIN_TARGET) $(LDFLAGS)

$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $(TEST_OBJS) -o $(TEST_TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(MAIN_OBJS) $(TEST_OBJS) $(MAIN_TARGET) $(TEST_TARGET)

.PHONY: all clean 
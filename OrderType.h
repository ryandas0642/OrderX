#pragma once

enum class OrderType
{
	Market,
	Limit,
	GoodTillCancel,
	StopLoss,
	TakeProfit,
	TrailingStop,
	OCO,
	Pegged,
	MarketOnClose,
	Spread,
	Straddle
};

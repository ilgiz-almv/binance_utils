from datetime import datetime

# Trading constants and risk management parameters
TAKER_RATE = 0.0005
ATR_MULTIPLIER = {
    'BTCUSDT': {
        'long': 4.5,
        'short': 5.5
    }
}
RISK_PROFIT_COEF = {
    'BTCUSDT': {
        'long': 2.0,
        'short': 2.0
    }
}
HOURS_PER_TRADE = {
    'BTCUSDT': {
        'long': 25.0,
        'short': 30.0
    }
}
PERP_PRICE_PRECISION = {
    'BTCUSDT': 1
}
PERP_AMOUNT_PRECISION = {
    'BTCUSDT': 3
}

MIN_STOP_LOSS_PCT = 0.004
MAX_STOP_LOSS_PCT = 0.018

# Data structure for storing trade details
trade_date_columns = ['time', 't_exit']
trade_dtype = {
    'symbol': 'object',
    'trade_dir': 'object',
    't_lim': 'float64',
    'coins_amount': 'float64',
    'limit_price': 'float64',
    'limit_avgPrice': 'float64',
    'limit_executedQty': 'float64',
    'stop_price': 'float64',
    'stop_avgPrice': 'float64',
    'stop_executedQty': 'float64',
    'tp_price': 'float64',
    'tp_avgPrice': 'float64',
    'tp_executedQty': 'float64',
    'limit_id': 'object',
    'stop_id': 'object',
    'tp_id': 'object',
    'limit_status': 'object',
    'stop_status': 'object',
    'tp_status': 'object',
    'atr': 'float64',
    'atr_mult': 'float64',
    'risk_profit_coef': 'float64',
    'min_stop_loss': 'float64',
    'max_stop_loss': 'float64',
    'sl': 'float64',
    'reason_exit': 'object',
    'trade_res': 'float64',
    'duration_h': 'float64',
    'exit_avgPrice': 'float64',
    'exit_executedQty': 'float64'
}


def num_to_str(amount, precision):
    """Convert a number to a string with a given precision."""
    return "{:0.0{}f}".format(amount, precision)


def calc_trade_res(entry_price, exit_price, amount, direction):
    """Calculate trade result based on entry and exit prices."""
    if direction.lower() == 'long':
        return (exit_price - entry_price) * amount - calc_commission(entry_price, exit_price, amount)
    elif direction.lower() == 'short':
        return (entry_price - exit_price) * amount - calc_commission(entry_price, exit_price, amount)


def stop_loss(atr, atr_multiplier, min_stop_loss, max_stop_loss=None):
    """Calculate the stop loss level based on ATR and risk management parameters."""
    if max_stop_loss:
        return min(max(atr * atr_multiplier, min_stop_loss), max_stop_loss)
    else:
        return max(atr * atr_multiplier, min_stop_loss)


def stop_loss_price(symbol, entry_price, direction, stop_loss):
    """Calculate the stop loss price for a given trade direction."""
    price_precision = PERP_PRICE_PRECISION[symbol]
    if direction == 'long':
        return round(entry_price - stop_loss, price_precision)
    elif direction == 'short':
        return round(entry_price + stop_loss, price_precision)


def take_profit_price(symbol, entry_price, direction, stop_loss, risk_profit_coef):
    """Calculate the take profit price based on risk/reward ratio."""
    price_precision = PERP_PRICE_PRECISION[symbol]
    if direction == 'long':
        return round(entry_price + risk_profit_coef * stop_loss, price_precision)
    elif direction == 'short':
        return round(entry_price - risk_profit_coef * stop_loss, price_precision)


def calc_commission(entry_price, exit_price, amount_per_trade, taker_rate=TAKER_RATE):
    """Calculate the trading commission for a given trade."""
    return taker_rate * (entry_price + exit_price) * amount_per_trade


def calculate_sl_commission(direction, entry_price, stop_loss, amount_per_trade):
    """Calculate the commission cost when a stop loss is triggered."""
    if direction == 'long':
        return calc_commission(entry_price, entry_price - stop_loss, amount_per_trade)
    elif direction == 'short':
        return calc_commission(entry_price, entry_price + stop_loss, amount_per_trade)


def amount_per_trade(symbol, direction, entry_price, stop_loss, usd_per_trade):
    """Determine the trade amount based on risk management parameters."""
    amount_precision = PERP_AMOUNT_PRECISION[symbol]
    coef = 1.0
    coins_per_trade = coef * usd_per_trade / stop_loss
    while coef > 0.5:
        coef -= 0.01
        coins_per_trade = coef * usd_per_trade / stop_loss
        commission = calculate_sl_commission(direction, entry_price, stop_loss, coins_per_trade)
        if (coins_per_trade * stop_loss + commission) <= usd_per_trade:
            break
    return float(round(coins_per_trade, amount_precision))


def create_new_trade(symbol, timestamp, direction, entry_price, atr, usd_per_trade):
    """Create a new trade dictionary with predefined risk parameters."""
    if not isinstance(timestamp, datetime):
        raise ValueError(f"timestamp {timestamp} must be a datetime object")

    price_precision = PERP_PRICE_PRECISION[symbol]
    entry_price = round(float(entry_price), price_precision)
    min_stop_loss = round(entry_price * MIN_STOP_LOSS_PCT, price_precision)
    max_stop_loss = round(entry_price * MAX_STOP_LOSS_PCT, price_precision)
    atr_multiplier = ATR_MULTIPLIER[symbol][direction]
    hours_per_trade = HOURS_PER_TRADE[symbol][direction]
    risk_profit_coef = RISK_PROFIT_COEF[symbol][direction]

    sl = round(stop_loss(atr, atr_multiplier, min_stop_loss, max_stop_loss), price_precision)
    sl_price = stop_loss_price(symbol, entry_price, direction, sl)
    tp_price = take_profit_price(symbol, entry_price, direction, sl, risk_profit_coef)
    coins_amount = amount_per_trade(symbol, direction, entry_price, sl, usd_per_trade)

    return {
        'symbol': symbol,
        'time': timestamp,
        'trade_dir': direction,
        't_lim': hours_per_trade,
        'coins_amount': coins_amount,
        'limit_price': entry_price,
        'stop_price': sl_price,
        'tp_price': tp_price,
        'limit_status': 'not_created',
        'stop_status': 'not_created',
        'tp_status': 'not_created',
        'atr': atr,
        'atr_mult': atr_multiplier,
        'risk_profit_coef': risk_profit_coef,
        'min_stop_loss': min_stop_loss,
        'max_stop_loss': max_stop_loss,
        'sl': sl,
        'reason_exit': 'no_exit',
        'trade_res': 0.0,
        'duration_h': 0.0
    }

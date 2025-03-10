import logging
import pandas as pd
from tabulate import tabulate
import risk_and_trade_params as tff


def update_orders_info(client, trade):
    """
    Updates trade order statuses (limit, stop-loss, and take-profit) by retrieving data from Binance API.
    Retries multiple times in case of failures and logs the process.

    :param client: Binance API client instance.
    :param trade: Dictionary containing trade details.
    :return: Updated trade dictionary with order statuses and execution details.
    """
    logging.info("\nupdate_orders_info running..\n")
    max_attempts = 5

    # Updating limit order information
    limit_info = None
    attempts = 0
    while (trade['limit_status'] in ['NEW', 'PARTIALLY_FILLED']) and (limit_info is None) and (attempts < max_attempts):
        try:
            logging.info(f"\nFetching limit order info: {trade['limit_id']}\n")
            limit_info = client.futures_get_order(symbol=trade['symbol'], origClientOrderId=trade['limit_id'])
            logging.info(f"\n{tabulate([limit_info], headers='keys', tablefmt='simple')}\n")
            trade['limit_status'] = limit_info['status']
            trade['limit_avgPrice'] = float(limit_info['avgPrice'])
            trade['limit_executedQty'] = float(limit_info['executedQty'])
        except Exception as e:
            attempts += 1
            logging.error(f"\nError fetching limit order info. Attempt {attempts}: {type(e).__name__} - {str(e)}\n")
            if attempts == max_attempts:
                logging.error(f"\nMax attempts reached. Unable to update limit order info for {trade['limit_id']}\n")

    # Updating stop-loss order information
    stop_info = None
    attempts = 0
    while (trade['stop_status'] in ['NEW']) and (stop_info is None) and (attempts < max_attempts):
        try:
            logging.info(f"\nFetching stop-loss order info: {trade['stop_id']}\n")
            stop_info = client.futures_get_order(symbol=trade['symbol'], origClientOrderId=trade['stop_id'])
            logging.info(f"\n{tabulate([stop_info], headers='keys', tablefmt='simple')}\n")
            trade['stop_status'] = stop_info['status']
            trade['stop_avgPrice'] = float(stop_info['avgPrice'])
            trade['stop_executedQty'] = float(stop_info['executedQty'])
            if stop_info['status'] == 'FILLED':
                trade['t_exit'] = pd.to_datetime(stop_info['updateTime'], unit='ms', utc=True)
                trade['reason_exit'] = 'stop_loss'
                trade['duration_h'] = round((trade['t_exit'] - trade['time']).total_seconds() / 3600, 1)
                trade['trade_res'] = tff.calc_trade_res(
                    entry_price=trade['limit_avgPrice'],
                    exit_price=trade['stop_avgPrice'],
                    amount=trade['stop_executedQty'],
                    direction=trade['trade_dir']
                )
        except Exception as e:
            attempts += 1
            logging.error(f"\nError fetching stop-loss order info. Attempt {attempts}: {type(e).__name__} - {str(e)}\n")
            if attempts == max_attempts:
                logging.error(f"\nMax attempts reached. Unable to update stop-loss order info for {trade['stop_id']}\n")

    # Updating take-profit order information
    tp_info = None
    attempts = 0
    while (trade['tp_status'] in ['NEW']) and (tp_info is None) and (attempts < max_attempts):
        try:
            logging.info(f"\nFetching take-profit order info: {trade['tp_id']}\n")
            tp_info = client.futures_get_order(symbol=trade['symbol'], origClientOrderId=trade['tp_id'])
            logging.info(f"\n{tabulate([tp_info], headers='keys', tablefmt='simple')}\n")
            trade['tp_status'] = tp_info['status']
            trade['tp_avgPrice'] = float(tp_info['avgPrice'])
            trade['tp_executedQty'] = float(tp_info['executedQty'])
            if tp_info['status'] == 'FILLED':
                trade['t_exit'] = pd.to_datetime(tp_info['updateTime'], unit='ms', utc=True)
                trade['reason_exit'] = 'take_profit'
                trade['duration_h'] = round((trade['t_exit'] - trade['time']).total_seconds() / 3600, 1)
                trade['trade_res'] = tff.calc_trade_res(
                    entry_price=trade['limit_avgPrice'],
                    exit_price=trade['tp_avgPrice'],
                    amount=trade['tp_executedQty'],
                    direction=trade['trade_dir']
                )
        except Exception as e:
            attempts += 1
            logging.error(
                f"\nError fetching take-profit order info. Attempt {attempts}: {type(e).__name__} - {str(e)}\n"
                )
            if attempts == max_attempts:
                logging.error(
                    f"\nMax attempts reached. Unable to update take-profit order info for {trade['tp_id']}\n"
                    )

    logging.info("\nupdate_orders_info() has completed.\n")
    return trade


def open_orders(client, trade):
    """
    Places limit, stop-loss, and take-profit orders for a trade.
    Retries multiple times in case of failures and logs the process.

    :param client: Binance API client instance.
    :param trade: Dictionary containing trade details.
    :return: Updated trade dictionary with order statuses.
    """
    logging.info("\nopen_orders running..\n")
    max_attempts = 5

    # Define trade direction settings
    if trade['trade_dir'].lower() == 'long':
        side, side_opposite, positionSide = 'BUY', 'SELL', 'LONG'
    elif trade['trade_dir'].lower() == 'short':
        side, side_opposite, positionSide = 'SELL', 'BUY', 'SHORT'

    # Limit order placement
    attempts = 0
    while trade['limit_status'] == 'not_created' and attempts < max_attempts:
        try:
            logging.info(f"\nCreating LIMIT order {trade['limit_id']}\n")
            limit_order = client.futures_create_order(
                symbol=trade['symbol'],
                side=side,  # 'BUY' / 'SELL'
                positionSide=positionSide,  # 'LONG' / 'SHORT'
                type=client.FUTURE_ORDER_TYPE_MARKET,
                quantity=trade['coins_amount'],
                newClientOrderId=trade['limit_id']
            )
            logging.info(f"\n{tabulate([limit_order], headers='keys', tablefmt='simple')}\n")
            trade['limit_status'] = limit_order['status']
            trade['limit_avgPrice'] = float(limit_order['avgPrice'])
            trade['limit_executedQty'] = float(limit_order['executedQty'])
        except Exception as e:
            attempts += 1
            logging.error(f"\nError creating LIMIT order. Attempt {attempts}: {type(e).__name__} - {str(e)}\n")
            if attempts == max_attempts:
                logging.error(f"\nMax attempts reached. LIMIT order {trade['limit_id']} not created.\n")

    # Stop-loss order placement
    attempts = 0
    while (
        trade['limit_status'] in ['NEW', 'FILLED', 'PARTIALLY_FILLED']
        and trade['stop_status'] == 'not_created'
        and attempts < max_attempts
    ):
        try:
            logging.info(f"\nCreating STOP order {trade['stop_id']}\n")
            stop_order = client.futures_create_order(
                symbol=trade['symbol'],
                side=side_opposite,  # 'BUY' / 'SELL'
                positionSide=positionSide,  # 'LONG' / 'SHORT'
                type=client.FUTURE_ORDER_TYPE_STOP_MARKET,
                quantity=trade['coins_amount'],
                stopPrice=trade['stop_price'],
                newClientOrderId=trade['stop_id']
            )
            logging.info(f"\n{tabulate([stop_order], headers='keys', tablefmt='simple')}\n")
            trade['stop_status'] = stop_order['status']
        except Exception as e:
            attempts += 1
            logging.error(f"\nError creating STOP order. Attempt {attempts}: {type(e).__name__} - {str(e)}\n")
            if attempts == max_attempts:
                logging.error(f"\nMax attempts reached. STOP order {trade['stop_id']} not created.\n")
        trade = update_orders_info(client, trade)

    # Take-profit order placement
    attempts = 0
    while (
        trade['limit_status'] in ['FILLED', 'PARTIALLY_FILLED']
        and trade['stop_status'] not in ['FILLED']
        and trade['tp_status'] == 'not_created'
        and attempts < max_attempts
    ):
        try:
            logging.info(f"\nCreating TAKE PROFIT order {trade['tp_id']}\n")
            tp_order = client.futures_create_order(
                symbol=trade['symbol'],
                side=side_opposite,  # 'BUY' / 'SELL'
                positionSide=positionSide,  # 'LONG' / 'SHORT'
                type=client.FUTURE_ORDER_TYPE_LIMIT,
                timeInForce='GTC',
                quantity=trade['coins_amount'],
                price=trade['tp_price'],
                newClientOrderId=trade['tp_id']
            )
            logging.info(f"\n{tabulate([tp_order], headers='keys', tablefmt='simple')}\n")
            trade['tp_status'] = tp_order['status']
        except Exception as e:
            attempts += 1
            logging.error(f"\nError creating TAKE PROFIT order. Attempt {attempts}: {type(e).__name__} - {str(e)}\n")
            if attempts == max_attempts:
                logging.error(f"\nMax attempts reached. TAKE PROFIT order {trade['tp_id']} not created.\n")

    logging.info("\nopen_orders() has completed.\n")
    return trade


def cancel_orders(client, trade):
    """
    Cancels open orders (limit, stop-loss, and take-profit) for a given trade.
    Retries multiple times in case of failures and logs the process.

    :param client: Binance API client instance.
    :param trade: Dictionary containing trade details.
    :return: Updated trade dictionary with order statuses.
    """
    logging.info("\ncancel_orders running..\n")

    trade = update_orders_info(client, trade)
    max_attempts = 5

    # Cancel limit order
    attempts = 0
    while (
        trade['limit_status'] in ['NEW', 'PARTIALLY_FILLED']
        and attempts < max_attempts
    ):
        try:
            logging.info(
                f"\nCancelling LIMIT order {trade['limit_id']} (status: {trade['limit_status']})\n"
            )
            cancel_limit = client.futures_cancel_order(
                symbol=trade['symbol'],
                origClientOrderId=trade['limit_id']
            )
            logging.info(f"\n{tabulate([cancel_limit], headers='keys', tablefmt='simple')}\n")
            trade['limit_status'] = cancel_limit['status']
        except Exception as e:
            attempts += 1
            logging.error(
                f"\nError cancelling LIMIT order (Attempt {attempts}): {type(e).__name__} - {str(e)}\n"
            )
            if attempts == max_attempts:
                logging.error(
                    f"\nMax attempts reached. LIMIT order {trade['limit_id']} not canceled.\n"
                )

    # Cancel stop-loss order
    attempts = 0
    while trade['stop_status'] in ['NEW'] and attempts < max_attempts:
        try:
            logging.info(
                f"\nCancelling STOP order {trade['stop_id']} (status: {trade['stop_status']})\n"
            )
            cancel_stop = client.futures_cancel_order(
                symbol=trade['symbol'],
                origClientOrderId=trade['stop_id']
            )
            logging.info(f"\n{tabulate([cancel_stop], headers='keys', tablefmt='simple')}\n")
            trade['stop_status'] = cancel_stop['status']
        except Exception as e:
            attempts += 1
            logging.error(
                f"\nError cancelling STOP order (Attempt {attempts}): {type(e).__name__} - {str(e)}\n"
            )
            if attempts == max_attempts:
                logging.error(
                    f"\nMax attempts reached. STOP order {trade['stop_id']} not canceled.\n"
                )

    # Cancel take-profit order
    attempts = 0
    while (
        trade['tp_status'] in ['NEW', 'PARTIALLY_FILLED']
        and attempts < max_attempts
    ):
        try:
            logging.info(
                f"\nCancelling TAKE PROFIT order {trade['tp_id']} (status: {trade['tp_status']})\n"
            )
            cancel_tp = client.futures_cancel_order(
                symbol=trade['symbol'],
                origClientOrderId=trade['tp_id']
            )
            logging.info(f"\n{tabulate([cancel_tp], headers='keys', tablefmt='simple')}\n")
            trade['tp_status'] = cancel_tp['status']
        except Exception as e:
            attempts += 1
            logging.error(
                f"\nError cancelling TAKE PROFIT order (Attempt {attempts}): {type(e).__name__} - {str(e)}\n"
            )
            if attempts == max_attempts:
                logging.error(
                    f"\nMax attempts reached. TAKE PROFIT order {trade['tp_id']} not canceled.\n"
                )

    logging.info("\ncancel_orders() has completed.\n")
    return trade


def close_orders(client, trade):
    """
    Closes an open trade by placing a market order if the stop-loss or take-profit has not been filled.
    Retries multiple times in case of failures and logs the process.

    :param client: Binance API client instance.
    :param trade: Dictionary containing trade details.
    :return: Updated trade dictionary with order statuses.
    """
    logging.info("\nclose_orders running..\n")
    max_attempts = 5

    # Define closing order direction
    if trade['trade_dir'].lower() == 'long':
        side, positionSide = 'SELL', 'LONG'
    elif trade['trade_dir'].lower() == 'short':
        side, positionSide = 'BUY', 'SHORT'

    market_order = None
    attempts = 0
    while (
        trade['limit_status'] in ['FILLED', 'PARTIALLY_FILLED']
        and trade['stop_status'] != 'FILLED'
        and trade['tp_status'] != 'FILLED'
        and market_order is None
        and attempts < max_attempts
    ):
        try:
            logging.info(f"\nClosing order {trade['limit_id']} via MARKET order\n")

            # Fetch the latest take-profit order info
            logging.info(f"\nUpdating TP order info: {trade['tp_id']}\n")
            tp_info = client.futures_get_order(
                symbol=trade['symbol'], origClientOrderId=trade['tp_id']
            )
            logging.info(f"\n{tabulate([tp_info], headers='keys', tablefmt='simple')}\n")

            # Create a market order for the remaining position size
            market_order = client.futures_create_order(
                symbol=trade['symbol'],
                side=side,
                positionSide=positionSide,
                type=client.FUTURE_ORDER_TYPE_MARKET,
                quantity=round(
                    trade['limit_executedQty'] - float(tp_info['executedQty']),
                    tff.PERP_AMOUNT_PRECISION[trade['symbol']]
                )
            )
            logging.info(f"\nMarket order placed:\n{tabulate([market_order], headers='keys', tablefmt='simple')}\n")

            # Fetch updated market order info
            new_market_info = client.futures_get_order(
                symbol=trade['symbol'], orderId=market_order['orderId']
            )
            logging.info(f"\nMarket order status updated: {new_market_info['status']}\n")

            if new_market_info['status'] == 'FILLED':
                logging.info(f"\nCancelling remaining orders for {trade['limit_id']}\n")

                trade['t_exit'] = pd.to_datetime(
                    new_market_info['updateTime'], unit='ms', utc=True
                )
                trade['reason_exit'] = 'out_of_time'
                trade['duration_h'] = round(
                    (trade['t_exit'] - trade['time']).total_seconds() / 3600, 1
                )
                trade['exit_avgPrice'] = float(new_market_info['avgPrice'])
                trade['exit_executedQty'] = float(new_market_info['executedQty'])
                trade['trade_res'] = tff.calc_trade_res(
                    entry_price=trade['limit_avgPrice'],
                    exit_price=trade['exit_avgPrice'],
                    amount=trade['exit_executedQty'],
                    direction=trade['trade_dir']
                )

                trade = cancel_orders(client, trade)
                trade['limit_status'] = 'CLOSED'
                logging.info(f"\nLimit order {trade['limit_id']} closed by market.\n")
            else:
                logging.info("\nMarket order not FILLED, retrying..\n")
                market_order = None
                attempts += 1
        except Exception as e:
            attempts += 1
            logging.error(
                f"\nError closing market order (Attempt {attempts}): {type(e).__name__} - {str(e)}\n"
            )
            if attempts == max_attempts:
                logging.error(
                    "\nMax attempts reached. MARKET order was not created.\n"
                )

    logging.info("\nclose_orders() has completed.\n")
    return trade

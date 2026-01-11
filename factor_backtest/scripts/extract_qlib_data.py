"""
Extract CSI500 data from qlib and save as standalone files.
This script creates decoupled data files for factor backtesting.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

import qlib
from qlib.constant import REG_CN
from qlib.data import D


def extract_csi500_data(
    output_dir: str = "../data",
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    use_all_stocks: bool = True,
):
    """Extract CSI500 related data from qlib
    
    Args:
        use_all_stocks: If True, extract all A-share stocks (required for accurate backtest).
                       If False, only extract CSI500 constituents (may miss historical stocks).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize qlib
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    logger.info("Qlib initialized")
    
    # 1. Get stock pool
    # Use 'all' market to get all A-share stocks for complete coverage
    # This is important because factor predictions may include historical CSI500 constituents
    if use_all_stocks:
        logger.info("Extracting ALL A-share stocks for complete coverage...")
        instruments = D.instruments(market="all")
    else:
        logger.info("Extracting CSI500 stock pool only...")
        instruments = D.instruments(market="csi500")
    stock_list = D.list_instruments(instruments=instruments, start_time=start_date, end_time=end_date, as_list=True)
    
    # Save stock pool
    stock_pool_path = output_path / "csi500_stock_pool.txt"
    with open(stock_pool_path, "w") as f:
        for stock in sorted(stock_list):
            f.write(f"{stock}\n")
    logger.info(f"Saved {len(stock_list)} stocks to {stock_pool_path}")
    
    # 2. Get daily price data (OHLCV + change for limit detection)
    logger.info("Extracting daily price data...")
    fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$factor", "$change"]
    price_df = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_date,
        end_time=end_date,
    )
    price_df.columns = ["open", "high", "low", "close", "volume", "vwap", "adj_factor", "change"]
    price_df = price_df.reset_index()
    price_df = price_df.rename(columns={"datetime": "date", "instrument": "symbol"})
    
    # Save price data
    price_path = output_path / "csi500_daily_price.parquet"
    price_df.to_parquet(price_path, index=False)
    logger.info(f"Saved price data ({len(price_df)} rows) to {price_path}")
    
    # 3. Get benchmark data (CSI500 index: SH000905)
    logger.info("Extracting benchmark data...")
    benchmark_df = D.features(
        instruments=["SH000905"],
        fields=["$close"],
        start_time=start_date,
        end_time=end_date,
    )
    benchmark_df.columns = ["close"]
    benchmark_df = benchmark_df.reset_index()
    benchmark_df = benchmark_df.rename(columns={"datetime": "date", "instrument": "symbol"})
    benchmark_df["return"] = benchmark_df["close"].pct_change()
    
    # Save benchmark data
    benchmark_path = output_path / "csi500_benchmark.parquet"
    benchmark_df.to_parquet(benchmark_path, index=False)
    logger.info(f"Saved benchmark data ({len(benchmark_df)} rows) to {benchmark_path}")
    
    # 4. Calculate and save daily returns for all stocks
    # IMPORTANT: Use t+2/t+1 - 1 to match Alpha158 label
    # This is the return from buying at t+1 close and selling at t+2 close
    # (Factor is known at t close, trade at t+1, realize return at t+2)
    logger.info("Calculating daily returns (t+2/t+1 - 1)...")
    returns_df = D.features(
        instruments=instruments,
        fields=["Ref($close, -2)/Ref($close, -1) - 1"],  # t+2 close / t+1 close - 1
        start_time=start_date,
        end_time=end_date,
    )
    returns_df.columns = ["return_1d"]
    returns_df = returns_df.reset_index()
    returns_df = returns_df.rename(columns={"datetime": "date", "instrument": "symbol"})
    
    # Save returns data
    returns_path = output_path / "csi500_daily_returns.parquet"
    returns_df.to_parquet(returns_path, index=False)
    logger.info(f"Saved returns data ({len(returns_df)} rows) to {returns_path}")
    
    # 5. Summary statistics
    logger.info("\n" + "="*50)
    logger.info("DATA EXTRACTION SUMMARY")
    logger.info("="*50)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Number of stocks: {len(stock_list)}")
    logger.info(f"Price data rows: {len(price_df)}")
    logger.info(f"Returns data rows: {len(returns_df)}")
    logger.info(f"Output directory: {output_path.absolute()}")
    
    return {
        "stock_pool": stock_list,
        "price_df": price_df,
        "benchmark_df": benchmark_df,
        "returns_df": returns_df,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract CSI500 data from qlib")
    parser.add_argument("--output_dir", type=str, default="../data", help="Output directory")
    parser.add_argument("--start_date", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--csi500_only", action="store_true", 
                        help="Only extract CSI500 constituents (default: extract all stocks)")
    
    args = parser.parse_args()
    
    extract_csi500_data(
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        use_all_stocks=not args.csi500_only,
    )

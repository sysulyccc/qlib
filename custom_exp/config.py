# Rolling training configuration for qlib experiments
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class RollingConfig:
    """Rolling training configuration"""
    # Data range
    data_start: str = "2010-01-01"
    data_end: str = "2024-12-31"
    
    # Rolling parameters: 3 years train+valid, 1 year test
    train_years: int = 3  # train + valid years
    test_years: int = 1   # test years
    train_valid_ratio: float = 0.8  # 4:1 split for train:valid
    
    # Market settings
    market: str = "csi500"
    benchmark: str = "SH000905"
    
    # Provider
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    
    # Output directories (use absolute path)
    output_dir: str = "/home/ethan/qlib/custom_exp"
    
    # Ensemble settings
    ensemble_weight_current: float = 0.5  # weight for current model
    # remaining 0.5 is split equally among historical models


@dataclass  
class ModelConfig:
    """Model configuration"""
    name: str
    model_class: str
    module_path: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    is_ts: bool = False  # whether it's a time-series model requiring TSDatasetH
    step_len: int = 20   # sequence length for time-series models


# Pre-defined model configurations
LIGHTGBM_CONFIG = ModelConfig(
    name="lightgbm",
    model_class="LGBModel",
    module_path="qlib.contrib.model.gbdt",
    kwargs={
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    }
)

XGBOOST_CONFIG = ModelConfig(
    name="xgboost",
    model_class="XGBModel",
    module_path="qlib.contrib.model.xgboost",
    kwargs={
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
    }
)

LINEAR_CONFIG = ModelConfig(
    name="linear",
    model_class="LinearModel",
    module_path="qlib.contrib.model.linear",
    kwargs={}
)

MLP_CONFIG = ModelConfig(
    name="mlp",
    model_class="DNNModelPytorch",
    module_path="qlib.contrib.model.pytorch_nn",
    kwargs={
        "d_feat": 158,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.1,
        "n_epochs": 100,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2048,
        "GPU": 0,
    }
)

LSTM_CONFIG = ModelConfig(
    name="lstm",
    model_class="LSTM",
    module_path="qlib.contrib.model.pytorch_lstm_ts",
    kwargs={
        "d_feat": 158,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 200,
        "lr": 1e-3,
        "early_stop": 20,
        "batch_size": 800,
        "metric": "loss",
        "loss": "mse",
        "GPU": 0,
    },
    is_ts=True,
    step_len=20,
)

GRU_CONFIG = ModelConfig(
    name="gru",
    model_class="GRU",
    module_path="qlib.contrib.model.pytorch_gru_ts",
    kwargs={
        "d_feat": 158,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 200,
        "lr": 2e-4,
        "early_stop": 10,
        "batch_size": 800,
        "metric": "loss",
        "loss": "mse",
        "GPU": 0,
    },
    is_ts=True,
    step_len=20,
)

ALSTM_CONFIG = ModelConfig(
    name="alstm",
    model_class="ALSTM",
    module_path="qlib.contrib.model.pytorch_alstm_ts",
    kwargs={
        "d_feat": 158,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 200,
        "lr": 1e-3,
        "early_stop": 20,
        "batch_size": 800,
        "metric": "loss",
        "loss": "mse",
        "rnn_type": "GRU",
        "GPU": 0,
    },
    is_ts=True,
    step_len=20,
)

TRANSFORMER_CONFIG = ModelConfig(
    name="transformer",
    model_class="Transformer",
    module_path="qlib.contrib.model.pytorch_transformer_ts",
    kwargs={
        "d_feat": 158,
        "d_model": 64,
        "nhead": 2,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 200,
        "lr": 1e-4,
        "early_stop": 20,
        "batch_size": 800,
        "metric": "loss",
        "loss": "mse",
        "GPU": 0,
    },
    is_ts=True,
    step_len=20,
)

# All available models
ALL_MODELS = {
    "lightgbm": LIGHTGBM_CONFIG,
    "xgboost": XGBOOST_CONFIG,
    "linear": LINEAR_CONFIG,
    "mlp": MLP_CONFIG,
    "lstm": LSTM_CONFIG,
    "gru": GRU_CONFIG,
    "alstm": ALSTM_CONFIG,
    "transformer": TRANSFORMER_CONFIG,
}


def get_backtest_config(benchmark: str = "SH000905", strategy_type: str = "topk"):
    """Get backtest configuration
    
    Args:
        benchmark: benchmark index
        strategy_type: "topk" for TopkDropoutStrategy, "long_only" for simple top decile long only
    """
    if strategy_type == "topk":
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": None,
                "topk": 50,
                "n_drop": 5,
            },
        }
    elif strategy_type == "long_only":
        # Top decile (10%) long only strategy
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": None,
                "topk": 50,  # will be dynamically set based on universe size / 10
                "n_drop": 0,  # no dropout, simple rebalance
            },
        }
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": strategy_config,
        "backtest": {
            "start_time": None,
            "end_time": None,
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

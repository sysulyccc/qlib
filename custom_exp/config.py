# Rolling training configuration for qlib experiments
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class RollingConfig:
    """Rolling training configuration"""
    # Data range
    data_start: str = "2010-01-01"
    data_end: str = "2024-12-31"
    
    # Rolling parameters: 6 years train+valid, 1 year test
    train_years: int = 6
    test_years: int = 1
    train_valid_ratio: float = 0.8
    
    # Market settings
    market: str = "csi500"
    benchmark: str = "SH000905"
    
    # Provider
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    
    # Output directories (use absolute path)
    output_dir: str = "/home/ethan/qlib/custom_exp"
    
    # Ensemble settings
    ensemble_weight_current: float = 0.5

@dataclass  
class ModelConfig:
    """Model configuration"""
    name: str
    model_class: str
    module_path: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Data Handler settings
    handler_class: str = "Alpha158"  # Alpha158 or Alpha360
    handler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Processor settings
    learn_processors: Optional[List[Dict]] = None
    infer_processors: Optional[List[Dict]] = None
    
    # Dataset settings
    dataset_class: str = "DatasetH"  # DatasetH, TSDatasetH, or MTSDatasetH
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    is_ts: bool = False  # Legacy
    step_len: int = 20   # Legacy


# --- Constants & Common Processors ---

ALPHA158_GRU_FEATURES = [
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", 
    "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", 
    "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
]

def get_processors(norm_type="robust", fields_group="feature", filter_cols=None, drop_vwap=False):
    """Generate processors based on normalization type"""
    
    infer_procs = []
    learn_procs = []
    
    # 1. Filter columns (if needed)
    if filter_cols:
        infer_procs.append({"class": "FilterCol", "kwargs": {"fields_group": fields_group, "col_list": filter_cols}})
    
    # 2. Drop specific columns (e.g. VWAP0 for MLP 158)
    if drop_vwap:
        infer_procs.append({"class": "DropCol", "kwargs": {"col_list": ["VWAP0"]}})
        learn_procs.append({"class": "DropCol", "kwargs": {"col_list": ["VWAP0"]}})

    # 3. Normalization
    if norm_type == "robust":
        # RobustZScoreNorm
        infer_procs.append({"class": "RobustZScoreNorm", "kwargs": {"fields_group": fields_group, "clip_outlier": True}})
    elif norm_type == "zscore":
        # ZScoreNorm
        infer_procs.append({"class": "ZScoreNorm", "kwargs": {"fields_group": fields_group}})
    elif norm_type == "none":
        pass
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
        
    # 4. Fillna (Always needed)
    if norm_type == "robust":
        # RobustZScore usually pairs with Fillna
        infer_procs.append({"class": "Fillna", "kwargs": {"fields_group": fields_group}})
    elif norm_type == "zscore":
         # ZScore usually pairs with CSZFillna? Or just Fillna? 
         # Benchmarks often use CSZFillna for MLP 158.
         # Let's stick to Fillna for simplicity unless it's MLP 158 which used CSZFillna in original config.
         # But to be consistent across "zscore" variants, let's use Fillna or CSZFillna.
         # For TimeSeries, usually Fillna.
         infer_procs.append({"class": "Fillna", "kwargs": {"fields_group": fields_group}})
         
    # 5. Label processing (Learn only)
    # MLP 158 used DropnaProcessor, DropnaLabel, CSZScoreNorm for label.
    # Others used DropnaLabel, CSRankNorm for label.
    # We will standardize on DropnaLabel + CSRankNorm for label for most, 
    # except MLP/LGB might prefer something else? 
    # Let's stick to the previous config defaults for Label: CSRankNorm is robust.
    
    learn_procs.append({"class": "DropnaLabel"})
    learn_procs.append({"class": "CSRankNorm", "kwargs": {"fields_group": "label"}})
    
    return infer_procs, learn_procs

# --- Model Configurations ---

ALL_MODELS = {}

# Experiment Factors
NORMS = ["zscore", "robust"]
STEP_LENS = [10, 20, 30]

# 1. LightGBM (LGB)
# Need to ensure qlib.contrib.model.gbdt is available
for dataset in ["158", "360"]:
    for norm in NORMS:
        name = f"lgb_{dataset}_{norm}"
        
        # Handler & Input dim
        handler = "Alpha158" if dataset == "158" else "Alpha360"
        
        # Processors
        # For LGB, we don't necessarily need normalization, but user asked for standardization methods.
        # LGB handles unscaled data well, but we'll apply it as requested.
        infer_p, learn_p = get_processors(norm, drop_vwap=(dataset=="158"))
        
        ALL_MODELS[name] = ModelConfig(
            name=name,
            model_class="LGBModel",
            module_path="qlib.contrib.model.gbdt",
            kwargs={
                "loss": "mse",
                "early_stopping_rounds": 50,
                "num_boost_round": 1000,
                "GPU": 0, # Placeholder, set by runner
            },
            handler_class=handler,
            infer_processors=infer_p,
            learn_processors=learn_p,
        )

# 2. MLP
for dataset in ["158", "360"]:
    for norm in NORMS:
        name = f"mlp_{dataset}_{norm}"
        handler = "Alpha158" if dataset == "158" else "Alpha360"
        input_dim = 157 if dataset == "158" else 360 # 158 - VWAP0 = 157
        
        infer_p, learn_p = get_processors(norm, drop_vwap=(dataset=="158"))
        
        ALL_MODELS[name] = ModelConfig(
            name=name,
            model_class="DNNModelPytorch",
            module_path="qlib.contrib.model.pytorch_nn",
            kwargs={
                "loss": "mse",
                "lr": 0.002,
                "optimizer": "adam",
                "max_steps": 8000,
                "batch_size": 4096 if dataset == "360" else 8192,
                "GPU": 0,
                "pt_model_kwargs": {"input_dim": input_dim}
            },
            handler_class=handler,
            infer_processors=infer_p,
            learn_processors=learn_p,
        )

# 3. GRU
# 3.1 Alpha158 (Time Series, Filtered to 20 feats)
for step in STEP_LENS:
    for norm in NORMS:
        name = f"gru_158_{step}_{norm}"
        
        infer_p, learn_p = get_processors(norm, filter_cols=ALPHA158_GRU_FEATURES)
        
        ALL_MODELS[name] = ModelConfig(
            name=name,
            model_class="GRU",
            module_path="qlib.contrib.model.pytorch_gru_ts",
            kwargs={
                "d_feat": 20,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "n_epochs": 200,
                "lr": 2e-4,
                "early_stop": 10,
                "batch_size": 800,
                "metric": "loss",
                "loss": "mse",
                "n_jobs": 20,
                "GPU": 0,
            },
            handler_class="Alpha158",
            dataset_class="TSDatasetH",
            dataset_kwargs={"step_len": step},
            infer_processors=infer_p,
            learn_processors=learn_p,
        )

# 3.2 Alpha360 (Sequence is inherent in data, but reshaped in model)
for norm in NORMS:
    name = f"gru_360_{norm}"
    infer_p, learn_p = get_processors(norm)
    
    ALL_MODELS[name] = ModelConfig(
        name=name,
        model_class="GRU",
        module_path="qlib.contrib.model.pytorch_gru",
        kwargs={
            "d_feat": 6, # Alpha360 is 60x6
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
        handler_class="Alpha360",
        # Default DatasetH, no step_len needed as it consumes 360 columns which model reshapes
        infer_processors=infer_p,
        learn_processors=learn_p,
    )

# 4. Transformer
# 4.1 Alpha158
for step in STEP_LENS:
    for norm in NORMS:
        name = f"transformer_158_{step}_{norm}"
        infer_p, learn_p = get_processors(norm, filter_cols=ALPHA158_GRU_FEATURES)
        
        ALL_MODELS[name] = ModelConfig(
            name=name,
            model_class="TransformerModel",
            module_path="qlib.contrib.model.pytorch_transformer_ts",
            kwargs={
                "d_feat": 20,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 2,
                "dropout": 0.0,
                "lr": 2e-4,
                "early_stop": 10,
                "batch_size": 800,
                "metric": "loss",
                "loss": "mse",
                "GPU": 0,
                "seed": 0,
                "n_jobs": 20,
            },
            handler_class="Alpha158",
            dataset_class="TSDatasetH",
            dataset_kwargs={"step_len": step},
            infer_processors=infer_p,
            learn_processors=learn_p,
        )

# 4.2 Alpha360
for norm in NORMS:
    name = f"transformer_360_{norm}"
    infer_p, learn_p = get_processors(norm)
    
    ALL_MODELS[name] = ModelConfig(
        name=name,
        model_class="TransformerModel",
        module_path="qlib.contrib.model.pytorch_transformer",
        kwargs={
            "d_feat": 6,
            "seed": 0,
            "n_epochs": 200,
            "lr": 1e-3,
            "early_stop": 10,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "GPU": 0,
        },
        handler_class="Alpha360",
        infer_processors=infer_p,
        learn_processors=learn_p,
    )

# 5. Advanced Models (Fixed Configs)
# HIST
ALL_MODELS["hist_360"] = ModelConfig(
    name="hist_360",
    model_class="HIST",
    module_path="qlib.contrib.model.pytorch_hist",
    kwargs={
        "d_feat": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 200,
        "lr": 1e-4,
        "early_stop": 20,
        "metric": "ic",
        "loss": "mse",
        "base_model": "LSTM",
        "model_path": None, # Train from scratch
        "stock2concept": "/home/ethan/qlib/examples/benchmarks/HIST/qlib_csi300_stock2concept.npy",
        "stock_index": "/home/ethan/qlib/examples/benchmarks/HIST/qlib_csi300_stock_index.npy",
        "GPU": 0,
    },
    handler_class="Alpha360",
    infer_processors=[
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    learn_processors=[
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
)

# IGMTF
ALL_MODELS["igmtf_360"] = ModelConfig(
    name="igmtf_360",
    model_class="IGMTF",
    module_path="qlib.contrib.model.pytorch_igmtf",
    kwargs={
        "d_feat": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 200,
        "lr": 1e-4,
        "early_stop": 20,
        "metric": "ic",
        "loss": "mse",
        "base_model": "LSTM",
        "model_path": None,
        "GPU": 0,
    },
    handler_class="Alpha360",
    infer_processors=[
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    learn_processors=[
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
)

# SFM
ALL_MODELS["sfm_360"] = ModelConfig(
    name="sfm_360",
    model_class="SFM",
    module_path="qlib.contrib.model.pytorch_sfm",
    kwargs={
        "d_feat": 6,
        "hidden_size": 64,
        "output_dim": 32,
        "freq_dim": 25,
        "dropout_W": 0.5,
        "dropout_U": 0.5,
        "n_epochs": 20,
        "lr": 1e-3,
        "batch_size": 1600,
        "early_stop": 20,
        "eval_steps": 5,
        "loss": "mse",
        "optimizer": "adam",
        "GPU": 0,
    },
    handler_class="Alpha360",
    infer_processors=[
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    learn_processors=[
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
)

# TRA
ALL_MODELS["tra_360"] = ModelConfig(
    name="tra_360",
    model_class="TRAModel",
    module_path="qlib.contrib.model.pytorch_tra",
    kwargs={
        "tra_config": {
            "num_states": 3,
            "rnn_arch": "LSTM",
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "tau": 1.0,
            "src_info": "LR_TPE",
        },
        "model_config": {
            "input_size": 6,
            "hidden_size": 64,
            "num_layers": 2,
            "rnn_arch": "LSTM",
            "use_attn": True,
            "dropout": 0.0,
        },
        "model_type": "RNN",
        "lr": 1e-3,
        "n_epochs": 100,
        "early_stop": 20,
        "lamb": 1.0,
        "rho": 0.99,
        "alpha": 0.5,
        "transport_method": "router",
        "memory_mode": "sample",
        "eval_train": False,
        "eval_test": True,
        "pretrain": True,
        "freeze_model": False,
        "freeze_predictors": False,
    },
    handler_class="Alpha360",
    dataset_class="MTSDatasetH",
    dataset_kwargs={
        "seq_len": 60,
        "input_size": 6,
        "num_states": 3,
        "batch_size": 1024,
        "memory_mode": "sample",
        "drop_last": True,
    },
    infer_processors=[
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    learn_processors=[
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
)


def get_backtest_config(benchmark: str = "SH000905", strategy_type: str = "topk"):
    """Get backtest configuration"""
    if strategy_type == "topk":
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": None,
                "topk": 50,
                "n_drop": 5,
            }
        }
    elif strategy_type == "long_only":
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": None,
                "topk": 50,
                "n_drop": 0,
            }
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

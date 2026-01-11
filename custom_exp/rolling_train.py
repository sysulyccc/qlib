"""
Rolling training framework for qlib models (v4).
Features:
- Fixed ensemble logic: re-predict with historical models on current test data
- Time-series model support with TSDatasetH + step_len
- Model name includes step_len for TS models
- Full backtest with model performance graphs
- Strategy name in file naming
- Test mode for quick validation
"""
import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.eva.alpha import calc_ic
from qlib.backtest import backtest as normal_backtest
from qlib.contrib.report import analysis_model, analysis_position

import plotly.io as pio

from config import (
    RollingConfig, ModelConfig, ALL_MODELS,
    LIGHTGBM_CONFIG, get_backtest_config
)

warnings.filterwarnings("ignore")


@dataclass
class FoldResult:
    """Results for a single fold"""
    fold_id: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    pred_df: pd.DataFrame
    pred_normalized: pd.DataFrame
    label_df: pd.DataFrame
    ic: float
    icir: float
    rank_ic: float
    rank_icir: float
    backtest_metrics: Dict[str, Any]
    ensemble_metrics: Optional[Dict[str, float]] = None
    ensemble_backtest: Optional[Dict[str, Any]] = None


class RollingTrainer:
    """Rolling training manager with fixed ensemble logic"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        rolling_config: RollingConfig = None,
        gpu_id: int = 0,
        strategy_type: str = "topk",
        test_mode: bool = False,
    ):
        self.model_config = model_config
        self.config = rolling_config or RollingConfig()
        self.gpu_id = gpu_id
        self.strategy_type = strategy_type
        self.test_mode = test_mode
        
        # Initialize qlib
        qlib.init(provider_uri=self.config.provider_uri, region=REG_CN)
        
        # Build model name with key params
        self.model_name = self._build_model_name()
        
        # Setup output directories
        self.base_dir = Path(self.config.output_dir)
        self.pred_dir = self.base_dir / "model_pred" / self.model_name
        self.weight_dir = self.base_dir / "weights" / self.model_name
        self.pdf_dir = self.base_dir / "pdf" / self.model_name
        self.log_dir = self.base_dir / "log"
        
        for d in [self.pred_dir, self.weight_dir, self.pdf_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to file
        self.log_file = self._setup_logger()
        logger.info(f"Logging to {self.log_file}")
        logger.info(f"Model name: {self.model_name}")
        if self.test_mode:
            logger.info("*** TEST MODE: Using minimal epochs ***")
        
        # Storage for fold results
        self.fold_results: List[FoldResult] = []
        self.historical_models: List[Any] = []
        self.all_periods = None
        
        # Ensemble analysis matrix: rows=test_fold, cols=model_from_fold
        self.ensemble_ic_matrix: Dict[int, Dict[int, float]] = {}
    
    def _build_model_name(self) -> str:
        """Build model name with key parameters"""
        parts = [self.model_config.name]
        
        # Add step_len for time-series models
        if self.model_config.is_ts:
            parts.append(f"{self.model_config.step_len}")
        
        # Add market
        parts.append(self.config.market)
        
        # Add strategy type
        parts.append(self.strategy_type)
        
        return "_".join(parts)
    
    def _setup_logger(self) -> Path:
        """Setup loguru to output to both console and file"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.model_name}_{timestamp}.log"
        
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, level="DEBUG", rotation="100 MB")
        
        return log_file
    
    def _generate_rolling_periods(self) -> List[Tuple[str, str, str, str, str, str]]:
        """Generate rolling periods based on config."""
        periods = []
        
        data_start = pd.Timestamp(self.config.data_start)
        data_end = pd.Timestamp(self.config.data_end)
        
        train_valid_years = self.config.train_years
        test_years = self.config.test_years
        train_ratio = self.config.train_valid_ratio
        
        train_years = int(train_valid_years * train_ratio)
        valid_years = train_valid_years - train_years
        
        current_start = data_start
        
        while True:
            train_start = current_start
            train_end = train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
            valid_start = train_end + pd.DateOffset(days=1)
            valid_end = valid_start + pd.DateOffset(years=valid_years) - pd.DateOffset(days=1)
            test_start = valid_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)
            
            if test_start > data_end:
                break
            
            if test_end > data_end:
                test_end = data_end
            
            periods.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                valid_start.strftime("%Y-%m-%d"),
                valid_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))
            
            current_start = current_start + pd.DateOffset(years=test_years)
        
        return periods
    
    def _create_dataset(
        self,
        train_seg: Tuple[str, str],
        valid_seg: Tuple[str, str],
        test_seg: Tuple[str, str],
        fit_start: str,
        fit_end: str,
    ):
        """Create dataset with given segments - use TSDatasetH for time-series models"""
        handler_config = {
            "start_time": train_seg[0],
            "end_time": test_seg[1],
            "fit_start_time": fit_start,
            "fit_end_time": fit_end,
            "instruments": self.config.market,
        }
        handler = Alpha158(**handler_config)
        
        segments = {
            "train": train_seg,
            "valid": valid_seg,
            "test": test_seg,
        }
        
        if self.model_config.is_ts:
            dataset = TSDatasetH(
                handler=handler,
                segments=segments,
                step_len=self.model_config.step_len,
            )
        else:
            dataset = DatasetH(
                handler=handler,
                segments=segments,
            )
        
        return dataset
    
    def _create_model(self):
        """Create model instance with GPU support"""
        model_kwargs = self.model_config.kwargs.copy()
        
        if "GPU" in model_kwargs:
            model_kwargs["GPU"] = self.gpu_id
        
        # In test mode, use minimal epochs
        if self.test_mode:
            if "n_epochs" in model_kwargs:
                model_kwargs["n_epochs"] = 5
            if "early_stop" in model_kwargs:
                model_kwargs["early_stop"] = 3
            if "num_boost_round" in model_kwargs:
                model_kwargs["num_boost_round"] = 20
        
        model_config = {
            "class": self.model_config.model_class,
            "module_path": self.model_config.module_path,
            "kwargs": model_kwargs,
        }
        model = init_instance_by_config(model_config)
        return model
    
    def _cross_sectional_normalize(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score normalization for predictions"""
        pred_normalized = pred_df.copy()
        
        def normalize_group(group):
            score = group["score"]
            mean = score.mean()
            std = score.std()
            if std > 0:
                group["score"] = (score - mean) / std
            else:
                group["score"] = 0.0
            return group
        
        pred_normalized = pred_normalized.groupby(level="datetime", group_keys=False).apply(normalize_group)
        return pred_normalized
    
    def _calculate_metrics(
        self,
        pred_df: pd.DataFrame,
        label_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate IC metrics"""
        pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index).dropna()
        
        if pred_label.empty:
            return {"IC": 0, "ICIR": 0, "Rank IC": 0, "Rank ICIR": 0}
        
        ic, ric = calc_ic(pred_label.iloc[:, 1], pred_label.iloc[:, 0])
        
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std() if ic.std() > 0 else 0,
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std() if ric.std() > 0 else 0,
        }
        return metrics
    
    def _run_backtest(
        self,
        pred_df: pd.DataFrame,
        test_start: str,
        test_end: str,
    ) -> Dict[str, Any]:
        """Run backtest and return metrics"""
        backtest_config = get_backtest_config(self.config.benchmark, self.strategy_type)
        
        backtest_config["strategy"]["kwargs"]["signal"] = pred_df
        backtest_config["backtest"]["start_time"] = test_start
        backtest_config["backtest"]["end_time"] = test_end
        
        try:
            portfolio_metric_dict, indicator_dict = normal_backtest(
                executor=backtest_config["executor"],
                strategy=backtest_config["strategy"],
                **backtest_config["backtest"]
            )
            
            report_normal, positions = portfolio_metric_dict.get("1day", (None, None))
            if report_normal is not None:
                excess_return_wo_cost = risk_analysis(
                    report_normal["return"] - report_normal["bench"],
                    freq="day"
                )
                excess_return_w_cost = risk_analysis(
                    report_normal["return"] - report_normal["bench"] - report_normal["cost"],
                    freq="day"
                )
                
                return {
                    "report": report_normal,
                    "positions": positions,
                    "excess_return_without_cost": excess_return_wo_cost,
                    "excess_return_with_cost": excess_return_w_cost,
                    "annualized_return_wo_cost": excess_return_wo_cost.loc["annualized_return", "risk"],
                    "sharpe_wo_cost": excess_return_wo_cost.loc["information_ratio", "risk"],
                    "annualized_return_w_cost": excess_return_w_cost.loc["annualized_return", "risk"],
                    "sharpe_w_cost": excess_return_w_cost.loc["information_ratio", "risk"],
                    "max_drawdown": excess_return_w_cost.loc["max_drawdown", "risk"],
                }
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
        
        return {}
    
    def _generate_historical_predictions(
        self,
        dataset,
        label_df: pd.DataFrame,
        fold_id: int,
    ) -> List[pd.DataFrame]:
        """Generate predictions from all historical models on current test data
        
        This is the KEY fix: we re-predict using historical models on the current test data,
        rather than using their saved predictions from their own test periods.
        """
        historical_preds_normalized = []
        
        if fold_id not in self.ensemble_ic_matrix:
            self.ensemble_ic_matrix[fold_id] = {}
        
        if not self.historical_models:
            return historical_preds_normalized
        
        logger.info(f"\n--- Evaluating {len(self.historical_models)} historical models on fold {fold_id} test data ---")
        
        for hist_fold_id, hist_model in enumerate(self.historical_models):
            try:
                # Generate predictions using historical model on CURRENT test data
                pred = hist_model.predict(dataset)
                if isinstance(pred, pd.Series):
                    pred_df = pred.to_frame("score")
                else:
                    pred_df = pred
                
                # Normalize
                pred_normalized = self._cross_sectional_normalize(pred_df)
                historical_preds_normalized.append(pred_normalized)
                
                # Calculate IC for ensemble matrix
                metrics = self._calculate_metrics(pred_df, label_df)
                self.ensemble_ic_matrix[fold_id][hist_fold_id] = metrics["Rank IC"]
                
                logger.info(f"  Model M{hist_fold_id} on T{fold_id}: RankIC={metrics['Rank IC']:.4f}")
            except Exception as e:
                logger.warning(f"  Failed to evaluate model {hist_fold_id} on fold {fold_id}: {e}")
                self.ensemble_ic_matrix[fold_id][hist_fold_id] = np.nan
        
        return historical_preds_normalized
    
    def _ensemble_predictions(
        self,
        current_pred: pd.DataFrame,
        historical_preds: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """Ensemble predictions: current 0.5, historical split 0.5"""
        if not historical_preds:
            return current_pred
        
        ensemble_pred = current_pred.copy()
        w_current = self.config.ensemble_weight_current
        w_historical = (1.0 - w_current) / len(historical_preds)
        
        ensemble_score = ensemble_pred["score"] * w_current
        
        for hist_pred in historical_preds:
            aligned_hist = hist_pred.reindex(ensemble_pred.index)
            aligned_hist = aligned_hist.fillna(0)
            ensemble_score = ensemble_score + aligned_hist["score"] * w_historical
        
        ensemble_pred["score"] = ensemble_score
        # Re-normalize ensemble predictions
        ensemble_pred = self._cross_sectional_normalize(ensemble_pred)
        
        return ensemble_pred
    
    def _save_predictions_parquet(self, fold_result: FoldResult):
        """Save predictions as parquet with date, symbol, {model}_pred columns"""
        fold_id = fold_result.fold_id
        pred_col_name = f"{self.model_config.name}_pred"
        
        pred_df = fold_result.pred_df.copy()
        pred_df = pred_df.reset_index()
        pred_df = pred_df.rename(columns={
            "datetime": "date",
            "instrument": "symbol",
            "score": pred_col_name,
        })
        
        pred_path = self.pred_dir / f"fold_{fold_id}_pred.parquet"
        pred_df.to_parquet(pred_path, index=False)
        
        pred_norm_df = fold_result.pred_normalized.copy()
        pred_norm_df = pred_norm_df.reset_index()
        pred_norm_df = pred_norm_df.rename(columns={
            "datetime": "date",
            "instrument": "symbol",
            "score": f"{pred_col_name}_normalized",
        })
        
        pred_norm_path = self.pred_dir / f"fold_{fold_id}_pred_normalized.parquet"
        pred_norm_df.to_parquet(pred_norm_path, index=False)
        
        logger.info(f"Saved predictions to {pred_path}")
    
    def _save_model_weights(self, fold_id: int, model):
        """Save model weights"""
        weight_path = self.weight_dir / f"fold_{fold_id}_model.pkl"
        with open(weight_path, "wb") as f:
            pickle.dump(model, f)
    
    def _generate_and_save_reports(
        self, 
        pred_df: pd.DataFrame,
        label_df: pd.DataFrame,
        backtest_metrics: Dict[str, Any],
        prefix: str,
    ):
        """Generate and save qlib-style reports as PDF"""
        if not backtest_metrics or "report" not in backtest_metrics:
            return
        
        report_df = backtest_metrics["report"]
        
        # Backtest report
        try:
            fig_list = analysis_position.report_graph(report_df, show_notebook=False)
            if fig_list:
                pdf_path = self.pdf_dir / f"{prefix}_backtest_report.pdf"
                pio.write_image(fig_list[0], str(pdf_path), format="pdf", width=1200, height=1400)
                logger.info(f"Saved backtest report to {pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to generate backtest report: {e}")
        
        # Model performance graphs
        try:
            pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index).dropna()
            pred_label.columns = ["label", "score"]
            
            if not pred_label.empty:
                fig_list = analysis_model.model_performance_graph(
                    pred_label, 
                    show_notebook=False,
                    graph_names=["group_return", "pred_ic"]
                )
                
                for i, fig in enumerate(fig_list):
                    pdf_path = self.pdf_dir / f"{prefix}_model_perf_{i}.pdf"
                    pio.write_image(fig, str(pdf_path), format="pdf", width=1200, height=800)
                
                logger.info(f"Saved {len(fig_list)} model performance graphs")
        except Exception as e:
            logger.warning(f"Failed to generate model performance graph: {e}")
    
    def train_single_fold(
        self,
        fold_id: int,
        train_start: str,
        train_end: str,
        valid_start: str,
        valid_end: str,
        test_start: str,
        test_end: str,
    ) -> FoldResult:
        """Train a single fold"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_id}: train [{train_start}, {train_end}], "
                   f"valid [{valid_start}, {valid_end}], test [{test_start}, {test_end}]")
        logger.info(f"{'='*60}")
        
        # Create dataset
        dataset = self._create_dataset(
            train_seg=(train_start, train_end),
            valid_seg=(valid_start, valid_end),
            test_seg=(test_start, test_end),
            fit_start=train_start,
            fit_end=train_end,
        )
        
        # Create and train model
        model = self._create_model()
        logger.info(f"Training {self.model_config.name} model (GPU={self.gpu_id})...")
        model.fit(dataset)
        
        # Generate predictions
        logger.info("Generating predictions...")
        pred = model.predict(dataset)
        if isinstance(pred, pd.Series):
            pred_df = pred.to_frame("score")
        else:
            pred_df = pred
        
        # Cross-sectional normalize
        pred_normalized = self._cross_sectional_normalize(pred_df)
        
        # Get labels for IC calculation
        label_df = dataset.prepare("test", col_set="label")
        label_df.columns = ["label"]
        
        # Calculate metrics for current model
        metrics = self._calculate_metrics(pred_df, label_df)
        logger.info(f"Fold {fold_id} Current Model Metrics:")
        logger.info(f"  IC={metrics['IC']:.4f}, ICIR={metrics['ICIR']:.4f}, "
                   f"RankIC={metrics['Rank IC']:.4f}, RankICIR={metrics['Rank ICIR']:.4f}")
        
        # Store current model's IC in ensemble matrix (diagonal element)
        if fold_id not in self.ensemble_ic_matrix:
            self.ensemble_ic_matrix[fold_id] = {}
        self.ensemble_ic_matrix[fold_id][fold_id] = metrics["Rank IC"]
        
        # Run backtest for current model
        logger.info("Running backtest for current model...")
        backtest_metrics = self._run_backtest(pred_df, test_start, test_end)
        
        if backtest_metrics:
            logger.info(f"Current Model Backtest: Ann.Return(w/cost)={backtest_metrics.get('annualized_return_w_cost', 0):.4f}, "
                       f"Sharpe(w/cost)={backtest_metrics.get('sharpe_w_cost', 0):.4f}, "
                       f"MaxDD={backtest_metrics.get('max_drawdown', 0):.4f}")
        
        # Generate reports for current model
        self._generate_and_save_reports(pred_df, label_df, backtest_metrics, f"fold_{fold_id}")
        
        # Save predictions as parquet first (before ensemble)
        fold_result = FoldResult(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            pred_df=pred_df,
            pred_normalized=pred_normalized,
            label_df=label_df,
            ic=metrics["IC"],
            icir=metrics["ICIR"],
            rank_ic=metrics["Rank IC"],
            rank_icir=metrics["Rank ICIR"],
            backtest_metrics=backtest_metrics,
        )
        
        self._save_predictions_parquet(fold_result)
        self._save_model_weights(fold_id, model)
        
        # Ensemble prediction (from fold 1 onwards)
        if fold_id > 0 and self.historical_models:
            logger.info(f"\n--- Ensemble with {len(self.historical_models)} historical models ---")
            
            # KEY FIX: Generate predictions from historical models on CURRENT test data
            historical_preds = self._generate_historical_predictions(dataset, label_df, fold_id)
            
            # Ensemble
            ensemble_pred = self._ensemble_predictions(pred_normalized, historical_preds)
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_metrics(ensemble_pred, label_df)
            logger.info(f"Ensemble Metrics: IC={ensemble_metrics['IC']:.4f}, "
                       f"ICIR={ensemble_metrics['ICIR']:.4f}, "
                       f"RankIC={ensemble_metrics['Rank IC']:.4f}")
            
            # Ensemble backtest
            ensemble_backtest = self._run_backtest(ensemble_pred, test_start, test_end)
            if ensemble_backtest:
                logger.info(f"Ensemble Backtest: Ann.Return(w/cost)={ensemble_backtest.get('annualized_return_w_cost', 0):.4f}, "
                           f"Sharpe(w/cost)={ensemble_backtest.get('sharpe_w_cost', 0):.4f}")
            
            fold_result.ensemble_metrics = ensemble_metrics
            fold_result.ensemble_backtest = ensemble_backtest
            
            # Save ensemble predictions
            ensemble_df = ensemble_pred.reset_index()
            ensemble_df = ensemble_df.rename(columns={
                "datetime": "date",
                "instrument": "symbol",
                "score": f"{self.model_config.name}_ensemble_pred",
            })
            ensemble_path = self.pred_dir / f"fold_{fold_id}_ensemble_pred.parquet"
            ensemble_df.to_parquet(ensemble_path, index=False)
        
        # Store model for future ensemble
        self.historical_models.append(model)
        self.fold_results.append(fold_result)
        
        return fold_result
    
    def _run_full_backtest(self):
        """Run backtest on concatenated predictions from all folds"""
        logger.info("\n" + "="*80)
        logger.info("FULL BACKTEST (All Folds Combined)")
        logger.info("="*80)
        
        if not self.fold_results:
            logger.warning("No fold results to combine")
            return
        
        # Concatenate all predictions
        all_preds = []
        all_labels = []
        for result in self.fold_results:
            all_preds.append(result.pred_df)
            all_labels.append(result.label_df)
        
        full_pred = pd.concat(all_preds, axis=0)
        full_pred = full_pred[~full_pred.index.duplicated(keep='first')]
        full_pred = full_pred.sort_index()
        
        full_label = pd.concat(all_labels, axis=0)
        full_label = full_label[~full_label.index.duplicated(keep='first')]
        full_label = full_label.sort_index()
        
        # Get date range
        first_test_start = self.fold_results[0].test_start
        last_test_end = self.fold_results[-1].test_end
        
        logger.info(f"Full backtest period: {first_test_start} to {last_test_end}")
        logger.info(f"Total predictions: {len(full_pred)}")
        
        # Calculate full metrics
        full_metrics = self._calculate_metrics(full_pred, full_label)
        logger.info(f"Full Period Metrics: IC={full_metrics['IC']:.4f}, ICIR={full_metrics['ICIR']:.4f}, "
                   f"RankIC={full_metrics['Rank IC']:.4f}")
        
        # Run backtest
        backtest_metrics = self._run_backtest(full_pred, first_test_start, last_test_end)
        
        if backtest_metrics:
            logger.info(f"Full Backtest Results:")
            logger.info(f"  Ann.Return(w/cost): {backtest_metrics.get('annualized_return_w_cost', 0):.4f}")
            logger.info(f"  Sharpe(w/cost): {backtest_metrics.get('sharpe_w_cost', 0):.4f}")
            logger.info(f"  MaxDD: {backtest_metrics.get('max_drawdown', 0):.4f}")
            
            # Generate and save full reports (including model perf graphs)
            self._generate_and_save_reports(full_pred, full_label, backtest_metrics, "full")
        
        # Save full predictions
        full_pred_df = full_pred.reset_index()
        full_pred_df = full_pred_df.rename(columns={
            "datetime": "date",
            "instrument": "symbol",
            "score": f"{self.model_config.name}_pred",
        })
        full_pred_path = self.pred_dir / "full_pred.parquet"
        full_pred_df.to_parquet(full_pred_path, index=False)
        logger.info(f"Saved full predictions to {full_pred_path}")
        
        return backtest_metrics
    
    def _print_ensemble_matrix(self):
        """Print diagonal matrix showing model performance across folds"""
        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE ANALYSIS MATRIX (RankIC)")
        logger.info("Rows: Test Fold, Columns: Model trained on Fold")
        logger.info("="*80)
        
        if not self.ensemble_ic_matrix:
            return
        
        n_folds = len(self.fold_results)
        matrix = np.full((n_folds, n_folds), np.nan)
        
        for test_fold, model_ics in self.ensemble_ic_matrix.items():
            for model_fold, ic in model_ics.items():
                if test_fold < n_folds and model_fold < n_folds:
                    matrix[test_fold, model_fold] = ic
        
        # Create DataFrame
        fold_names = [f"M{i}" for i in range(n_folds)]
        test_names = [f"T{i}" for i in range(n_folds)]
        
        matrix_df = pd.DataFrame(matrix, index=test_names, columns=fold_names)
        
        # Print matrix
        print("\n" + matrix_df.round(4).to_string())
        
        # Analysis
        logger.info("\nAnalysis:")
        
        # Diagonal elements (model tested on its own period)
        diagonal = np.diag(matrix)
        logger.info(f"  Diagonal Mean (self-test): {np.nanmean(diagonal):.4f}")
        
        # Off-diagonal elements (historical models on future data)
        off_diag = []
        for i in range(n_folds):
            for j in range(i):
                if not np.isnan(matrix[i, j]):
                    off_diag.append(matrix[i, j])
        
        if off_diag:
            logger.info(f"  Off-diagonal Mean (historical on future): {np.mean(off_diag):.4f}")
            
            # Check if ensemble is beneficial
            diag_mean = np.nanmean(diagonal[1:])  # exclude first fold
            off_diag_mean = np.mean(off_diag)
            
            if off_diag_mean > 0:
                logger.info(f"  Historical models still have predictive power (avg RankIC={off_diag_mean:.4f})")
                logger.info(f"  Ensemble is likely beneficial!")
            else:
                logger.info(f"  Historical models have weak/negative IC, ensemble may not help")
        
        # Save matrix
        matrix_path = self.pred_dir / "ensemble_matrix.csv"
        matrix_df.to_csv(matrix_path)
        logger.info(f"\nEnsemble matrix saved to {matrix_path}")
    
    def _print_summary(self):
        """Print summary of all folds"""
        logger.info("\n" + "="*80)
        logger.info("ROLLING TRAINING SUMMARY")
        logger.info("="*80)
        
        summary_data = []
        for result in self.fold_results:
            row = {
                "Fold": result.fold_id,
                "Test Period": f"{result.test_start} ~ {result.test_end}",
                "IC": result.ic,
                "ICIR": result.icir,
                "RankIC": result.rank_ic,
                "Ann.Ret": result.backtest_metrics.get("annualized_return_w_cost", 0),
                "Sharpe": result.backtest_metrics.get("sharpe_w_cost", 0),
                "MaxDD": result.backtest_metrics.get("max_drawdown", 0),
            }
            # Add ensemble metrics if available
            if result.ensemble_metrics:
                row["Ens.RankIC"] = result.ensemble_metrics.get("Rank IC", 0)
            if result.ensemble_backtest:
                row["Ens.Sharpe"] = result.ensemble_backtest.get("sharpe_w_cost", 0)
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        logger.info("\nAverage Metrics:")
        logger.info(f"  IC: {summary_df['IC'].mean():.4f}")
        logger.info(f"  ICIR: {summary_df['ICIR'].mean():.4f}")
        logger.info(f"  RankIC: {summary_df['RankIC'].mean():.4f}")
        logger.info(f"  Ann.Return: {summary_df['Ann.Ret'].mean():.4f}")
        logger.info(f"  Sharpe: {summary_df['Sharpe'].mean():.4f}")
        
        if "Ens.RankIC" in summary_df.columns:
            ens_rank_ic = summary_df["Ens.RankIC"].dropna()
            if len(ens_rank_ic) > 0:
                logger.info(f"  Ensemble RankIC (avg): {ens_rank_ic.mean():.4f}")
        
        summary_path = self.pred_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to {summary_path}")
    
    def run(self) -> List[FoldResult]:
        """Run all rolling folds"""
        self.all_periods = self._generate_rolling_periods()
        
        # In test mode, only run first 2 folds
        if self.test_mode and len(self.all_periods) > 2:
            self.all_periods = self.all_periods[:2]
            logger.info(f"TEST MODE: Limiting to {len(self.all_periods)} folds")
        
        logger.info(f"Total {len(self.all_periods)} rolling folds to train")
        
        for fold_id, (train_start, train_end, valid_start, valid_end, test_start, test_end) in enumerate(self.all_periods):
            self.train_single_fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
            )
        
        # Print fold summary
        self._print_summary()
        
        # Print ensemble matrix
        self._print_ensemble_matrix()
        
        # Run full backtest
        self._run_full_backtest()
        
        return self.fold_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rolling training for qlib models")
    parser.add_argument("--model", type=str, default="lightgbm", 
                       choices=list(ALL_MODELS.keys()),
                       help="Model to train")
    parser.add_argument("--market", type=str, default="csi500",
                       help="Market/universe to use")
    parser.add_argument("--data_start", type=str, default="2010-01-01",
                       help="Data start date")
    parser.add_argument("--data_end", type=str, default="2024-12-31",
                       help="Data end date")
    parser.add_argument("--train_years", type=int, default=3,
                       help="Training + validation years")
    parser.add_argument("--test_years", type=int, default=1,
                       help="Test years")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID to use for PyTorch models")
    parser.add_argument("--strategy", type=str, default="topk",
                       choices=["topk", "long_only"],
                       help="Backtest strategy type")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: minimal epochs and only 2 folds")
    
    args = parser.parse_args()
    
    # Setup config
    rolling_config = RollingConfig(
        data_start=args.data_start,
        data_end=args.data_end,
        train_years=args.train_years,
        test_years=args.test_years,
        market=args.market,
        benchmark="SH000905" if "500" in args.market else "SH000300",
    )
    
    model_config = ALL_MODELS[args.model]
    
    trainer = RollingTrainer(
        model_config, 
        rolling_config, 
        gpu_id=args.gpu,
        strategy_type=args.strategy,
        test_mode=args.test,
    )
    results = trainer.run()
    
    return results


if __name__ == "__main__":
    main()

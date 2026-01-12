import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from qlib.contrib.model.pytorch_gru_ts import GRU, GRUModel
from qlib.contrib.model.pytorch_lstm_ts import LSTM, LSTMModel
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset

class ReshapeModule(nn.Module):
    """Wraps a model to reshape flattened Alpha360 input (B, 360) to (B, 60, 6)"""
    def __init__(self, base_model, seq_len=60, d_feat=6):
        super().__init__()
        self.base_model = base_model
        self.seq_len = seq_len
        self.d_feat = d_feat
        
    def forward(self, x):
        # x: (Batch, 360) where 360 = 6 features * 60 steps
        # Data layout in Qlib Alpha360: [F1_T0..T59, F2_T0..T59, ..., F6_T0..T59]
        if x.dim() == 2:
            # 1. View as (Batch, Features, TimeSteps)
            x = x.view(-1, self.d_feat, self.seq_len)
            # 2. Permute to (Batch, TimeSteps, Features) for RNN
            x = x.permute(0, 2, 1)
        elif x.dim() == 3 and x.shape[1] == 1:
            # Handle case where TSDatasetH with step_len=1 is used
            x = x.view(-1, self.d_feat, self.seq_len)
            x = x.permute(0, 2, 1)
            
        return self.base_model(x)

class GRU360(GRU):
    """GRU model for Alpha360 (flattened) data"""
    def __init__(self, d_feat=6, **kwargs):
        # d_feat is 6 (internal feature dim)
        super().__init__(d_feat=d_feat, **kwargs)
        
        # Replace the internal model with the reshaping wrapper
        # Re-create GRUModel to ensure clean state
        gru_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.GRU_model = ReshapeModule(gru_model, seq_len=60, d_feat=6)
        self.GRU_model.to(self.device)
        
        # Update optimizer to track new parameters
        if self.optimizer == "adam":
            self.train_optimizer = torch.optim.Adam(self.GRU_model.parameters(), lr=self.lr)
        elif self.optimizer == "gd":
            self.train_optimizer = torch.optim.SGD(self.GRU_model.parameters(), lr=self.lr)

    def train_epoch(self, data_loader):
        print(f"DEBUG: Entering GRU360.train_epoch. Model type: {type(self)}")
        self.GRU_model.train()
        for data, weight in data_loader:
            # data: (Batch, Features+Label)
            # print(f"DEBUG: data shape: {data.shape}")
            feature = data[:, :-1].to(self.device)
            label = data[:, -1].to(self.device)
            
            pred = self.GRU_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))
            
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GRU_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        # print(f"DEBUG: Entering GRU360.test_epoch")
        self.GRU_model.eval()
        scores = []
        losses = []
        for data, weight in data_loader:
            feature = data[:, :-1].to(self.device)
            label = data[:, -1].to(self.device)
            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())
                score = self.metric_fn(pred, label)
                scores.append(score.item())
        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        print(f"DEBUG: Entering GRU360.fit. Model type: {type(self)}")
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.model.utils import ConcatDataset
        from qlib.utils import get_or_create_path
        import copy
        
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        # If it's DataFrame (DatasetH), convert to values. TSDataSampler (TSDatasetH) handles this internally.
        if isinstance(dl_train, pd.DataFrame):
            dl_train_values = dl_train.values
            dl_valid_values = dl_valid.values
        else:
            # Fallback if somehow we got something else, though config says DatasetH
            dl_train.config(fillna_type="ffill+bfill")
            dl_valid.config(fillna_type="ffill+bfill")
            dl_train_values = dl_train
            dl_valid_values = dl_valid

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        else:
            # Reweighter usually expects DataFrame
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)

        # Create datasets compatible with DataLoader
        # For DataFrame, values is (N, Features+Label). ConcatDataset zips data and weights.
        
        # Simple wrapper for numpy array to be a Dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __getitem__(self, index):
                return self.data[index]
            def __len__(self):
                return len(self.data)

        if isinstance(dl_train, pd.DataFrame):
            train_ds = SimpleDataset(dl_train_values)
            valid_ds = SimpleDataset(dl_valid_values)
        else:
            train_ds = dl_train_values
            valid_ds = dl_valid_values

        train_loader = DataLoader(
            ConcatDataset(train_ds, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(valid_ds, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = copy.deepcopy(self.GRU_model.state_dict())
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        
        if isinstance(dl_test, pd.DataFrame):
            test_values = dl_test.values
            # Fillna logic for DataFrame if needed, though usually handled by processors
            dl_test.fillna(0, inplace=True) # Basic fill if not done by processor
        else:
            dl_test.config(fillna_type="ffill+bfill")
            test_values = dl_test
            
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __getitem__(self, index):
                return self.data[index]
            def __len__(self):
                return len(self.data)

        if isinstance(dl_test, pd.DataFrame):
            test_ds = SimpleDataset(test_values)
        else:
            test_ds = test_values
            
        # Standard DataLoader for 2D data
        test_loader = DataLoader(ConcatDataset(test_ds), batch_size=self.batch_size, num_workers=self.n_jobs)
        
        self.GRU_model.eval()
        preds = []

        for data, in test_loader: # ConcatDataset with 1 element yields tuple (data,)
            feature = data[:, :-1].to(self.device)
            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.index if isinstance(dl_test, pd.DataFrame) else dl_test.get_index())

class LSTM360(LSTM):
    """LSTM model for Alpha360 (flattened) data"""
    def __init__(self, d_feat=6, **kwargs):
        super().__init__(d_feat=d_feat, **kwargs)
        
        lstm_model = LSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.LSTM_model = ReshapeModule(lstm_model, seq_len=60, d_feat=6)
        self.LSTM_model.to(self.device)
        
        if self.optimizer == "adam":
            self.train_optimizer = torch.optim.Adam(self.LSTM_model.parameters(), lr=self.lr)
        elif self.optimizer == "gd":
            self.train_optimizer = torch.optim.SGD(self.LSTM_model.parameters(), lr=self.lr)

    # Use GRU360's fit method (it's identical logic)
    fit = GRU360.fit

    def train_epoch(self, data_loader):
        self.LSTM_model.train()
        for data, weight in data_loader:
            feature = data[:, :-1].to(self.device)
            label = data[:, -1].to(self.device)
            
            pred = self.LSTM_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))
            
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.LSTM_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.LSTM_model.eval()
        scores = []
        losses = []
        for data, weight in data_loader:
            feature = data[:, :-1].to(self.device)
            label = data[:, -1].to(self.device)
            with torch.no_grad():
                pred = self.LSTM_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())
                score = self.metric_fn(pred, label)
                scores.append(score.item())
        return np.mean(losses), np.mean(scores)

    # Use GRU360's predict (identical except for model attribute, need to handle that)
    # Wait, GRU360.predict uses self.GRU_model. LSTM needs self.LSTM_model.
    # So we copy predict logic or make it generic.
    
    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        
        if isinstance(dl_test, pd.DataFrame):
            test_values = dl_test.values
            dl_test.fillna(0, inplace=True)
        else:
            dl_test.config(fillna_type="ffill+bfill")
            test_values = dl_test
            
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __getitem__(self, index):
                return self.data[index]
            def __len__(self):
                return len(self.data)

        if isinstance(dl_test, pd.DataFrame):
            test_ds = SimpleDataset(test_values)
        else:
            test_ds = test_values
            
        test_loader = DataLoader(ConcatDataset(test_ds), batch_size=self.batch_size, num_workers=self.n_jobs)
        
        self.LSTM_model.eval()
        preds = []

        for data, in test_loader:
            feature = data[:, :-1].to(self.device)
            with torch.no_grad():
                pred = self.LSTM_model(feature.float()).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.index if isinstance(dl_test, pd.DataFrame) else dl_test.get_index())


import os
import warnings
import joblib
import numpy as np
import pandas as pd
from schema.data_schema import ForecastingSchema
from momentfm import MOMENTPipeline
from sklearn.exceptions import NotFittedError
from logger import get_logger
from prediction.forecasting_dataset import ForecastingDataset
from sklearn.preprocessing import StandardScaler
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_FOLDER_NAME = "model_bin"

logger = get_logger(task_name="model")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def control_randomness(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Forecaster:
    """A wrapper class for the MOMENT Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "MOMENT TIMESERIES FOUNDATION MODEL - FORECASTING"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        use_static_covariates:bool = True,
        use_future_covariates: bool = True,
        max_windows: int = 10000,
        learning_rate: float = 1e-4,
        max_epoch: int = 3,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new MOMENT Forecaster

        Args:

            data_schema (ForecastingSchema):
                The schema of the data.

            use_static_covariates (bool):
                Whether the model should use static covariates if available.

            use_future_covariates (bool):
                Whether the model should use future covariates if available.

            max_windows (int): The maximum number of windows to use for training.

            learning_rate (float): The learning rate for finetuning the construction head.

            max_epoch (int): The max number of epochs to use for training.

            **kwargs:
                Optional arguments.
        """
        self.data_schema = data_schema
        self.max_windows = max_windows
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.random_state = random_state

        # not supported for MOMENT and future and static covatiates are considered as just features
        use_past_covariates = False,

        self.dataset = ForecastingDataset(
            forecast_horizon=self.data_schema.forecast_length,
            random_seed=self.random_state,
        )

        self.use_static_covariates = use_static_covariates and (
            len(data_schema.static_covariates) > 0
        )
        self.use_past_covariates = (
            use_past_covariates and len(data_schema.past_covariates) > 0
        )
        self.use_future_covariates = use_future_covariates and (
            len(data_schema.future_covariates) > 0
            or self.data_schema.time_col_dtype in ["DATE", "DATETIME"]
        )

        self.kwargs = kwargs
        self._is_trained = False
        control_randomness(self.random_state)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self.scaler = {}
        cols_to_drop = []

        cols_to_drop += [self.data_schema.time_col]

        if not self.use_past_covariates and set(
            self.data_schema.past_covariates
        ).issubset(data.columns):
            cols_to_drop += self.data_schema.past_covariates

        if not self.use_future_covariates and set(
            self.data_schema.future_covariates
        ).issubset(data.columns):
            cols_to_drop += self.data_schema.future_covariates

        if not self.use_static_covariates and set(
            self.data_schema.static_covariates
        ).issubset(data.columns):
            cols_to_drop += self.data_schema.static_covariates

        data_features = data.drop(columns=cols_to_drop)

        grouped = data_features.groupby(self.data_schema.id_col)
        target_index = data_features.columns.get_loc(self.data_schema.target)
        self.dataset.set_target_index(target_index - 1)

        for id, df in grouped:
            df_features = df.drop(columns=[self.data_schema.id_col])
            self.scaler[str(id)] = StandardScaler()
            self.scaler[str(id)].fit(df_features.values)
            df_features_scaled = self.scaler[str(id)].transform(df_features.values)
            self.dataset.extend_to_windows(series_id=str(id), data=df_features_scaled)

        indicies = [i for i in range(len(self.dataset.timeseries))]
        shuffled = random.sample(indicies, len(indicies))[: self.max_windows]

        self.dataset.timeseries = list(np.array(self.dataset.timeseries)[shuffled])
        self.dataset.forecast = list(np.array(self.dataset.forecast)[shuffled])
        self.dataset.input_mask = list(np.array(self.dataset.input_mask)[shuffled])

    def _finetune_head(self):
        # Load data
        train_loader = DataLoader(self.dataset, batch_size=8, shuffle=True)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        min_iter = 50
        cur_epoch = 0
        max_epoch = min_iter // len(train_loader) if len(train_loader) < min_iter else self.max_epoch

        # Move the model to the GPU
        self.model = self.model.to(device)

        # Move the loss function to the GPU
        criterion = criterion.to(device)

        # Enable mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # Create a OneCycleLR scheduler
        max_lr = 1e-4
        total_steps = len(train_loader) * max_epoch
        scheduler = OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3
        )

        # Gradient clipping value
        max_norm = 5.0
        self.model.train()
        while cur_epoch < max_epoch:
            losses = []
            for timeseries, forecast, input_mask in tqdm(
                train_loader, total=len(train_loader)
            ):
                # Move the data to the GPU
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(torch.float32).to(device)
                forecast = forecast.float().to(device)

                with torch.cuda.amp.autocast():
                    output = self.model(timeseries, input_mask)

                loss = criterion(output.forecast, forecast)

                # Scales the loss for mixed precision training
                scaler.scale(loss).backward()

                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                losses.append(loss.item())

            losses = np.array(losses)
            average_loss = np.average(losses)
            print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

            # Step the learning rate scheduler
            scheduler.step()
            cur_epoch += 1

            self.model.eval()

    def fit(
        self,
        train_data: pd.DataFrame,
    ) -> None:
        """Fit the Forecaster model.
            MOMENT is a family of open-source foundation models for general-purpose time series analysis.

        Args:
            train_data (pd.DataFrame): Training data.

        """
        self._prepare_data(train_data)
        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": self.data_schema.forecast_length,
                "head_dropout": 0.1,
                "weight_decay": 0,
                "freeze_encoder": True,  # Freeze the patch embedding layer
                "freeze_embedder": True,  # Freeze the transformer encoder
                "freeze_head": False,  # The linear forecasting head must be trained
            },
        )
        self.model.init()
        self.model = self.model.to(device)
        self._finetune_head()
        self.dataset._clear_train()
        self._is_trained = True

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.
        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The predictions dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        res = []
        for id, df in test_data.groupby(self.data_schema.id_col, sort=False):
            test = torch.from_numpy(
                self.dataset.test[str(id)][None, :, :].astype(np.float32)
            ).to(device)
            self.model = self.model.to(device)

            pred = self.model(test)
            pred_reverse_scaled = self.scaler[str(id)].inverse_transform(
                np.squeeze(pred.forecast.detach().cpu().numpy(), axis=0)
            )
            target_res = pred_reverse_scaled[self.dataset.target_index]
            res += list(target_res)
        test_data[prediction_col_name] = res
        return test_data

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        # self.model.save_pretrained(os.path.join(model_dir_path, MODEL_FOLDER_NAME))
        # joblib.dump(self.model.config, os.path.join(model_dir_path, MODEL_FOLDER_NAME, 'model_config.pkl'))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        forecaster = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

        # model_config = joblib.load(os.path.join(model_dir_path, MODEL_FOLDER_NAME, 'model_config.pkl'))
        # forecaster.model = MOMENTPipeline.from_pretrained(os.path.join(model_dir_path, MODEL_FOLDER_NAME), config = model_config)
        return forecaster

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    data_schema: ForecastingSchema,
    train_data: pd.DataFrame,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting used to do prediction.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)

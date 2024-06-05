from torch.utils.data import Dataset
import numpy as np

class ForecastingDataset (Dataset):

    def __init__(self, 
                 forecast_horizon: int = 1,
                 data_stride_len: int = 1,
                 task_name: str = "forecasting",
                 random_seed: int = 42
                 ):

        self.seq_len = 512 #fixed for MOMENT model
        self.forecast_horizon = forecast_horizon
        self.data_stride_len = data_stride_len
        self.timeseries = []
        self.forecast = []
        self.input_mask = []
        self.test = {}
        self.target_index = 0


    def __getitem__(self, index):
        return self.timeseries[index] , self.forecast[index] , self.input_mask[index]

    def __len__(self):
        return len(self.timeseries)

    def extend_to_windows(self, series_id:str, data:np.ndarray):
        num_points = data.shape[0]

        if num_points < self.forecast_horizon:
            # If data is less than forecast_horizon, we cannot create a valid window
            return
        
        if num_points < self.seq_len + self.forecast_horizon:
            # Calculate padding length needed
            padding_length = self.seq_len + self.forecast_horizon - num_points
            # Pad data with zeros on the left
            padding = np.zeros((padding_length, data.shape[1]))
            padded_data = np.vstack((padding, data))
            
            # Create a single window
            timeseries_window = padded_data[:-self.forecast_horizon].T
            forecast_window = padded_data[-self.forecast_horizon:].T
            input_mask_window = np.zeros(self.seq_len + self.forecast_horizon)
            input_mask_window[padding_length:padding_length + num_points - self.forecast_horizon] = 1
            input_mask_window[-self.forecast_horizon:] = 1
            
            self.timeseries.append(timeseries_window)
            self.forecast.append(forecast_window)
            self.input_mask.append(input_mask_window[:-self.forecast_horizon])  # The input mask should match the timeseries length
            
            self.test[series_id] = timeseries_window
            return
        
        for start in range(0, num_points - self.seq_len - self.forecast_horizon + 1, self.data_stride_len):
            end = start + self.seq_len
            forecast_end = end + self.forecast_horizon
            
            # Create windows
            timeseries_window = data[start:end, :].T
            forecast_window = data[end:forecast_end, :].T
            input_mask_window = np.ones(self.seq_len)
            
            # Append to lists
            self.timeseries.append(timeseries_window)
            self.forecast.append(forecast_window)
            self.input_mask.append(input_mask_window)

        self.test[series_id] = self.timeseries[-1]

    def _clear_train(self):
        self.timeseries = None
        self.forecast = None
        self.input_mask = None

    def set_target_index(self, i):
        self.target_index = i


if __name__ == "__main__":
    # Simulated data example
    data = np.random.rand(25, 1)  # 10,000 time steps with 10 features each

    # Initialize dataset
    dataset = ForecastingDataset(forecast_horizon=24, data_stride_len=1)
    
    # Extend to windows
    dataset.extend_to_windows(data)
    
    # Check the lengths
    print(f"Number of windows: {len(dataset)}")

    # Access the first window
    timeseries, forecast, input_mask = dataset[0]
    print(timeseries)
    print(f"Timeseries shape: {timeseries.shape}")
    print(f"Forecast shape: {forecast.shape}")
    print(f"Input mask shape: {input_mask.shape}")



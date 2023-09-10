import numpy as np
import torch
import torch.nn as nn


class HoltWinters(nn.Module):
    
    def __init__(self, init_a=0.1, init_b=0.1, init_g=0.1, slen=12):
        super(HoltWinters, self).__init__()
        
        # Smoothing parameters
        self.alpha = nn.Parameter(torch.tensor(init_a))
        self.beta = nn.Parameter(torch.tensor(init_b))  # Trend smoothing parameter
        self.gamma = nn.Parameter(torch.tensor(init_g))
        
        # Initial parameters
        self.init_season = nn.Parameter(torch.tensor(np.random.random(size=slen)))
        self.init_trend = nn.Parameter(torch.tensor(0.1))        

        # Season length used to pick appropriate past season step
        self.slen = slen
        
        # Sigmoid used to normalize the parameters to be between 0 and 1 if needed
        self.sig = nn.Sigmoid()
        
    def forward(self, series, series_shifts, n_preds=8, rv=False):
        batch_size = series.shape[0]
        init_season_batch = self.init_season.repeat(batch_size).view(batch_size, -1)
        
        # Use roll to allow for our random input shifts
        seasonals = torch.stack([torch.roll(j, int(rol)) for j, rol in zip(init_season_batch, series_shifts)]).float()
        
        # Convert to a list to avoid inplace tensor changes
        seasonals = [x.squeeze() for x in torch.split(seasonals, 1, dim=1)]
        
        result = []
        trend = self.init_trend.repeat(batch_size)  # Initialize trend for each batch
        
        if rv:
            value_list = []
            season_list = []
            trend_list = []
        
        for i in range(series.shape[1] + n_preds):
            if i == 0:
                smooth = series[:, 0]
                result.append(smooth + trend)
                if rv:
                    value_list.append(smooth)
                    season_list.append(seasonals[i % self.slen])
                    trend_list.append(trend)
            else:
                smooth_prev = smooth
                trend_prev = trend
                season_prev = seasonals[i % self.slen]
                
                # Update equations
                smooth = self.alpha * (series[:, i] - season_prev) + (1 - self.alpha) * (smooth_prev + trend_prev)
                trend = self.beta * (smooth - smooth_prev) + (1 - self.beta) * trend_prev
                seasonals.append(self.gamma * (series[:, i] - smooth) + (1 - self.gamma) * season_prev)
                
                result.append(smooth + trend + seasonals[i % self.slen])
                
                if rv:
                    value_list.append(smooth)
                    season_list.append(seasonals[i % self.slen])
                    trend_list.append(trend)
        
        if rv:
            return torch.stack(result, dim=1), torch.stack(value_list, dim=1), torch.stack(season_list, dim=1), torch.stack(trend_list, dim=1)
        else:
            return torch.stack(result, dim=1)[:, -n_preds:]


class ESRNN(nn.Module):
    
    def __init__(self, hidden_size=16, slen=12, pred_len=12, use_trend=True):
        super(ESRNN, self).__init__()
        
        self.hw = HoltWinters(init_a=0.1, init_b=0.1, init_g=0.1)
        self.rnn = nn.GRU(hidden_size=hidden_size, input_size=1, batch_first=True)
        self.lin = nn.Linear(hidden_size, pred_len)
        self.pred_len = pred_len
        self.slen = slen
        self.use_trend = use_trend  # Whether to use the trend component or not
        
    def forward(self, series, shifts):
        batch_size = series.shape[0]
        result, smoothed_value, smoothed_season, smoothed_trend = self.hw(series, shifts, rv=True, n_preds=0)
        
        # De-seasonalize and de-level considering the trend if use_trend is True
        de_season = series - smoothed_season
        if self.use_trend:
            de_level = de_season - smoothed_value - smoothed_trend
        else:
            de_level = de_season - smoothed_value
        
        noise = torch.randn(de_level.shape[0], de_level.shape[1])
        noisy = de_level  # +noise
        noisy = noisy.unsqueeze(2)
        
        feature = self.rnn(noisy)[1].squeeze()
        pred = self.lin(feature)
        
        season_forecast = [smoothed_season[:, i % self.slen] for i in range(self.pred_len)]
        season_forecast = torch.stack(season_forecast, dim=1)
        
        # Add back the trend for forecasting if use_trend is True
        if self.use_trend:
            trend_forecast = [smoothed_trend[:, i] for i in range(self.pred_len)]
            trend_forecast = torch.stack(trend_forecast, dim=1)
            return smoothed_value[:, -1].unsqueeze(1) + season_forecast + trend_forecast + pred
        else:
            return smoothed_value[:, -1].unsqueeze(1) + season_forecast + pred

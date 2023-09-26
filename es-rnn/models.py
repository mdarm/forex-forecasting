import numpy as np
import torch
import torch.nn as nn


class HoltsWintersNoTrend(nn.Module):
    def __init__(self, init_a=0.1, init_g=0.1, slen=5, mode='multiplicative'):
        super(HoltsWintersNoTrend, self).__init__()
        
        # Holt-Winters trainable parameters
        self.alpha = nn.Parameter(torch.tensor(init_a))
        self.gamma = nn.Parameter(torch.tensor(init_g))

        self.init_season = nn.Parameter(torch.tensor(np.random.random(size=slen)))
        self.mode = mode
        self.slen = slen
        
    def forward(self, series, series_shifts, n_preds=14, return_coefficients=False):
        batch_size = series.shape[0]
        init_season_batch = self.init_season.repeat(batch_size).view(batch_size, -1)
        
        # Use roll to allow for our random input shifts
        seasonals = torch.stack([torch.roll(j, int(rol)) for j, rol in zip(init_season_batch, series_shifts)]).float()
        
        # Convert to a list to avoid inplace tensor changes
        seasonals = [x.squeeze() for x in torch.split(seasonals, 1, dim=1)]
        
        result = []
        
        if return_coefficients:
            value_list = []
            season_list = []
        for i in range(series.shape[1] + n_preds):
            if i == 0:
                smooth = series[:, 0]
                result.append(smooth)
                if return_coefficients:
                    value_list.append(smooth)
                    season_list.append(seasonals[i % self.slen])
                    continue
            if i < series.shape[1]:
                smooth_prev = smooth
                season_prev = seasonals[i % self.slen]
                
                # Calculate level and seasonality for current timestep to deseason and delevel
                # the data for the RNN
                if self.mode == 'additive':
                    smooth = self.alpha * (series[:, i] - season_prev) + (1 - self.alpha) * smooth_prev
                    seasonals.append(self.gamma * (series[:, i] - smooth) + (1 - self.gamma) * season_prev)
                else:
                    smooth = self.alpha * (series[:, i] / season_prev) + (1 - self.alpha) * smooth_prev
                    seasonals.append(self.gamma * (series[:, i] / smooth) + (1 - self.gamma) * season_prev)
                              
                if return_coefficients:
                    value_list.append(smooth)
                    season_list.append(seasonals[i % self.slen])
        
                # Calculate smoothed series 
                if self.mode == 'additive':
                    result.append(smooth + seasonals[i % self.slen])
                else:
                    result.append(smooth * seasonals[i % self.slen])

        if return_coefficients:
            return torch.stack(result, dim=1), torch.stack(value_list, dim=1), torch.stack(season_list, dim=1)
        else:
            return torch.stack(result, dim=1)[:, -n_preds:]


class ESRNN(nn.Module):
    def __init__(self, hidden_size=16, slen=14, pred_len=14, mode='multiplicative'):
        super(ESRNN, self).__init__()
        
        self.hw = HoltsWintersNoTrend(init_a=0.1, init_g=0.1, mode=mode)
        self.rnn = nn.GRU(hidden_size=hidden_size, input_size=1, batch_first=True)
        self.lin = nn.Linear(hidden_size, pred_len)
        self.pred_len = pred_len
        self.slen = slen
        
    def forward(self, series, shifts):
        _, smoothed_level, smoothed_season = self.hw(series, shifts,
                                                     return_coefficients=True, n_preds=0)
        
        if self.hw.mode == 'additive':
            de_season = series - smoothed_season
        else:
            de_season = series / smoothed_season

        de_level = de_season / smoothed_level
        de_level = de_level.unsqueeze(2)
        
        feature = self.rnn(de_level)[1].squeeze()
        pred = self.lin(feature)
        
        season_forecast = [smoothed_season[:, i % self.slen] for i in range(self.pred_len)]
        season_forecast = torch.stack(season_forecast, dim=1)

        # Re-season and re-level the RNN's output
        if self.hw.mode == 'additive':
            return smoothed_level[:, -1].unsqueeze(1) + season_forecast + pred
        else:
            return smoothed_level[:, -1].unsqueeze(1) * season_forecast * pred

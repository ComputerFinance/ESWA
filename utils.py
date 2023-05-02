import numpy as np
import statsmodels.api as sm


class Indicators:
    def __init__(self, annual_basis=252, risk_free=0.03):
        self.annual_basis = annual_basis
        self.risk_free = risk_free

    def calculate_log_returns(self, percentage_change):
        return percentage_change.apply(np.log1p).iloc[1:, :]

    def calculate_cumulative_returns(self, percentage_change):
        return (1 + percentage_change).cumprod() - 1

    def calculate_expected_annual_return(self, percentage_change):
        return np.sum(percentage_change.mean() * self.annual_basis)

    def calculate_annual_variance(self, percentage_change):
        return self.annual_basis * np.var(percentage_change)

    def calculate_annual_volatility(self, percentage_change):
        return np.sqrt(self.annual_basis) * np.std(percentage_change)

    def calculate_max_drawdown(self, cummulative_returns):
        cummulative_invest = 1 + cummulative_returns
        up = cummulative_invest.cummax()
        dd = (cummulative_invest / up) - 1
        return np.sum(dd.min())

    def calculate_sharpe_ratio(self, percentage_change):
        excess_returns = percentage_change - self.risk_free / self.annual_basis
        return np.sqrt(self.annual_basis) * excess_returns.mean() / excess_returns.std()

    def calculate_beta(self, portfolio_returns, bench_returns):

        Y = portfolio_returns.values
        X = bench_returns.values
        Z = sm.add_constant(X)
        model = sm.OLS(Y, Z).fit()
        return model.params[1]

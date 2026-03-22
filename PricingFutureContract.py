import numpy as np

class ForwardFutureContract:

    def __init__(self, spot_price, risk_free_rate, maturity,
                 position="long", delivery_price=None):
        self.S = spot_price
        self.r = risk_free_rate
        self.T = maturity
        self.position = position.lower()
        self.K = delivery_price

    # forward price = future price (assuming no dividends and no storage)
    def forward_price(self):
        return self.S * np.exp(self.r * self.T)

    def contract_value(self):
        if self.K is None:
            raise ValueError('Delivery price K must be specified')

        # value of contract at time t
        # V(t) = S(t) - K exp{-r(T-t)}
        # for now, take t=0
        V = self.S - self.K * np.exp(-self.r * self.T)

        if self.position == "short":
            V = -V

        return V

contract = ForwardFutureContract(
    spot_price=100,
    risk_free_rate=0.05,
    maturity=1,
    position="long",
    delivery_price=105
)

contract.K = contract.forward_price()

print(f"Futures price: ${contract.forward_price():.2f}")
print(f"Contract value: ${contract.contract_value():.2f}")
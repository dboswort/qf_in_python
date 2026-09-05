"""
The class defined in this module allows for the calculation of the forward price and the value of the 
contract based on the spot price, risk-free rate, time to expiry, position (long or short), and delivery (strike) price.
"""
import numpy as np

class ForwardFuturesContract:

    def __init__(self, spot_price, risk_free_rate, expiry,
                 position="long", delivery_price=None):
        self.S = spot_price
        self.r = risk_free_rate
        self.T = expiry
        self.position = position.lower()
        self.K = delivery_price

    # forward price = continuously compounded future price (assuming no dividends/storage costs)
    def forward_price(self):
        return self.S * np.exp(self.r * self.T)

    def contract_value(self):
        if self.K is None:
            raise ValueError('Delivery price K must be specified')

        # assume t=0 for convenience, so V(0) = S - K exp{-rT}
        V = self.S - self.K * np.exp(-self.r * self.T)

        if self.position == "short":
            V = -V

        return V

if __name__ == "__main__":
    contract = ForwardFuturesContract(
        spot_price=100,
        risk_free_rate=0.05,
        expiry=1,
        position="long",
        delivery_price=105
    )

    contract.K = contract.forward_price()

    print(f"Futures price: ${contract.forward_price():.2f}")
    print(f"Contract value: ${contract.contract_value():.2f}")
import math


class ZeroCouponBond:
    # constructor
    def __init__(self, principal, maturity, interest_rate):
        self.principal = principal                # face value
        self.maturity = maturity                  # time-to-maturity
        self.interest_rate = interest_rate / 100  # market interest rate

    def present_value_discrete(self, cf, n):
        return cf / (1 + self.interest_rate) ** n

    def calculate_price_discrete(self):
        return self.present_value_discrete(self.principal, self.maturity)

    def present_value_continuous(self, cf, t):
        return cf * math.exp(- self.interest_rate * t)

    def calculate_price_continuous(self):
        return self.present_value_continuous(self.principal, self.maturity)


class CouponBond:
    # constructor
    def __init__(self, principal, bond_rate, maturity, market_rate):
        self.principal = principal              # face value
        self.bond_rate = bond_rate / 100        # interest rate of bond
        self.maturity = maturity                # time-to-maturity
        self.market_rate = market_rate / 100    # market interest rate

    def present_value_discrete(self, cf, n):
        return cf / (1 + self.market_rate) ** n

    def calculate_price_discrete(self):
        # discount the coupon payments
        price = 0
        for t in range(1, self.maturity + 1):
            cf = self.principal * self.bond_rate # cashflow
            price += self.present_value_discrete(cf, t)   # present value of cash flow

        # discount principal amount
        price += self.present_value_discrete(self.principal, self.maturity)

        return price

    def present_value_continuous(self, cf, t):
        return cf * math.exp(- self.market_rate * t)

    def calculate_price_continuous(self):
        # discount the coupon payments
        price = 0
        for t in range(1, self.maturity + 1):
            cf = self.principal * self.bond_rate # cashflow
            price += self.present_value_continuous(cf, t)   # present value of cash flow

        # discount principal amount
        price += self.present_value_continuous(self.principal, self.maturity)

        return price


if __name__ == '__main__':
    zero_bond = ZeroCouponBond(principal=1000, maturity=2, interest_rate=4)
    print(f"ZB present value: ${zero_bond.calculate_price_continuous():.2f}")
    print(f"ZB cash flow present value: ${zero_bond.present_value_continuous(10, 1):.2f}")

    coupon_bond = CouponBond(principal=1000,
                             bond_rate=10,
                             maturity=3,
                             market_rate=4)
    print("CB present value: $%.2f" % coupon_bond.calculate_price_continuous())
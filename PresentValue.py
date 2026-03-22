import math

def future_value_discrete(pv: float, r: float, n: int) -> float:
    return pv * (1 + r) ** n


def future_value_continuous(pv: float, r: float, t: float) -> float:
    return pv * math.exp(r * t)


def present_value_discrete(fv: float, r: float, n: int) -> float:
    return fv / (1 + r)**n


def present_value_continuous(fv: float, r: float, t: float) -> float:
    return fv * math.exp(-r * t)


present_value = 1000    # dollars
r = 0.08                # risk-free rate
n = 2                  # years

result = future_value_discrete(present_value, r, n)
print(f'Future Value (annual compounding): ${result:.2f}')   # format string to 2 d.p.

result = future_value_continuous(present_value, r, n)
print(f'Future Value (continuous compounding): ${result:.2f}')   # format string to 2 d.p.

result = present_value_discrete(20000, 0.06, 13)
print(f'Present Value (annual compounding): ${result:.2f}')

result = present_value_continuous(20000, 0.06, 13)
print(f'Present Value (continuous compounding): ${result:.2f}')
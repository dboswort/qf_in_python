"""
PresentValue.py

Determines future value and present value using discrete and continuous compounding.
"""
import math


def future_value_discrete(pv: float, r: float, n: int) -> float:
    """
    Returns the future value assuming discrete, annual compounding.
    
    Arguments:
    pv - present value
    r - risk-free rate, e.g. 0.01 for 1.0%
    n - Accrual period in years
    """
    return pv * (1 + r) ** n


def future_value_continuous(pv: float, r: float, t: float) -> float:
    """
    Returns the future value assuming continuous compounding.
    
    Arguments:
    pv - present value
    r - risk-free rate, e.g. 0.01 for 1.0%
    n - Accrual period in years
    """
    return pv * math.exp(r * t)


def present_value_discrete(fv: float, r: float, n: int) -> float:
    """
    Returns the present value assuming discrete, annual compounding.
    
    Arguments:
    fv - future value
    r - risk-free rate, e.g. 0.01 for 1.0%
    n - Accrual period in years
    """
    return fv / (1 + r)**n


def present_value_continuous(fv: float, r: float, t: float) -> float:
    """
    Returns the present value assuming continuous compounding.
    
    Arguments:
    fv - future value
    r - risk-free rate, e.g. 0.01 for 1.0%
    n - Accrual period in years
    """
    return fv * math.exp(-r * t)


present_value = 1000    # PV in dollars
r = 0.035               # risk-free rate
n = 2                   # years

fv_d = future_value_discrete(present_value, r, n)
print(f'Future value (annual compounding): ${fv_d:.2f}')  
fv_c = future_value_continuous(present_value, r, n)
print(f'Future value (continuous compounding): ${fv_c:.2f}')
print(f'Absolute difference: ${abs(fv_d-fv_c):.2f}')

pv_d = present_value_discrete(20000, 0.06, 13)
print(f'Present value (annual compounding): ${pv_d:.2f}')
pv_c = present_value_continuous(20000, 0.06, 13)
print(f'Present value (continuous compounding): ${pv_c:.2f}')
print(f'Absolute difference: ${abs(pv_d-pv_c):.2f}')

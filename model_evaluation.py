"""premium_interface.py

Module header for premium calculation and customer selection system.

This header file defines the function interfaces used for determining which 
customers to insure and setting their insurance premiums. The implementation 
is located in the corresponding module (e.g., `set_premium.py`).

Modules required:
    - pandas
    - numpy
    - joblib
    - model_dev (contains `customer_claim_probability`)
    - customer (contains the `Customer` class)
    - customer_selection (contains `knapsack_selection`)
"""

import pandas as pd
import numpy as np
import joblib
from model_dev import customer_claim_probability
from customer import Customer
from customer_selection import greedy_selection

def set_premium(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a dataframe of data on potential customers, identifies which customers to take on, and sets premiums for them.
    This function may load a saved model (previously trained using supervised learning) to save time.
    :param customer_data: a dataframe with the same columns as customer_data.csv
    :return: a DataFrame of the same lenght as customer_data, with the following columns:
            p_claim:  your estimated probability that each customer in customer_data will make a claim
            desired_customer:  a boolean column indicating which customers you want to take on
            premium:    the amount of the premium to charge each customer
    """

    model = joblib.load('final_insurance_model_truefinal.pkl') 
    features = joblib.load('best_model_features_truefinal.pkl')

    customer_data_subset = customer_data[features]

    p_claims = customer_claim_probability(model, customer_data_subset)

    customers = []
    for i in range(len(customer_data)):
        coverage_amt = customer_data['coverage_amount'].loc[i]
        customer = Customer(customer_id=i, coverage_amt=coverage_amt, p_claim=p_claims[i])
        customers.append(customer)

    # get a list of all premiums
    premiums = [cust.premium for cust in customers]

    # filter out customers with the highest premiums as it would be too competitive (top 5%)
    max_premium = np.percentile(premiums, 95)
    eligible_customers = [cust for cust in customers if cust.premium <= max_premium]

    # get a list of all profits from the filtered list
    profits = [cust.profit for cust in eligible_customers]

    # filter out the lowest profits (bottom 35%)
    min_profit = np.percentile(profits, 35)
    eligible_customers = [cust for cust in eligible_customers if min_profit <= cust.profit]
    
    solution = greedy_selection(eligible_customers)

    for ids in solution:
        customers[ids].desired_customer = True

    total_spent = 0
    total_customers = 0
    for cust in customers:
        if cust.desired_customer:
            total_spent += cust.coverage_amt
            total_customers += 1

    print(f"Total spent: {total_spent:,}")
    print(f"Total customers: {total_customers:,}")
    print(f"Total expected profit: ${sum(cust.profit for cust in customers if cust.desired_customer):,.2f}")

    return pd.DataFrame({
        'p_claim': [cust.p_claim for cust in customers],
        'desired_customer': [cust.desired_customer for cust in customers],
        'premium': [cust.premium for cust in customers]
    })

if __name__ == "__main__":
    customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
    premium_table = set_premium(customer_data)
    print(premium_table)
    #np.savetxt('customerinfo.txt', premium_table.values, fmt=['%10.2f', '%18s', '%12.2f'], header='  p_claim   desired_customer     premium', comments='')
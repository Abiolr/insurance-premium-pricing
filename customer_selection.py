# customer_selection.py
import numpy as np

def greedy_selection(eligible_customers, max_coverage=100_000_000):
    """
    Selects customers using a greedy knapsack approach.
    It prioritizes customers with the lowest coverage_amt first.

    :param eligible_customers: A list of Customer objects *already filtered*
                               to be profitable and acceptable.
    :param max_coverage: The $100M cap.
    :return: A 'set' of customer_ids for the selected customers.
    """

    sorted_customers = sorted(eligible_customers, key=lambda cust: cust.coverage_amt)

    selected_customer_ids = set()
    current_coverage = 0

    for customer in sorted_customers:
        if current_coverage + customer.coverage_amt <= max_coverage:
            current_coverage += customer.coverage_amt
            selected_customer_ids.add(customer.customer_id)
        else:
            break
    
    return selected_customer_ids
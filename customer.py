"""customer.py

This module defines the Customer class, which is used to model an insurance customer,
which is used to handle computing, their perceived utility, expected cost, indifferent price,
premium, as well as the profit.

"""

import numpy as np

class Customer:
    """Represents an insurance customer with utility-based premium calculation.

    This class models a customer's risk and behaviour based on the probability
    of making a claim (P_claim), the premium offer, and their precieved utility.
    
    Attributes:
    customer_id (int): Unique identifier for the customer.
    coverage_amt (int): The insurance coverage amount.
    p_claim (float): Probability that the customer makes a claim.
    perceived_utility (float): Expected utility considering the risk of claim.
    expected_cost (float): The insurer’s expected payout for the customer.
    indifferent_amount (float): The monetary loss that yields the same utility
        as the expected utility, used for pricing decisions.
    premium (float): The premium the insurer would charge the customer.
    profit (float): The difference between premium and expected cost.
    desired_customer (bool): Flag to indicate if the customer meets the
        insurer’s selection criteria.
    """
    def __init__(self, customer_id: int, coverage_amt: int, p_claim: float):
        """Initialize a Customer.

        Args:
            customer_id (int): The customer’s unique ID.
            coverage_amt (int): Coverage ammount from table
            p_claim (float): Probability of making a claim.
        """
        self.customer_id = customer_id
        self.coverage_amt = coverage_amt
        self.p_claim = p_claim
        self.perceived_utility = self.calculate_perceived_utility()
        self.expected_cost = self.calculate_expected_cost()
        self.indifferent_amount = self.calculate_indifferent_amount()
        self.premium = self.calculate_premium()
        self.profit = self.calculate_profit()
        self.desired_customer = False

    def calculate_utility(self, dollar_amt: float) -> float:
        """Calculate a customer's base utility

        This fillows the utility calculation from the 
        assignment instructions

        Args:
            dollar_amt (float): The dollar ammount of their coverage
        
        Returns:
            float: The base utility value of the customer. 

        """
        return -263.31 + 22.09 * np.log(dollar_amt + 150_000)

    def calculate_perceived_utility(self) -> float:
        """Computer customers precieved utility 

        Uses the base utility and the p_claim to determine 
        the preceived utility of the cusomter.

        Returns:
            float: Customers precieved utitlity 
        """
        utility_if_no_claim = self.calculate_utility(0.0) 
        utility_if_claim = self.calculate_utility(-self.coverage_amt) 
        return (1 - self.p_claim) * utility_if_no_claim + self.p_claim * utility_if_claim

    def solve_for_dollar_amount(self, utility_value: float) -> float:
        """Solve for the dollar ammount by inversing the utility equation

        Args:
            utility_value (float): A utility value.

        Returns:
            float: The dollar amount corresponding to that utility.
        """
        return np.exp((utility_value + 263.31) / 22.09) - 150_000

    def calculate_indifferent_amount(self) -> float:
        """Calculate the dollar ammount the customer is indifferent to buying insuanrce

        This funciton determines the ammount the customer
        would be indifferent to taking insurance vs not taking insurance
        so any ammount lower than this is offering them utility. 

        Returns:
            float: The indifferent dollar ammount
        """
        dollars_lost = self.solve_for_dollar_amount(self.perceived_utility) 
        return -dollars_lost

    def calculate_expected_cost(self) -> float:
        """Compute the insurer’s expected cost based on claim probability.

        Returns:
            float: The expected payout (coverage * probability of claim).
        """
        return self.p_claim * self.coverage_amt

    def calculate_premium(self) -> float:
        """Calculate the customer’s premium using a weighted formula.

        The premium is a weighted average between expected cost and indifferent
        amount, with weights of 0.75 and 0.25 respectively.

        Returns:
            float: The premium amount charged to the customer.
        """
        return (self.expected_cost * 0.75) + (self.indifferent_amount * 0.25)
    
    def calculate_profit(self) -> float:
        """Compute the insurer’s profit from this customer.

        Profit is the difference between the premium charged and the expected cost.

        Returns:
            float: The insurer’s expected profit from this customer.
        """
        return self.premium - self.expected_cost

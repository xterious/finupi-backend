"""
Credit Scoring Module

This module calculates credit scores based on transaction features.
It provides functions for generating scores, explaining components,
and determining loan eligibility.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class CreditScoreCalculator:
    """
    Class to calculate and explain credit scores based on transaction features.
    """
    
    def __init__(self):
        """Initialize the CreditScoreCalculator class."""
        # Define component weights
        self.income_weight = 0.25
        self.expense_weight = 0.25
        self.discipline_weight = 0.25
        self.history_weight = 0.25
        
        # Define score thresholds for explanations
        self.score_thresholds = {
            "excellent": 90,
            "good": 75,
            "fair": 60,
            "poor": 40,
            "very_poor": 0
        }
        
        # Define features relevant to each component
        self.component_features = {
            "income": ["monthly_income", "income_frequency", "income_regularity", "income_sources", "income_growth"],
            "expense": ["monthly_expense", "expense_to_income_ratio", "essential_expense_ratio", "expense_consistency", "large_expense_frequency"],
            "discipline": ["savings_ratio", "low_balance_frequency", "balance_dips", "weekend_spending_ratio"],
            "history": ["transaction_frequency", "merchant_diversity", "data_months", "high_risk_merchant_ratio"]
        }
        
        # Feature importance within each component (must sum to 1 within each component)
        self.feature_weights = {
            # Income stability weights
            "monthly_income": 0.3,
            "income_frequency": 0.15,
            "income_regularity": 0.3,
            "income_sources": 0.15,
            "income_growth": 0.1,
            
            # Expense management weights
            "monthly_expense": 0.15,
            "expense_to_income_ratio": 0.40,
            "essential_expense_ratio": 0.20,
            "expense_consistency": 0.15,
            "large_expense_frequency": 0.10,
            
            # Financial discipline weights
            "savings_ratio": 0.40,
            "low_balance_frequency": 0.25,
            "balance_dips": 0.25,
            "weekend_spending_ratio": 0.10,
            
            # Transaction history weights
            "transaction_frequency": 0.25,
            "merchant_diversity": 0.25,
            "data_months": 0.35,
            "high_risk_merchant_ratio": 0.15
        }
        
        # Define feature scaling functions
        self.feature_scalers = {
            # Income stability scalers
            "monthly_income": lambda x: min(1.0, x / 50000),  # Scale income up to 50k
            "income_frequency": lambda x: min(1.0, x / 30),   # Scale up to 30 transactions per month
            "income_regularity": lambda x: x,                 # Already between 0-1
            "income_sources": lambda x: min(1.0, x / 3),      # Scale up to 3 sources
            "income_growth": lambda x: min(1.0, max(0, (x + 0.1) / 0.2)),  # Scale -0.1 to 0.1 growth
            
            # Expense management scalers
            "monthly_expense": lambda x, monthly_income: 1.0 - min(1.0, x / (monthly_income + 0.01)),  # Lower is better
            "expense_to_income_ratio": lambda x: 1.0 - x,     # Lower is better
            "essential_expense_ratio": lambda x: min(1.0, max(0, 1.25 * x - 0.25)),  # Optimal around 0.6-0.8
            "expense_consistency": lambda x: x,               # Already between 0-1
            "large_expense_frequency": lambda x: 1.0 - min(1.0, x / 5), # Lower is better
            
            # Financial discipline scalers
            "savings_ratio": lambda x: min(1.0, x / 0.2),     # Scale up to 20% savings
            "low_balance_frequency": lambda x: 1.0 - x,       # Lower is better
            "balance_dips": lambda x: 1.0 - x,                # Lower is better
            "weekend_spending_ratio": lambda x: min(1.0, max(0, 1.5 - x / 1.5)), # Optimal around 0.5-1.0
            
            # Transaction history scalers
            "transaction_frequency": lambda x: min(1.0, x / 2),  # Scale up to 2 transactions per day
            "merchant_diversity": lambda x: x,                  # Already between 0-1
            "data_months": lambda x: min(1.0, x / 3),           # Scale up to 3 months
            "high_risk_merchant_ratio": lambda x: 1.0 - min(1.0, x * 10)  # Lower is better, severe penalty
        }
        
        # Define improvement recommendations for each feature
        self.improvement_recommendations = {
            "monthly_income": [
                "Consider exploring additional income sources",
                "Look for opportunities to increase your primary income"
            ],
            "income_frequency": [
                "More frequent income patterns can improve your score",
                "Consider breaking down large income deposits into smaller, more regular amounts"
            ],
            "income_regularity": [
                "Regular, predictable income improves your creditworthiness",
                "Try to establish a consistent income pattern"
            ],
            "income_sources": [
                "Multiple income sources reduce financial risk",
                "Consider diversifying your income streams"
            ],
            "income_growth": [
                "A positive income trend strengthens your financial profile",
                "Focus on increasing your income over time"
            ],
            "expense_to_income_ratio": [
                "Try to keep your expenses below 70% of your income",
                "Reducing non-essential expenses can improve your score"
            ],
            "essential_expense_ratio": [
                "Maintain a healthy balance between essential and discretionary spending",
                "Aim for essential expenses to be 60-80% of total expenses"
            ],
            "expense_consistency": [
                "Consistent monthly expenses show financial stability",
                "Avoid large fluctuations in your spending patterns"
            ],
            "large_expense_frequency": [
                "Minimize large expenses that exceed 25% of your monthly income",
                "Break down large purchases into smaller amounts if possible"
            ],
            "savings_ratio": [
                "Try to save at least 20% of your income",
                "Increasing your savings rate demonstrates financial discipline"
            ],
            "low_balance_frequency": [
                "Avoid maintaining low or negative account balances",
                "Keep a buffer amount in your account at all times"
            ],
            "balance_dips": [
                "Avoid running out of money just before receiving income",
                "Better cash flow management can improve your score"
            ],
            "weekend_spending_ratio": [
                "Maintain balanced spending throughout the week",
                "Excessive weekend spending may indicate poor financial planning"
            ],
            "transaction_frequency": [
                "Regular transaction activity demonstrates active financial management",
                "Very low transaction counts may limit our ability to assess your habits"
            ],
            "merchant_diversity": [
                "Transacting with a wider variety of merchants shows financial flexibility",
                "Extremely limited merchant diversity may impact your score"
            ],
            "data_months": [
                "A longer transaction history allows for better assessment",
                "Continue using your account regularly to build history"
            ],
            "high_risk_merchant_ratio": [
                "Transactions with high-risk merchants can negatively impact your score",
                "Limit transactions with gambling services and frequent cash withdrawals"
            ]
        }
    
    def calculate_score(self, features: Dict) -> Dict:
        """
        Calculate credit score based on transaction features.
        
        Args:
            features: Dictionary of features extracted from transactions
            
        Returns:
            Dict: Credit score and components
        """
        # Calculate component scores
        income_score = self._calculate_income_score(features)
        expense_score = self._calculate_expense_score(features)
        discipline_score = self._calculate_discipline_score(features)
        history_score = self._calculate_history_score(features)
        
        # Calculate overall score (weighted average)
        overall_score = (
            income_score * self.income_weight +
            expense_score * self.expense_weight +
            discipline_score * self.discipline_weight +
            history_score * self.history_weight
        ) * 100  # Scale to 0-100
        
        # Round to nearest integer
        overall_score = round(overall_score)
        
        # Apply minimum score floor of 55 to ensure loan eligibility
        overall_score = max(55, overall_score)
        
        # Generate score category
        score_category = self._get_score_category(overall_score)
        
        # Generate explanations and recommendations
        explanations = self._generate_explanations(features, {
            "income": income_score,
            "expense": expense_score,
            "discipline": discipline_score,
            "history": history_score
        })
        
        # Calculate loan eligibility
        loan_eligibility = self._calculate_loan_eligibility(overall_score, features)
        
        # Prepare result
        result = {
            "overall_score": overall_score,
            "score_category": score_category,
            "component_scores": {
                "income_stability": round(income_score * 100),
                "expense_management": round(expense_score * 100),
                "financial_discipline": round(discipline_score * 100),
                "transaction_history": round(history_score * 100)
            },
            "explanations": explanations,
            "loan_eligibility": loan_eligibility
        }
        
        return result
    
    def _calculate_income_score(self, features: Dict) -> float:
        """
        Calculate income stability component score.
        
        Args:
            features: Dictionary of features
            
        Returns:
            float: Income stability score (0-1)
        """
        # Get relevant features
        monthly_income = features.get("monthly_income", 0)
        income_frequency = features.get("income_frequency", 0)
        income_regularity = features.get("income_regularity", 0)
        income_sources = features.get("income_sources", 0)
        income_growth = features.get("income_growth", 0)
        
        # Apply scaling functions
        scaled_monthly_income = self.feature_scalers["monthly_income"](monthly_income)
        scaled_income_frequency = self.feature_scalers["income_frequency"](income_frequency)
        scaled_income_regularity = self.feature_scalers["income_regularity"](income_regularity)
        scaled_income_sources = self.feature_scalers["income_sources"](income_sources)
        scaled_income_growth = self.feature_scalers["income_growth"](income_growth)
        
        # Calculate weighted score
        income_score = (
            scaled_monthly_income * self.feature_weights["monthly_income"] +
            scaled_income_frequency * self.feature_weights["income_frequency"] +
            scaled_income_regularity * self.feature_weights["income_regularity"] +
            scaled_income_sources * self.feature_weights["income_sources"] +
            scaled_income_growth * self.feature_weights["income_growth"]
        )
        
        return income_score
    
    def _calculate_expense_score(self, features: Dict) -> float:
        """
        Calculate expense management component score.
        
        Args:
            features: Dictionary of features
            
        Returns:
            float: Expense management score (0-1)
        """
        # Get relevant features
        monthly_expense = features.get("monthly_expense", 0)
        expense_to_income_ratio = features.get("expense_to_income_ratio", 1)  # Default to worst case
        essential_expense_ratio = features.get("essential_expense_ratio", 0)
        expense_consistency = features.get("expense_consistency", 0)
        large_expense_frequency = features.get("large_expense_frequency", 5)  # Default to bad case
        monthly_income = features.get("monthly_income", 1)  # To avoid division by zero
        
        # Apply scaling functions
        scaled_monthly_expense = self.feature_scalers["monthly_expense"](monthly_expense, monthly_income)
        scaled_expense_ratio = self.feature_scalers["expense_to_income_ratio"](expense_to_income_ratio)
        scaled_essential_ratio = self.feature_scalers["essential_expense_ratio"](essential_expense_ratio)
        scaled_expense_consistency = self.feature_scalers["expense_consistency"](expense_consistency)
        scaled_large_expense_freq = self.feature_scalers["large_expense_frequency"](large_expense_frequency)
        
        # Calculate weighted score
        expense_score = (
            scaled_monthly_expense * self.feature_weights["monthly_expense"] +
            scaled_expense_ratio * self.feature_weights["expense_to_income_ratio"] +
            scaled_essential_ratio * self.feature_weights["essential_expense_ratio"] +
            scaled_expense_consistency * self.feature_weights["expense_consistency"] +
            scaled_large_expense_freq * self.feature_weights["large_expense_frequency"]
        )
        
        return expense_score
    
    def _calculate_discipline_score(self, features: Dict) -> float:
        """
        Calculate financial discipline component score.
        
        Args:
            features: Dictionary of features
            
        Returns:
            float: Financial discipline score (0-1)
        """
        # Get relevant features
        savings_ratio = features.get("savings_ratio", 0)
        low_balance_frequency = features.get("low_balance_frequency", 1)  # Default to worst case
        balance_dips = features.get("balance_dips", 1)  # Default to worst case
        weekend_spending_ratio = features.get("weekend_spending_ratio", 2)  # Default to bad case
        
        # Apply scaling functions
        scaled_savings_ratio = self.feature_scalers["savings_ratio"](savings_ratio)
        scaled_low_balance_freq = self.feature_scalers["low_balance_frequency"](low_balance_frequency)
        scaled_balance_dips = self.feature_scalers["balance_dips"](balance_dips)
        scaled_weekend_spending = self.feature_scalers["weekend_spending_ratio"](weekend_spending_ratio)
        
        # Calculate weighted score
        discipline_score = (
            scaled_savings_ratio * self.feature_weights["savings_ratio"] +
            scaled_low_balance_freq * self.feature_weights["low_balance_frequency"] +
            scaled_balance_dips * self.feature_weights["balance_dips"] +
            scaled_weekend_spending * self.feature_weights["weekend_spending_ratio"]
        )
        
        return discipline_score
    
    def _calculate_history_score(self, features: Dict) -> float:
        """
        Calculate transaction history component score.
        
        Args:
            features: Dictionary of features
            
        Returns:
            float: Transaction history score (0-1)
        """
        # Get relevant features
        transaction_frequency = features.get("transaction_frequency", 0)
        merchant_diversity = features.get("merchant_diversity", 0)
        data_months = features.get("data_months", 0)
        high_risk_merchant_ratio = features.get("high_risk_merchant_ratio", 0.3)  # Default to bad case
        
        # Apply scaling functions
        scaled_transaction_freq = self.feature_scalers["transaction_frequency"](transaction_frequency)
        scaled_merchant_diversity = self.feature_scalers["merchant_diversity"](merchant_diversity)
        scaled_data_months = self.feature_scalers["data_months"](data_months)
        scaled_high_risk_ratio = self.feature_scalers["high_risk_merchant_ratio"](high_risk_merchant_ratio)
        
        # Calculate weighted score
        history_score = (
            scaled_transaction_freq * self.feature_weights["transaction_frequency"] +
            scaled_merchant_diversity * self.feature_weights["merchant_diversity"] +
            scaled_data_months * self.feature_weights["data_months"] +
            scaled_high_risk_ratio * self.feature_weights["high_risk_merchant_ratio"]
        )
        
        return history_score
    
    def _get_score_category(self, score: float) -> str:
        """
        Get the category label for a given score.
        
        Args:
            score: Credit score (0-100)
            
        Returns:
            str: Score category label
        """
        if score >= self.score_thresholds["excellent"]:
            return "excellent"
        elif score >= self.score_thresholds["good"]:
            return "good"
        elif score >= self.score_thresholds["fair"]:
            return "fair"
        elif score >= self.score_thresholds["poor"]:
            return "poor"
        else:
            return "very_poor"
    
    def _generate_explanations(self, features: Dict, component_scores: Dict) -> Dict:
        """
        Generate explanations and improvement recommendations.
        
        Args:
            features: Dictionary of features
            component_scores: Dictionary of component scores
            
        Returns:
            Dict: Explanations and recommendations
        """
        explanations = {}
        
        # Generate component explanations
        explanations["components"] = {
            "income_stability": self._explain_income_stability(features, component_scores["income"]),
            "expense_management": self._explain_expense_management(features, component_scores["expense"]),
            "financial_discipline": self._explain_financial_discipline(features, component_scores["discipline"]),
            "transaction_history": self._explain_transaction_history(features, component_scores["history"])
        }
        
        # Find weak components for improvement recommendations
        weak_components = []
        for component, score in component_scores.items():
            if score < 0.7:  # Consider components with score < 70% as weak
                weak_components.append(component)
        
        # Generate improvement recommendations
        explanations["improvement_recommendations"] = self._generate_recommendations(features, weak_components)
        
        return explanations
    
    def _explain_income_stability(self, features: Dict, score: float) -> Dict:
        """
        Generate explanation for income stability component.
        
        Args:
            features: Dictionary of features
            score: Component score (0-1)
            
        Returns:
            Dict: Explanation
        """
        monthly_income = features.get("monthly_income", 0)
        income_regularity = features.get("income_regularity", 0)
        income_sources = features.get("income_sources", 0)
        
        score_category = "excellent" if score > 0.9 else "good" if score > 0.7 else "fair" if score > 0.5 else "poor"
        
        explanation = {
            "summary": f"Your income stability is {score_category}.",
            "details": []
        }
        
        if monthly_income > 0:
            explanation["details"].append(f"Your monthly income is approximately ₹{monthly_income:.0f}.")
        
        if income_regularity > 0.8:
            explanation["details"].append("You have very regular income patterns.")
        elif income_regularity > 0.5:
            explanation["details"].append("You have somewhat regular income patterns.")
        else:
            explanation["details"].append("Your income patterns are irregular.")
        
        if income_sources > 2:
            explanation["details"].append("You have multiple income sources, which is positive.")
        elif income_sources > 1:
            explanation["details"].append("You have more than one income source.")
        else:
            explanation["details"].append("You rely on a single income source.")
        
        return explanation
    
    def _explain_expense_management(self, features: Dict, score: float) -> Dict:
        """
        Generate explanation for expense management component.
        
        Args:
            features: Dictionary of features
            score: Component score (0-1)
            
        Returns:
            Dict: Explanation
        """
        expense_ratio = features.get("expense_to_income_ratio", 1)
        essential_ratio = features.get("essential_expense_ratio", 0)
        
        score_category = "excellent" if score > 0.9 else "good" if score > 0.7 else "fair" if score > 0.5 else "poor"
        
        explanation = {
            "summary": f"Your expense management is {score_category}.",
            "details": []
        }
        
        if expense_ratio < 0.7:
            explanation["details"].append(f"You spend around {expense_ratio*100:.0f}% of your income, which is good.")
        elif expense_ratio < 0.9:
            explanation["details"].append(f"You spend around {expense_ratio*100:.0f}% of your income.")
        else:
            explanation["details"].append(f"You spend around {expense_ratio*100:.0f}% of your income, which is high.")
        
        if essential_ratio > 0.8:
            explanation["details"].append("Most of your spending is on essential expenses.")
        elif essential_ratio > 0.6:
            explanation["details"].append("You maintain a good balance between essential and non-essential expenses.")
        else:
            explanation["details"].append("A large portion of your spending is on non-essential items.")
        
        return explanation
    
    def _explain_financial_discipline(self, features: Dict, score: float) -> Dict:
        """
        Generate explanation for financial discipline component.
        
        Args:
            features: Dictionary of features
            score: Component score (0-1)
            
        Returns:
            Dict: Explanation
        """
        savings_ratio = features.get("savings_ratio", 0)
        low_balance_frequency = features.get("low_balance_frequency", 1)
        
        score_category = "excellent" if score > 0.9 else "good" if score > 0.7 else "fair" if score > 0.5 else "poor"
        
        explanation = {
            "summary": f"Your financial discipline is {score_category}.",
            "details": []
        }
        
        if savings_ratio > 0.2:
            explanation["details"].append(f"You save around {savings_ratio*100:.0f}% of your income, which is excellent.")
        elif savings_ratio > 0.1:
            explanation["details"].append(f"You save around {savings_ratio*100:.0f}% of your income, which is good.")
        elif savings_ratio > 0:
            explanation["details"].append(f"You save around {savings_ratio*100:.0f}% of your income, which could be improved.")
        else:
            explanation["details"].append("You are spending more than your income, which is concerning.")
        
        if low_balance_frequency < 0.1:
            explanation["details"].append("You rarely have a low account balance, which is excellent.")
        elif low_balance_frequency < 0.3:
            explanation["details"].append("You occasionally have a low account balance.")
        else:
            explanation["details"].append(f"You frequently have a low account balance ({low_balance_frequency*100:.0f}% of days).")
        
        return explanation
    
    def _explain_transaction_history(self, features: Dict, score: float) -> Dict:
        """
        Generate explanation for transaction history component.
        
        Args:
            features: Dictionary of features
            score: Component score (0-1)
            
        Returns:
            Dict: Explanation
        """
        data_months = features.get("data_months", 0)
        merchant_diversity = features.get("merchant_diversity", 0)
        high_risk_ratio = features.get("high_risk_merchant_ratio", 0)
        
        score_category = "excellent" if score > 0.9 else "good" if score > 0.7 else "fair" if score > 0.5 else "poor"
        
        explanation = {
            "summary": f"Your transaction history is {score_category}.",
            "details": []
        }
        
        if data_months < 1:
            explanation["details"].append(f"We have only {data_months*30:.0f} days of transaction history.")
        elif data_months < 3:
            explanation["details"].append(f"We have {data_months:.1f} months of transaction history.")
        else:
            explanation["details"].append(f"We have {data_months:.1f} months of transaction history, which is good.")
        
        merchant_count = merchant_diversity * 10  # Approximate, since we normalized by dividing by 10
        if merchant_count > 5:
            explanation["details"].append(f"You transact with a diverse set of merchants ({merchant_count:.0f}+ different merchants).")
        else:
            explanation["details"].append(f"You transact with a limited set of merchants ({merchant_count:.0f} different merchants).")
        
        if high_risk_ratio > 0.05:
            explanation["details"].append(f"{high_risk_ratio*100:.1f}% of your transactions are with high-risk merchants.")
        
        return explanation
    
    def _generate_recommendations(self, features: Dict, weak_components: List[str]) -> List[str]:
        """
        Generate improvement recommendations for weak components.
        
        Args:
            features: Dictionary of features
            weak_components: List of component names with low scores
            
        Returns:
            List: Improvement recommendations
        """
        recommendations = []
        
        # Look for the weakest features within weak components
        for component in weak_components:
            component_features = self.component_features.get(component, [])
            
            # Get feature scores for this component
            feature_scores = {}
            for feature in component_features:
                if feature in features:
                    # Calculate normalized feature score (0-1)
                    if feature == "monthly_expense":
                        monthly_income = features.get("monthly_income", 1)
                        score = self.feature_scalers[feature](features[feature], monthly_income)
                    else:
                        # Apply appropriate scaling function
                        try:
                            score = self.feature_scalers[feature](features[feature])
                        except TypeError:
                            # Some scalers might need additional parameters, use default
                            score = 0.5
                    
                    feature_scores[feature] = score
            
            # Find the weakest features (lowest scores)
            weak_features = sorted(feature_scores.items(), key=lambda x: x[1])[:2]
            
            # Add recommendations for weak features
            for feature, score in weak_features:
                if score < 0.7 and feature in self.improvement_recommendations:
                    # Randomly select one recommendation for this feature
                    import random
                    recommendation = random.choice(self.improvement_recommendations[feature])
                    recommendations.append(recommendation)
        
        # Limit to top 3 recommendations
        return recommendations[:3]
    
    def _calculate_loan_eligibility(self, score: float, features: Dict) -> Dict:
        """
        Calculate loan eligibility based on credit score and income.
        
        Args:
            score: Credit score (0-100)
            features: Dictionary of features
            
        Returns:
            Dict: Loan eligibility details
        """
        monthly_income = features.get("monthly_income", 0)
        expense_ratio = features.get("expense_to_income_ratio", 1)
        
        # Calculate disposable income
        disposable_income = monthly_income * (1 - expense_ratio)
        
        # Calculate maximum loan amount (3x monthly income for excellent scores, scaling down for lower scores)
        # With a minimum floor to ensure everyone gets a fair chance
        score_factor = max(0.3, score / 100)  # Minimum factor of 0.3 instead of 0.1
        max_loan_amount = monthly_income * 3 * score_factor
        
        # Calculate EMI capacity (50% of disposable income)
        emi_capacity = disposable_income * 0.5
        
        # Calculate interest rate based on score
        if score >= 95:
            interest_rate = 1.2
        elif score >= 90:
            interest_rate = 1.3
        elif score >= 85:
            interest_rate = 1.4
        elif score >= 80:
            interest_rate = 1.5
        elif score >= 75:
            interest_rate = 1.6
        elif score >= 70:
            interest_rate = 1.7
        elif score >= 65:
            interest_rate = 1.8
        elif score >= 60:
            interest_rate = 1.9
        elif score >= 55:
            interest_rate = 2.0
        elif score >= 50:
            interest_rate = 2.1
        elif score >= 45:
            interest_rate = 2.2
        elif score >= 40:
            interest_rate = 2.3
        elif score >= 35:
            interest_rate = 2.4
        else:
            interest_rate = 2.5
        
        # Calculate maximum loan duration (in months)
        max_duration = 12
        
        # Calculate monthly EMI for maximum loan
        # Using simple interest for simplicity in microloans
        monthly_interest_rate = interest_rate / 100 / 12
        total_interest = max_loan_amount * monthly_interest_rate * max_duration
        monthly_emi = (max_loan_amount + total_interest) / max_duration
        
        # Adjust loan amount if EMI exceeds capacity
        if monthly_emi > emi_capacity and emi_capacity > 0:
            # Recalculate max loan amount based on EMI capacity
            max_loan_amount = (emi_capacity * max_duration) / (1 + monthly_interest_rate * max_duration)
            monthly_emi = emi_capacity
        
        # Round to nearest 1000
        max_loan_amount = round(max_loan_amount / 1000) * 1000
        
        # Set minimum and maximum limits
        min_loan = 5000
        absolute_max_loan = 50000
        
        max_loan_amount = max(min_loan, min(max_loan_amount, absolute_max_loan))
        
        return {
            "eligible": True,  # Always eligible since we have a floor of 55
            "max_loan_amount": int(max_loan_amount),
            "interest_rate": interest_rate,
            "max_duration_months": max_duration,
            "monthly_emi": int(monthly_emi),
            "disposable_income": int(disposable_income)
        }


def calculate_credit_score(features: Dict) -> Dict:
    """
    Calculate credit score based on transaction features.
    
    Args:
        features: Dictionary of features extracted from transactions
        
    Returns:
        Dict: Credit score and components
    """
    calculator = CreditScoreCalculator()
    return calculator.calculate_score(features)


if __name__ == "__main__":
    # Test with sample features
    from transaction_parser import analyze_transactions
    from data_ingestion import load_sample_data
    
    # Load sample data
    sample_data = load_sample_data()
    
    # Extract features
    processed_data, features = analyze_transactions(sample_data)
    
    # Calculate credit score
    credit_score_result = calculate_credit_score(features)
    
    # Print results
    print("\nCredit Score Results:")
    print(f"Overall Score: {credit_score_result['overall_score']} - {credit_score_result['score_category'].capitalize()}")
    
    print("\nComponent Scores:")
    for component, score in credit_score_result['component_scores'].items():
        print(f"  {component}: {score}")
    
    print("\nLoan Eligibility:")
    loan = credit_score_result['loan_eligibility']
    print(f"  Eligible: {loan['eligible']}")
    print(f"  Maximum Loan: ₹{loan['max_loan_amount']:,}")
    print(f"  Interest Rate: {loan['interest_rate']}%")
    print(f"  Max Duration: {loan['max_duration_months']} months")
    print(f"  Monthly EMI: ₹{loan['monthly_emi']:,}")
    
    print("\nImprovement Recommendations:")
    for rec in credit_score_result['explanations']['improvement_recommendations']:
        print(f"  - {rec}") 
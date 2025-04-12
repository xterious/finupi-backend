from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import json
import os
import google.generativeai as genai
from typing import Dict, List, Tuple

# Import your existing components
from data_ingestion import TransactionData
from transaction_parser import TransactionParser, analyze_transactions
from credit_score import calculate_credit_score

app = Flask(__name__)

# MongoDB Connection
mongo_uri = "mongodb+srv://root:root@ipo-builder.lpq9ub9.mongodb.net/"
client = MongoClient(mongo_uri)
db = client["fin_credit_db"]  # Database name
transactions_collection = db["transactions"]  # Collection for storing transactions
credit_scores_collection = db["credit_scores"]  # Collection for storing credit scores

# Helper function to convert MongoDB data to DataFrame with proper structure
def prepare_transaction_data(transactions):
    """Convert MongoDB transaction data to the expected format for the model"""
    # Map fields from MongoDB to our expected format
    df = pd.DataFrame(transactions)
    
    # Ensure required columns exist
    required_columns = [
        'transaction_date', 'amount', 'transaction_type', 
        'description', 'sender_upi_id', 'receiver_upi_id'
    ]
    
    # Handle different timestamp/date formats
    if 'transaction_date' not in df.columns:
        if 'Timestamp' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['Timestamp'])
        elif 'timestamp' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['date'])
    else:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Handle amount field
    if 'amount' not in df.columns:
        if 'Amount (INR)' in df.columns:
            df['amount'] = df['Amount (INR)'].astype(float)
        elif 'Amount' in df.columns:
            df['amount'] = df['Amount'].astype(float)
        elif 'value' in df.columns:
            df['amount'] = df['value'].astype(float)
    elif 'amount' in df.columns:
        df['amount'] = df['amount'].astype(float)
    
    # Handle UPI IDs
    if 'sender_upi_id' not in df.columns:
        if 'Sender UPI ID' in df.columns:
            df['sender_upi_id'] = df['Sender UPI ID']
        elif 'sender' in df.columns:
            df['sender_upi_id'] = df['sender']
        else:
            df['sender_upi_id'] = 'unknown@upi'
    
    if 'receiver_upi_id' not in df.columns:
        if 'Receiver UPI ID' in df.columns:
            df['receiver_upi_id'] = df['Receiver UPI ID']
        elif 'receiver' in df.columns:
            df['receiver_upi_id'] = df['receiver']
        else:
            df['receiver_upi_id'] = 'unknown@upi'
    
    # Handle different transaction type formats
    if 'transaction_type' not in df.columns:
        if 'Type' in df.columns:
            # Map 'Sent' to 'debit' and 'Received' to 'credit'
            type_mapping = {
                'Sent': 'debit', 
                'Received': 'credit', 
                'SEND': 'debit', 
                'RECEIVE': 'credit',
                'send': 'debit',
                'receive': 'credit'
            }
            df['transaction_type'] = df['Type'].map(lambda x: type_mapping.get(x, 'debit'))
        elif 'type' in df.columns:
            # Map 'RECEIVE' to 'credit' and 'SEND' to 'debit'
            type_mapping = {'RECEIVE': 'credit', 'SEND': 'debit', 'receive': 'credit', 'send': 'debit'}
            df['transaction_type'] = df['type'].map(lambda x: type_mapping.get(x, x.lower() if isinstance(x, str) else 'debit'))
    else:
        # Ensure lowercase consistency
        df['transaction_type'] = df['transaction_type'].str.lower()
    
    # Handle description field
    if 'description' not in df.columns:
        if 'Note' in df.columns:
            df['description'] = df['Note']
        elif 'note' in df.columns:
            df['description'] = df['note']
        elif 'narration' in df.columns:
            df['description'] = df['narration']
        else:
            # Generate description from available data
            df['description'] = df.apply(lambda row: 
                f"Transaction of {row.get('amount', 'unknown amount')} from {row.get('sender_upi_id', 'unknown')} to {row.get('receiver_upi_id', 'unknown')}", 
                axis=1)
    
    # Add merchant field if not present
    if 'merchant' not in df.columns:
        # Extract merchant from description or use receiver_upi_id
        df['merchant'] = df.apply(lambda row: 
            row['receiver_upi_id'].split('@')[0] if row['transaction_type'] == 'debit' 
            else row['sender_upi_id'].split('@')[0], 
            axis=1)
    
    # Add any missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Add transaction_ref if not present
    if 'transaction_ref' not in df.columns:
        if 'Reference ID' in df.columns:
            df['transaction_ref'] = df['Reference ID']
        elif 'Status' in df.columns:
            # Use Status + timestamp to create a unique reference
            df['transaction_ref'] = df.apply(
                lambda row: f"TX-{row['Status']}-{pd.to_datetime(row['transaction_date']).strftime('%Y%m%d%H%M%S')}",
                axis=1
            )
        else:
            df['transaction_ref'] = [f"tx_{i}" for i in range(len(df))]
    
    return df

def get_deepseek_suggestions(explanation: str, improvements: List[str]) -> str:
    """
    Generate personalized credit improvement suggestions using Gemini LLM.
    
    Args:
        explanation: Explanation of the user's financial behavior
        improvements: List of improvement suggestions
        
    Returns:
        str: LLM-generated personalized improvement suggestion
    """
    prompt = f"""
    A user has the following financial behavior:
    - Explanation: {explanation}
    - Suggestions: {', '.join(improvements) if improvements else "None"}

    Write a professional but friendly 5-6 sentence suggestion to help the user improve their credit score.
    """

    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
        response = model.generate_content([prompt])  # Make sure it's wrapped in a list
        return response.text.strip()
    except Exception as e:
        return f"To improve your credit score, follow the recommendations provided. If you need more personalized advice, please consult with a financial advisor."

@app.route('/upload_transactions', methods=['POST'])
def upload_transactions():
    data = request.get_json()
    
    if not data or 'user_id' not in data or 'transactions' not in data:
        return jsonify({'error': 'Missing user_id or transactions data'}), 400
        
    try:
        user_id = data['user_id']
        transactions = data['transactions']
        
        # Check if user already exists
        existing_user = transactions_collection.find_one({'user_id': user_id})
        
        if existing_user:
            # Update existing user's transactions
            transactions_collection.update_one(
                {'user_id': user_id},
                {'$set': {'transactions': transactions}}
            )
            return jsonify({
                'message': f'Updated transactions for user {user_id}',
                'transaction_count': len(transactions)
            })
        else:
            # Create new user document
            user_document = {
                'user_id': user_id,
                'transactions': transactions
            }
            transactions_collection.insert_one(user_document)
            return jsonify({
                'message': f'Added new user {user_id} with transactions',
                'transaction_count': len(transactions)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_credit_score', methods=['POST'])
def get_credit_score():
    data = request.get_json()

    if not data or 'user_id' not in data:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    try:
        user_id = data['user_id']
        
        # Fetch user document from MongoDB
        user_data = transactions_collection.find_one({'user_id': user_id})
        
        if not user_data:
            return jsonify({'error': f'No user found with ID {user_id}. Please upload transactions first.'}), 404
            
        # Access transactions based on schema structure
        # Check both root level 'transactions' and possible nested paths
        transactions = None
        if 'transactions' in user_data and user_data['transactions']:
            transactions = user_data['transactions']
        elif 'data' in user_data and 'transactions' in user_data['data']:
            transactions = user_data['data']['transactions']
        
        if not transactions:
            return jsonify({'error': f'No transactions found for user {user_id}. Please upload transactions first.'}), 404
        
        # Convert transactions array to DataFrame
        df = prepare_transaction_data(transactions)
        
        # Create a TransactionData object and validate
        transaction_data = TransactionData()
        transaction_data.data = df
        if not transaction_data.validate_data():
            return jsonify({
                'error': 'Invalid transaction data',
                'validation_errors': transaction_data.validation_errors
            }), 400
        
        # Preprocess data
        preprocessed_data = transaction_data.preprocess_data()
        
        # Analyze transactions and extract features
        processed_data, features = analyze_transactions(preprocessed_data)
        
        # Calculate credit score
        credit_score_result = calculate_credit_score(features)
        
        # Get last 5 transactions
        last_5 = df.sort_values(by='transaction_date', ascending=False).head(5)
        last_5_dict = last_5.to_dict(orient='records')
        
        # Format datetime objects to strings for JSON serialization
        for tx in last_5_dict:
            if isinstance(tx.get('transaction_date'), pd.Timestamp):
                tx['transaction_date'] = tx['transaction_date'].strftime('%Y-%m-%d')
        
        # Generate personalized advice using Gemini
        improvement_recommendations = credit_score_result['explanations']['improvement_recommendations']
        explanation = "Your credit score analysis shows strengths and weaknesses in your financial behavior."
        personalized_advice = get_deepseek_suggestions(explanation, improvement_recommendations)
        
        # Store credit score result
        credit_score_record = {
            'user_id': user_id,
            'credit_score': credit_score_result['overall_score'],
            'score_category': credit_score_result['score_category'],
            'timestamp': datetime.now(),
            'component_scores': credit_score_result['component_scores'],
            'loan_eligibility': credit_score_result['loan_eligibility'],
            'improvement_recommendations': credit_score_result['explanations']['improvement_recommendations'],
            'personalized_advice': personalized_advice
        }
        
        # Update existing record or create new one if it doesn't exist
        credit_scores_collection.update_one(
            {'user_id': user_id},  # Filter by user_id
            {'$set': credit_score_record},  # Update with new data
            upsert=True  # Create if doesn't exist
        )

        return jsonify({
            'credit_score': credit_score_result['overall_score'],
            'score_category': credit_score_result['score_category'],
            'component_scores': credit_score_result['component_scores'],
            'loan_eligibility': credit_score_result['loan_eligibility'],
            'improvement_recommendations': credit_score_result['explanations']['improvement_recommendations'],
            'personalized_advice': personalized_advice,
            'last_5_transactions': last_5_dict
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/parse_transactions', methods=['POST'])
def parse_transactions():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Missing request data'}), 400
    
    transactions = None
    # Check different possible locations of transaction data
    if 'transactions' in data:
        transactions = data['transactions']
    elif 'user_id' in data:
        # Try to get transactions from MongoDB
        user_id = data['user_id']
        user_data = transactions_collection.find_one({'user_id': user_id})
        
        if user_data and 'transactions' in user_data:
            transactions = user_data['transactions']
        elif user_data and 'data' in user_data and 'transactions' in user_data['data']:
            transactions = user_data['data']['transactions']
    
    if not transactions:
        return jsonify({'error': 'No transaction data found'}), 400
        
    try:
        # Convert json to DataFrame
        df = prepare_transaction_data(transactions)
        
        # Create transaction data object and validate
        transaction_data = TransactionData()
        transaction_data.data = df
        if not transaction_data.validate_data():
            return jsonify({
                'error': 'Invalid transaction data',
                'validation_errors': transaction_data.validation_errors
            }), 400
        
        # Preprocess data
        preprocessed_data = transaction_data.preprocess_data()
        
        # Analyze transactions
        processed_data, features = analyze_transactions(preprocessed_data)
        
        # Get summary stats
        summary = transaction_data.get_summary_stats()
        
        # Format features for JSON serialization
        formatted_features = {}
        for k, v in features.items():
            if isinstance(v, (np.integer, np.floating)):
                formatted_features[k] = float(v)
            else:
                formatted_features[k] = v
        
        return jsonify({
            'summary': {
                'total_transactions': summary.get('total_transactions', 0),
                'date_range': summary.get('date_range', ['N/A', 'N/A']),
                'total_credit': summary.get('total_credit', 0),
                'total_debit': summary.get('total_debit', 0),
                'credit_amount': float(summary.get('credit_amount', 0)),
                'debit_amount': float(summary.get('debit_amount', 0)),
                'avg_transaction': float(summary.get('avg_transaction', 0)),
                'transaction_frequency': float(summary.get('transaction_frequency', 0))
            },
            'features': formatted_features
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/user_credit_history/<user_id>', methods=['GET'])
def get_user_credit_history(user_id):
    try:
        # Get user credit score (now we only have one record per user)
        score = credit_scores_collection.find_one(
            {'user_id': user_id},
            {'_id': 0}  # Exclude MongoDB ObjectId
        )
        
        if not score:
            return jsonify({'message': f'No credit score found for user {user_id}'}), 404
        
        # Convert datetime objects to strings for JSON serialization
        if isinstance(score.get('timestamp'), datetime):
            score['timestamp'] = score['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'user_id': user_id,
            'credit_score': score
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/financial_literacy_content', methods=['GET'])
def get_financial_literacy_content():
    """
    Endpoint to provide simple financial literacy content with a single quiz question.
    Returns a heading, one-liner explanation, and a related quiz question.
    """
    try:
        # Check if topic is specified
        topic = request.args.get('topic', None)
        
        # Define available financial literacy topics
        topics = [
            "Budgeting",
            "Saving",
            "Credit Score",
            "Emergency Fund",
            "Debt Management",
            "Investments",
            "UPI Safety",
            "Insurance",
            "Tax Planning",
            "Retirement Planning"
        ]
        
        # If no topic specified, return a random one
        if not topic:
            import random
            topic = random.choice(topics)
        
        # Generate content with Gemini or use fallback
        try:
            prompt = f"""
            Create a simple financial literacy tip about "{topic}" with exactly this JSON structure (provide just the JSON, no explanation):
            {{
                "heading": "A short clear heading about {topic}",
                "explanation": "A single sentence explaining the key concept of {topic}",
                "quiz": {{
                    "question": "A single question testing knowledge about {topic}",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_index": 0,
                    "explanation": "Brief explanation of the correct answer"
                }}
            }}
            Make the content educational and easy to understand.
            """
            
            model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
            response = model.generate_content([prompt])
            
            try:
                # Parse the JSON response
                content_json = json.loads(response.text)
                return jsonify(content_json)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
                if json_match:
                    content_json = json.loads(json_match.group(1))
                    return jsonify(content_json)
                json_match = re.search(r'```\n(.*?)\n```', response.text, re.DOTALL)
                if json_match:
                    content_json = json.loads(json_match.group(1))
                    return jsonify(content_json)
                # Fallback to pre-defined content
                return jsonify(get_simple_financial_tip(topic))
                
        except Exception as e:
            # Fallback to pre-defined content
            return jsonify(get_simple_financial_tip(topic))
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
def get_simple_financial_tip(topic):
    """Provide pre-defined financial literacy content as fallback"""
    tips = {
        "Budgeting": {
            "heading": "The 50/30/20 Rule",
            "explanation": "Allocate 50% of your income to needs, 30% to wants, and 20% to savings and debt repayment.",
            "quiz": {
                "question": "Which category in the 50/30/20 rule includes groceries?",
                "options": ["Needs", "Wants", "Savings", "Debt repayment"],
                "correct_index": 0,
                "explanation": "Groceries are considered essential expenses and fall under the 'needs' category."
            }
        },
        "Saving": {
            "heading": "Pay Yourself First",
            "explanation": "Set aside money for savings before spending on discretionary items to build wealth consistently.",
            "quiz": {
                "question": "What does 'paying yourself first' mean?",
                "options": ["Spending money on yourself", "Saving before expenses", "Avoiding all expenses", "Borrowing money"],
                "correct_index": 1,
                "explanation": "It means automatically setting aside savings before spending on other expenses."
            }
        },
        "Credit Score": {
            "heading": "Credit Score Factors",
            "explanation": "Your payment history and credit utilization ratio have the biggest impact on your credit score.",
            "quiz": {
                "question": "What is considered a good credit utilization ratio?",
                "options": ["Below 30%", "Above 50%", "Exactly 100%", "As high as possible"],
                "correct_index": 0,
                "explanation": "Keeping your credit utilization below 30% is generally recommended for a good credit score."
            }
        },
        "Emergency Fund": {
            "heading": "3-6 Months of Expenses",
            "explanation": "An emergency fund should ideally cover 3-6 months of essential expenses to handle unexpected financial shocks.",
            "quiz": {
                "question": "Where should you keep your emergency fund?",
                "options": ["In a savings account", "Invested in stocks", "In cryptocurrency", "As cash at home"],
                "correct_index": 0,
                "explanation": "Emergency funds should be kept in easily accessible accounts like savings accounts."
            }
        },
        "Debt Management": {
            "heading": "Avalanche vs. Snowball Method",
            "explanation": "The avalanche method (paying highest interest first) saves more money, while the snowball method (paying smallest debts first) provides psychological wins.",
            "quiz": {
                "question": "Which debt repayment strategy saves the most money?",
                "options": ["Avalanche method", "Snowball method", "Minimum payments", "Consolidation"],
                "correct_index": 0,
                "explanation": "The avalanche method saves more money by targeting high-interest debt first."
            }
        },
        "Investments": {
            "heading": "Compound Interest",
            "explanation": "Compound interest allows your investments to grow exponentially as you earn returns on both your principal and accumulated interest.",
            "quiz": {
                "question": "Why is starting to invest early so important?",
                "options": ["Compound interest", "Lower fees", "Simpler options", "Less paperwork"],
                "correct_index": 0,
                "explanation": "Starting early gives compound interest more time to work in your favor."
            }
        },
        "UPI Safety": {
            "heading": "Never Share OTP",
            "explanation": "Never share your UPI PIN or OTP with anyone, not even customer support, as legitimate services never ask for this information.",
            "quiz": {
                "question": "What should you do if someone requests your UPI PIN?",
                "options": ["Never share it", "Share only with banks", "Share if they offer money", "Share if they're from customer service"],
                "correct_index": 0,
                "explanation": "Never share your UPI PIN with anyone under any circumstances."
            }
        },
        "Insurance": {
            "heading": "Term vs. Whole Life Insurance",
            "explanation": "Term life insurance offers more coverage for less money but expires, while whole life insurance costs more but includes an investment component.",
            "quiz": {
                "question": "Which type of life insurance typically offers the most coverage for the lowest premium?",
                "options": ["Term life", "Whole life", "Universal life", "Variable life"],
                "correct_index": 0,
                "explanation": "Term life insurance provides the most coverage for the lowest cost but expires after a specific term."
            }
        },
        "Tax Planning": {
            "heading": "Tax-Advantaged Accounts",
            "explanation": "Utilizing tax-advantaged accounts like 401(k)s, IRAs, and HSAs can significantly reduce your tax burden while helping you save for important goals.",
            "quiz": {
                "question": "What is the main benefit of tax-advantaged accounts?",
                "options": ["Tax savings", "Higher returns", "No risk", "Unlimited contributions"],
                "correct_index": 0,
                "explanation": "Tax-advantaged accounts provide tax benefits either when contributing or withdrawing funds."
            }
        },
        "Retirement Planning": {
            "heading": "The 4% Rule",
            "explanation": "The 4% rule suggests you can withdraw 4% of your retirement savings annually with minimal risk of running out of money during a 30-year retirement.",
            "quiz": {
                "question": "According to the 4% rule, how much would you need saved to withdraw ₹40,000 monthly in retirement?",
                "options": ["₹1.2 crore", "₹48 lakh", "₹24 lakh", "₹96 lakh"],
                "correct_index": 0,
                "explanation": "₹40,000 monthly = ₹4.8 lakh annually. Using the 4% rule, you would need 25 times this amount (₹1.2 crore)."
            }
        }
    }
    
    # Return the specified topic or a default one
    return tips.get(topic, tips["Budgeting"])

@app.route('/financial_module/<module_id>', methods=['GET'])
def get_module_content(module_id):
    """Get detailed content for a specific financial literacy module"""
    try:
        modules = [
            "Budgeting Basics",
            "Saving Strategies",
            "Understanding Credit",
            "Debt Management",
            "Investment Fundamentals",
            "UPI and Digital Payments",
            "Tax Planning for Beginners",
            "Insurance Essentials"
        ]
        
        # Validate module_id
        try:
            module_index = int(module_id) - 1
            if module_index < 0 or module_index >= len(modules):
                return jsonify({"error": "Invalid module ID"}), 400
            module_name = modules[module_index]
        except ValueError:
            return jsonify({"error": "Module ID must be a number"}), 400
            
        return generate_module_content(module_name)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def generate_module_content(module_name):
    """Generate detailed content for a financial literacy module using Gemini AI"""
    try:
        # Generate content structure with Gemini
        prompt = f"""
        Create a detailed financial literacy module about "{module_name}" for users of a UPI-based credit scoring app.
        
        Return ONLY a JSON object with this exact structure (no explanation, just the JSON):
        {{
            "module_title": "{module_name}",
            "module_description": "A detailed 1-2 sentence description of the module",
            "completed": false,
            "lessons": [
                {{
                    "lesson_id": 1,
                    "title": "Lesson title",
                    "content": "A detailed educational paragraph about this specific aspect of {module_name} (about 4-5 sentences)",
                    "completed": false,
                    "key_points": ["Key point 1", "Key point 2", "Key point 3"]
                }},
                // 2 more similar lessons
            ],
            "quiz": {{
                "quiz_id": 1,
                "title": "Quiz on {module_name}",
                "completed": false,
                "questions": [
                    {{
                        "question_id": 1,
                        "question": "A question related to this module?",
                        "options": ["Option A", "Option B", "Option C", "Option D"],
                        "correct_answer": 0,  // Index of correct option (0-based)
                        "explanation": "Why this answer is correct"
                    }},
                    // 4 more similar questions
                ]
            }}
        }}
        
        The lessons should progress logically from basic to more advanced concepts.
        Make the quiz questions relevant to the lesson content and of varying difficulty.
        Ensure all content is accurate, educational, and user-friendly.
        """
        
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
        response = model.generate_content([prompt])
        
        try:
            # Parse the JSON response
            content_json = json.loads(response.text)
            return jsonify(content_json)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from text
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
            if json_match:
                content_json = json.loads(json_match.group(1))
                return jsonify(content_json)
            
            # Fallback to manual structure if JSON extraction fails
            return jsonify(get_fallback_module_content(module_name))
            
    except Exception as e:
        # Return fallback content if Gemini fails
        return jsonify(get_fallback_module_content(module_name))

def get_module_description(module_name):
    """Get a short description for each module"""
    descriptions = {
        "Budgeting Basics": "Learn how to create and maintain a personal budget, track expenses, and allocate funds effectively.",
        "Saving Strategies": "Discover various saving techniques, emergency funds, and how to set achievable financial goals.",
        "Understanding Credit": "Explore credit scores, credit reports, and responsible credit card usage.",
        "Debt Management": "Learn strategies for managing and reducing debt, understanding interest rates, and loan options.",
        "Investment Fundamentals": "Understand different investment vehicles, risk assessment, and building a balanced portfolio.",
        "UPI and Digital Payments": "Master the use of UPI, digital wallets, and secure online transaction practices.",
        "Tax Planning for Beginners": "Learn basic tax-saving strategies and understanding tax implications on various incomes.",
        "Insurance Essentials": "Understand different types of insurance and how to select the right coverage for your needs."
    }
    return descriptions.get(module_name, "Explore this financial topic to improve your financial literacy.")

def get_fallback_module_content(module_name):
    """Provide fallback content if AI generation fails"""
    if module_name == "Budgeting Basics":
        return {
            "module_title": "Budgeting Basics",
            "module_description": "Learn how to create and maintain a personal budget, track expenses, and allocate funds effectively.",
            "completed": False,
            "lessons": [
                {
                    "lesson_id": 1,
                    "title": "Creating Your First Budget",
                    "content": "A budget is a financial plan that helps you track income and expenses. Start by listing all sources of income and categorizing your expenses into fixed and variable costs. Fixed costs include rent, utilities, and loan payments, while variable costs include groceries, entertainment, and dining out. Aim to allocate your income using the 50/30/20 rule: 50% for needs, 30% for wants, and 20% for savings and debt repayment.",
                    "completed": False,
                    "key_points": ["List all income sources", "Categorize expenses", "Use the 50/30/20 rule", "Review and adjust regularly"]
                },
                {
                    "lesson_id": 2,
                    "title": "Tracking Your Expenses",
                    "content": "Consistent expense tracking is crucial for maintaining an effective budget. Use digital tools like budgeting apps, spreadsheets, or even a simple notebook to record every expense. Categorize each expense and review your spending patterns weekly. Look for areas where you might be overspending and identify opportunities to cut back. Remember that small expenses add up over time, so tracking even minor purchases is important.",
                    "completed": False,
                    "key_points": ["Record every expense", "Use digital tools", "Review weekly", "Watch for patterns"]
                },
                {
                    "lesson_id": 3,
                    "title": "Adjusting Your Budget",
                    "content": "A budget is not set in stone—it should evolve with your financial situation. Review your budget monthly and make necessary adjustments based on changes in income or expenses. If you consistently overspend in certain categories, either increase the allocation if possible or find ways to reduce spending. As your income grows, prioritize increasing your savings rate rather than lifestyle inflation. Develop a system for handling unexpected expenses through an emergency fund.",
                    "completed": False,
                    "key_points": ["Monthly reviews", "Adjust allocations as needed", "Prioritize saving over spending", "Plan for emergencies"]
                }
            ],
            "quiz": {
                "quiz_id": 1,
                "title": "Quiz on Budgeting Basics",
                "completed": False,
                "questions": [
                    {
                        "question_id": 1,
                        "question": "What is the recommended allocation for needs in the 50/30/20 budgeting rule?",
                        "options": ["30%", "20%", "50%", "40%"],
                        "correct_answer": 2,
                        "explanation": "The 50/30/20 rule suggests allocating 50% of your income to needs, 30% to wants, and 20% to savings and debt repayment."
                    },
                    {
                        "question_id": 2,
                        "question": "Which of these is NOT typically considered a fixed expense?",
                        "options": ["Rent", "Grocery shopping", "Car loan payment", "Insurance premium"],
                        "correct_answer": 1,
                        "explanation": "Grocery shopping is typically a variable expense because the amount spent can vary significantly from month to month."
                    },
                    {
                        "question_id": 3,
                        "question": "How often should you review and adjust your budget?",
                        "options": ["Daily", "Weekly", "Monthly", "Yearly"],
                        "correct_answer": 2,
                        "explanation": "A monthly review is recommended to ensure your budget stays relevant to your current financial situation."
                    },
                    {
                        "question_id": 4,
                        "question": "What should you prioritize as your income increases?",
                        "options": ["Spending more on luxury items", "Increasing your savings rate", "Taking on more debt", "Maintaining the same lifestyle"],
                        "correct_answer": 1,
                        "explanation": "As your income grows, prioritize increasing your savings rate rather than increasing your spending (lifestyle inflation)."
                    },
                    {
                        "question_id": 5,
                        "question": "Why is tracking small expenses important?",
                        "options": ["It's not important", "They add up over time", "Only large expenses matter", "To impress others"],
                        "correct_answer": 1,
                        "explanation": "Small expenses add up significantly over time and can have a major impact on your overall financial health."
                    }
                ]
            }
        }
    else:
        # Generic structure for other modules
        return {
            "module_title": module_name,
            "module_description": get_module_description(module_name),
            "completed": False,
            "lessons": [
                {
                    "lesson_id": 1,
                    "title": f"Introduction to {module_name}",
                    "content": f"This lesson introduces you to the fundamentals of {module_name}. Understanding these concepts will help you make better financial decisions and improve your overall financial health. We'll cover the basic principles, common terminology, and practical applications in your daily life.",
                    "completed": False,
                    "key_points": ["Understanding the basics", "Learning key terminology", "Practical applications"]
                },
                {
                    "lesson_id": 2,
                    "title": f"Intermediate {module_name} Concepts",
                    "content": f"Building on the basics, this lesson explores more detailed aspects of {module_name}. You'll learn about strategies to optimize your approach, common pitfalls to avoid, and how to assess what works best for your personal financial situation.",
                    "completed": False,
                    "key_points": ["Advanced strategies", "Avoiding common mistakes", "Personalization techniques"]
                },
                {
                    "lesson_id": 3,
                    "title": f"Mastering {module_name}",
                    "content": f"The final lesson covers advanced concepts in {module_name}. You'll discover how experts approach these topics, learn about tools and resources to help you excel, and understand how to integrate these concepts with other aspects of financial management.",
                    "completed": False,
                    "key_points": ["Expert techniques", "Useful tools and resources", "Integration with other financial aspects"]
                }
            ],
            "quiz": {
                "quiz_id": 1,
                "title": f"Quiz on {module_name}",
                "completed": False,
                "questions": [
                    {
                        "question_id": 1,
                        "question": f"What is the primary purpose of {module_name}?",
                        "options": ["To increase debt", "To improve financial health", "To avoid financial planning", "To increase spending"],
                        "correct_answer": 1,
                        "explanation": f"The primary purpose of {module_name} is to improve your overall financial health through better understanding and management."
                    },
                    {
                        "question_id": 2,
                        "question": "Which of these is a recommended financial practice?",
                        "options": ["Ignoring your finances", "Only focusing on short-term goals", "Regular reviewing and planning", "Spending more than you earn"],
                        "correct_answer": 2,
                        "explanation": "Regular reviewing and planning is essential for maintaining good financial health."
                    },
                    {
                        "question_id": 3,
                        "question": "What role does education play in financial literacy?",
                        "options": ["It's unnecessary", "It's helpful but optional", "It's fundamental to making good decisions", "It only matters for financial professionals"],
                        "correct_answer": 2,
                        "explanation": "Education is fundamental to making good financial decisions, regardless of your income level or financial goals."
                    },
                    {
                        "question_id": 4,
                        "question": "How often should you reassess your financial strategies?",
                        "options": ["Never", "Only when in financial trouble", "Regularly", "Once in a lifetime"],
                        "correct_answer": 2,
                        "explanation": "Regular reassessment helps ensure your financial strategies remain aligned with your goals as your life circumstances change."
                    },
                    {
                        "question_id": 5,
                        "question": "What is the benefit of learning about financial concepts?",
                        "options": ["It makes you worry more", "It helps you make better financial decisions", "It has no practical benefit", "It's only useful for the wealthy"],
                        "correct_answer": 1,
                        "explanation": "Learning about financial concepts helps everyone, regardless of income level, make better financial decisions that can improve their quality of life."
                    }
                ]
            }
        }

if __name__ == '__main__':  
    app.run(debug=True)
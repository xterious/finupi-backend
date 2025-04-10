#!/usr/bin/env python
"""
FinUPI Credit Score API

A Flask API that exposes the credit score model as a web service.
Now with Firebase Firestore integration for the FinUPI frontend.
"""

import os
import json
import tempfile
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from credit_score_model import FinUPICreditScoreModel, process_upi_data
import base64
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the finupi frontend
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize the credit score model
model = FinUPICreditScoreModel()

# Initialize Firebase if credentials are available
firebase_app = None
db = None

def init_firebase():
    global firebase_app, db
    try:
        # Check if Firebase credentials file exists
        cred_path = os.environ.get('FIREBASE_CREDENTIALS_PATH', 'firebase-credentials.json')
        print(f"Looking for Firebase credentials at: {os.path.abspath(cred_path)}")
        
        # For development, we can use the Firebase config from environment variables
        if not os.path.exists(cred_path):
            print(f"Firebase credentials file not found at {os.path.abspath(cred_path)}")
            print("Attempting to use environment variables instead...")
            # Create credentials file from environment variables
            firebase_config = {
                "type": "service_account",
                "project_id": os.environ.get('REACT_APP_FIREBASE_PROJECT_ID', 'finupi-452b4'),
                "private_key_id": os.environ.get('FIREBASE_PRIVATE_KEY_ID', ''),
                "private_key": os.environ.get('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
                "client_email": os.environ.get('FIREBASE_CLIENT_EMAIL', ''),
                "client_id": os.environ.get('FIREBASE_CLIENT_ID', ''),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.environ.get('FIREBASE_CLIENT_CERT_URL', '')
            }
            
            # If we don't have private key, try using application default credentials
            if not firebase_config["private_key"]:
                print("No private key found in environment variables, trying application default credentials...")
                firebase_app = firebase_admin.initialize_app()
            else:
                print("Using environment variables for Firebase credentials")
                creds = credentials.Certificate(firebase_config)
                firebase_app = firebase_admin.initialize_app(creds)
        else:
            print(f"Found Firebase credentials file at {os.path.abspath(cred_path)}")
            try:
                creds = credentials.Certificate(cred_path)
                firebase_app = firebase_admin.initialize_app(creds)
            except Exception as cred_error:
                print(f"Error with Firebase credentials file: {str(cred_error)}")
                raise
        
        # Get Firestore database instance
        db = firestore.client()
        print("Firebase initialized successfully")
        return True
    except Exception as e:
        print(f"Firebase initialization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Try to initialize Firebase
firebase_initialized = init_firebase()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({
        'status': 'healthy',
        'model': 'FinUPI Credit Score v1.0',
        'firebase': 'connected' if firebase_initialized else 'not connected'
    })

@app.route('/api/credit-score', methods=['POST'])
def calculate_credit_score():
    """
    Calculate credit score from transaction data.
    
    Expected JSON payload:
    {
        "transactions": [
            {
                "id": "tx001",
                "date": "2023-07-01 08:30:00",
                "merchant": "Customer1",
                "amount": 450,
                "type": "credit"
            },
            ...
        ],
        "userId": "user123" // Optional user ID for Firebase storage
    }
    """
    try:
        data = request.json
        
        if not data or 'transactions' not in data:
            return jsonify({
                'error': 'Missing transaction data',
                'status': 'error'
            }), 400
            
        transactions = data['transactions']
        user_id = data.get('userId')
        
        if not transactions or not isinstance(transactions, list):
            return jsonify({
                'error': 'Invalid transaction data format',
                'status': 'error'
            }), 400
            
        # Calculate credit score without generating visual reports
        results = model.calculate_credit_score(transactions)
        
        # Store in Firebase if user ID is provided and Firebase is initialized
        if user_id and firebase_initialized and db:
            try:
                # Format the credit score data for Firestore
                credit_score_data = {
                    'score': results['score'],
                    'level': results['level'],
                    'components': results['components'],
                    'loan_eligibility': results['loan_eligibility'],
                    'last_updated': datetime.now().isoformat()
                }
                
                # Update user's credit score in Firestore
                db.collection('users').document(user_id).set({
                    'creditScore': credit_score_data
                }, merge=True)
                
                print(f"Credit score updated in Firestore for user {user_id}")
            except Exception as firebase_error:
                print(f"Firebase update error: {str(firebase_error)}")
                # Continue even if Firebase update fails
        
        return jsonify({
            'status': 'success',
            'credit_score': results,
            'firebase_updated': bool(user_id and firebase_initialized and db)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/upload-transactions', methods=['POST'])
def upload_transactions():
    """
    Upload transaction data file and calculate credit score.
    
    Expects a file upload with the name 'transaction_file'.
    """
    try:
        # Check if the post request has the file part
        if 'transaction_file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'status': 'error'
            }), 400
            
        file = request.files['transaction_file']
        
        # If user does not select file
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 'error'
            }), 400
            
        # Check file extension
        if not (file.filename.endswith('.json') or file.filename.endswith('.csv') or 
                file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            return jsonify({
                'error': 'Unsupported file format. Please upload JSON, CSV, or Excel file',
                'status': 'error'
            }), 400
            
        # Save uploaded file to a temporary location
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        
        # Process the file
        try:
            results = process_upi_data(file_path)
            
            # Get transaction count
            transaction_count = 0
            if file.filename.endswith('.json'):
                with open(file_path, 'r') as f:
                    transaction_data = json.load(f)
                    transaction_count = len(transaction_data)
            elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                transaction_data = pd.read_excel(file_path)
                transaction_count = len(transaction_data)
            elif file.filename.endswith('.csv'):
                transaction_data = pd.read_csv(file_path)
                transaction_count = len(transaction_data)
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully processed {transaction_count} transactions',
                'transactionCount': transaction_count,
                'creditScore': results
            })
            
        finally:
            # Clean up temporary file
            try:
                os.remove(file_path)
                os.rmdir(temp_dir)
            except:
                pass
                
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/separate-transactions', methods=['POST'])
def separate_transactions():
    """
    Separate transactions by sender/receiver and provide aggregated analysis.
    
    Expects a file upload with the name 'transaction_file'.
    """
    try:
        # Check if the post request has the file part
        if 'transaction_file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'status': 'error'
            }), 400
            
        file = request.files['transaction_file']
        
        # If user does not select file
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 'error'
            }), 400
            
        # Check file extension
        if not (file.filename.endswith('.json') or file.filename.endswith('.csv') or 
                file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            return jsonify({
                'error': 'Unsupported file format. Please upload JSON, CSV, or Excel file',
                'status': 'error'
            }), 400
            
        # Save uploaded file to a temporary location
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        
        # Process the file
        try:
            # Load the transaction data
            if file.filename.endswith('.csv'):
                transaction_data = pd.read_csv(file_path)
            elif file.filename.endswith('.json'):
                with open(file_path, 'r') as f:
                    transaction_data = json.load(f)
            elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                transaction_data = pd.read_excel(file_path)
            
            # Create model and separate transactions
            model = FinUPICreditScoreModel()
            separated = model.separate_transactions_by_type(transaction_data)
            
            # Convert dataframes to dictionary for JSON response
            response_data = {
                'sent_transactions': separated['sent'].to_dict(orient='records') if len(separated['sent']) > 0 else [],
                'received_transactions': separated['received'].to_dict(orient='records') if len(separated['received']) > 0 else [],
                'transaction_count': {
                    'total': len(separated['sent']) + len(separated['received']),
                    'sent': len(separated['sent']),
                    'received': len(separated['received'])
                },
                'unique_receivers': list(separated['by_receiver'].keys())
            }
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully separated {response_data["transaction_count"]["total"]} transactions',
                'data': response_data
            })
            
        finally:
            # Clean up temporary file
            try:
                os.remove(file_path)
                os.rmdir(temp_dir)
            except:
                pass
                
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/aggregate-by-receiver', methods=['POST'])
def aggregate_by_receiver():
    """
    Aggregate transactions by receiver and provide detailed statistics.
    
    Expects a file upload with the name 'transaction_file'.
    """
    try:
        # Check if the post request has the file part
        if 'transaction_file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'status': 'error'
            }), 400
            
        file = request.files['transaction_file']
        
        # If user does not select file
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 'error'
            }), 400
            
        # Check file extension
        if not (file.filename.endswith('.json') or file.filename.endswith('.csv') or 
                file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            return jsonify({
                'error': 'Unsupported file format. Please upload JSON, CSV, or Excel file',
                'status': 'error'
            }), 400
            
        # Save uploaded file to a temporary location
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        
        # Process the file
        try:
            # Load the transaction data
            if file.filename.endswith('.csv'):
                transaction_data = pd.read_csv(file_path)
            elif file.filename.endswith('.json'):
                with open(file_path, 'r') as f:
                    transaction_data = json.load(f)
            elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                transaction_data = pd.read_excel(file_path)
            
            # Create model and aggregate transactions
            model = FinUPICreditScoreModel()
            aggregated = model.aggregate_transactions_by_receiver(transaction_data)
            
            # Convert dataframe to dictionary for JSON response
            # Handle datetime objects which are not JSON serializable
            aggregated_dict = aggregated.reset_index().to_dict(orient='records')
            for record in aggregated_dict:
                record['first_transaction'] = record['first_transaction'].strftime('%Y-%m-%d %H:%M:%S')
                record['last_transaction'] = record['last_transaction'].strftime('%Y-%m-%d %H:%M:%S')
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully aggregated transactions for {len(aggregated)} receivers',
                'data': aggregated_dict
            })
            
        finally:
            # Clean up temporary file
            try:
                os.remove(file_path)
                os.rmdir(temp_dir)
            except:
                pass
                
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """
    Generate a comprehensive credit score report with visualizations.
    
    Expects a file upload with the name 'transaction_file'.
    """
    try:
        data = request.json
        
        if not data or 'transactions' not in data:
            return jsonify({
                'error': 'Missing transaction data',
                'status': 'error'
            }), 400
            
        transactions = data['transactions']
        
        # Create temp directory for report files
        temp_dir = tempfile.mkdtemp()
        report_file = os.path.join(temp_dir, 'credit_report.png')
        
        # Generate report with visualizations
        results = model.generate_report(transactions, report_file)
        
        excel_file = report_file.replace('.png', '.xlsx')
        
        # Return Excel file as attachment
        if request.args.get('format') == 'excel' and os.path.exists(excel_file):
            return send_file(
                excel_file,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='finupi_credit_report.xlsx'
            )
            
        # Return PNG file as attachment
        elif os.path.exists(report_file):
            return send_file(
                report_file,
                mimetype='image/png',
                as_attachment=True, 
                download_name='finupi_credit_report.png'
            )
            
        else:
            return jsonify({
                'error': 'Failed to generate report',
                'status': 'error'
            }), 500
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/process-transactions', methods=['POST'])
def process_transactions():
    """
    Process direct JSON transactions and calculate credit score.
    
    Expects a JSON payload with a list of transactions.
    Format of each transaction:
    {
        "Transaction ID": "TXN001",
        "Timestamp": 45757.385671296295,  # Excel date serial format
        "Sender Name": "Ramesh",
        "Sender UPI ID": "ramesh@upi",
        "Receiver Name": "Kirana Store",
        "Receiver UPI ID": "kirana123@upi",
        "Amount (INR)": 180,
        "Status": "SUCCESS"  # Optional
    }
    """
    try:
        # Get JSON data
        transaction_data = request.json
        
        # Check if the data is a list
        if not isinstance(transaction_data, list):
            return jsonify({
                'error': 'Invalid format. Expected a list of transaction objects',
                'status': 'error'
            }), 400
            
        # Check if we have any data
        if len(transaction_data) == 0:
            return jsonify({
                'error': 'Empty transaction list',
                'status': 'error'
            }), 400
            
        # Convert Excel serial dates to datetime if needed
        for tx in transaction_data:
            if 'Timestamp' in tx and isinstance(tx['Timestamp'], (int, float)):
                # Convert Excel serial date to datetime
                # Excel dates are days since 1900-01-01, with 1900-01-01 = 1
                # But Python's datetime counts from 1900-01-01 = 0, so we subtract 1
                # Also, Excel incorrectly treats 1900 as a leap year, so for dates after Feb 28, 1900,
                # we need to subtract another day
                excel_date = tx['Timestamp']
                # If the excel date is greater than 60 (1900-02-29), adjust for Excel's leap year bug
                if excel_date > 60:
                    excel_date -= 1
                    
                # Convert to datetime
                # Excel dates are days since 1899-12-30
                days_since_epoch = int(excel_date)
                seconds_in_day = (excel_date - days_since_epoch) * 86400  # 24*60*60 seconds in a day
                
                from datetime import datetime, timedelta
                epoch = datetime(1899, 12, 30)
                tx_date = epoch + timedelta(days=days_since_epoch, seconds=seconds_in_day)
                
                # Replace the timestamp with the datetime string
                tx['Timestamp'] = tx_date.strftime('%Y-%m-%d %H:%M:%S')
            
        # Create model and generate results
        model = FinUPICreditScoreModel()
        results = model.calculate_credit_score(transaction_data)
            
        return jsonify({
            'status': 'success',
            'message': f'Successfully processed {len(transaction_data)} transactions',
            'transactionCount': len(transaction_data),
            'creditScore': results
        })
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/process-and-report', methods=['POST'])
def process_and_report():
    """
    Process direct JSON transactions and generate a full credit report.
    
    Expects the same format as /api/process-transactions but returns
    a detailed report with visualizations.
    """
    try:
        # Get JSON data
        transaction_data = request.json
        
        # Check if the data is a list
        if not isinstance(transaction_data, list):
            return jsonify({
                'error': 'Invalid format. Expected a list of transaction objects',
                'status': 'error'
            }), 400
            
        # Check if we have any data
        if len(transaction_data) == 0:
            return jsonify({
                'error': 'Empty transaction list',
                'status': 'error'
            }), 400
            
        # Convert Excel serial dates to datetime if needed
        for tx in transaction_data:
            if 'Timestamp' in tx and isinstance(tx['Timestamp'], (int, float)):
                # Convert Excel serial date to datetime
                excel_date = tx['Timestamp']
                # If the excel date is greater than 60 (1900-02-29), adjust for Excel's leap year bug
                if excel_date > 60:
                    excel_date -= 1
                    
                # Convert to datetime
                # Excel dates are days since 1899-12-30
                days_since_epoch = int(excel_date)
                seconds_in_day = (excel_date - days_since_epoch) * 86400  # 24*60*60 seconds in a day
                
                from datetime import datetime, timedelta
                epoch = datetime(1899, 12, 30)
                tx_date = epoch + timedelta(days=days_since_epoch, seconds=seconds_in_day)
                
                # Replace the timestamp with the datetime string
                tx['Timestamp'] = tx_date.strftime('%Y-%m-%d %H:%M:%S')
            
        # Generate temporary file paths for the report
        temp_dir = tempfile.mkdtemp()
        report_file = os.path.join(temp_dir, 'credit_report.png')
        
        # Create model and generate report
        model = FinUPICreditScoreModel()
        results = model.generate_report(transaction_data, report_file)
        
        # If the report file was generated, return it
        if 'report_file' in results and os.path.exists(results['report_file']):
            # Get the Excel report if it exists
            excel_report = None
            if 'excel_report' in results and os.path.exists(results['excel_report']):
                excel_file = results['excel_report']
                excel_data = None
                with open(excel_file, 'rb') as f:
                    excel_data = f.read()
                
                # Clean up Excel file
                try:
                    os.remove(excel_file)
                except:
                    pass
                
                # Add Excel data to response
                if excel_data:
                    excel_report = base64.b64encode(excel_data).decode('utf-8')
            
            # Read the PNG report file
            with open(results['report_file'], 'rb') as f:
                image_data = f.read()
                
            # Remove the temporary files
            try:
                os.remove(results['report_file'])
                os.rmdir(temp_dir)
            except:
                pass
                
            # Include the report image as base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            
            # Return the credit score results and the base64 encoded report image
            return jsonify({
                'status': 'success',
                'message': f'Successfully processed {len(transaction_data)} transactions',
                'transactionCount': len(transaction_data),
                'creditScore': results,
                'reportImage': encoded_image,
                'excelReport': excel_report
            })
        else:
            # Return just the credit score results
            return jsonify({
                'status': 'success',
                'message': f'Successfully processed {len(transaction_data)} transactions',
                'transactionCount': len(transaction_data),
                'creditScore': results
            })
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/user-credit-score/<user_id>', methods=['GET'])
def get_user_credit_score(user_id):
    """
    Get the credit score for a specific user from Firebase.
    """
    if not firebase_initialized or not db:
        return jsonify({
            'error': 'Firebase not initialized',
            'status': 'error'
        }), 500
        
    try:
        # Retrieve user document from Firestore
        user_doc = db.collection('users').document(user_id).get()
        
        if not user_doc.exists:
            return jsonify({
                'error': 'User not found',
                'status': 'error'
            }), 404
            
        user_data = user_doc.to_dict()
        
        # Check if user has credit score data
        if 'creditScore' not in user_data:
            return jsonify({
                'error': 'No credit score data found for this user',
                'status': 'error'
            }), 404
            
        return jsonify({
            'status': 'success',
            'credit_score': user_data['creditScore']
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/transactions-upload-with-user/<user_id>', methods=['POST'])
def upload_transactions_with_user(user_id):
    """
    Upload transaction data file, calculate credit score, and save to Firebase for a specific user.
    """
    try:
        # Check if the post request has the file part
        if 'transaction_file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'status': 'error'
            }), 400
            
        file = request.files['transaction_file']
        
        # If user does not select file
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 'error'
            }), 400
            
        # Check file extension
        if not (file.filename.endswith('.json') or file.filename.endswith('.csv') or 
                file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            return jsonify({
                'error': 'Unsupported file format. Please upload JSON, CSV, or Excel file',
                'status': 'error'
            }), 400
            
        # Save uploaded file to a temporary location
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        
        # Process the file
        try:
            results = process_upi_data(file_path)
            
            # Get transaction count
            transaction_count = 0
            if file.filename.endswith('.json'):
                with open(file_path, 'r') as f:
                    transaction_data = json.load(f)
                    transaction_count = len(transaction_data)
            elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                transaction_data = pd.read_excel(file_path)
                transaction_count = len(transaction_data)
            elif file.filename.endswith('.csv'):
                transaction_data = pd.read_csv(file_path)
                transaction_count = len(transaction_data)
            
            # Store in Firebase if Firebase is initialized
            if firebase_initialized and db:
                try:
                    # Format the credit score data for Firestore
                    credit_score_data = {
                        'score': results['score'],
                        'level': results['level'],
                        'components': results['components'],
                        'loan_eligibility': results['loan_eligibility'],
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    # Update user's credit score in Firestore
                    db.collection('users').document(user_id).set({
                        'creditScore': credit_score_data
                    }, merge=True)
                    
                    print(f"Credit score updated in Firestore for user {user_id}")
                except Exception as firebase_error:
                    print(f"Firebase update error: {str(firebase_error)}")
                    # Continue even if Firebase update fails
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully processed {transaction_count} transactions',
                'transactionCount': transaction_count,
                'creditScore': results,
                'firebase_updated': firebase_initialized and db is not None
            })
            
        finally:
            # Clean up temporary file
            try:
                os.remove(file_path)
                os.rmdir(temp_dir)
            except:
                pass
                
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Enable debug mode in development
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    # Start the server
    app.run(host='0.0.0.0', port=port, debug=debug)
    print(f"FinUPI Credit Score API is running on port {port}") 
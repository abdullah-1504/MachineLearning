import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load model and supporting files
try:
    rf_pipeline = joblib.load("rf_pipeline.joblib")
    feature_order = joblib.load("model_featur9es.joblib")
    explainer = joblib.load("shap_explainer.joblib")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure all model files are in the current directory")

def to_scalar(value):
    """Convert various data types to scalar values"""
    if isinstance(value, (pd.Series, pd.Index)):
        return value.iloc[0] if hasattr(value, 'iloc') else value[0]
    elif isinstance(value, (np.ndarray, list)):
        return value[0]
    else:
        return value

def get_plain_english_explanation(feature, value):
    """Convert technical features to userfriendly explanations"""
    value = to_scalar(value)
    
    explanations = {
        "isOrigNeg": "The sender's account would go into negative balance",
        "isOrigZero": "The sender's account started with zero balance",
        "isDestNeg": "The receiver's account would go negative",
        "isDestZero": "The receiver's account started with zero balance",
        "type_CASH_OUT": "This is a cash withdrawal transaction (higher risk)",
        "type_TRANSFER": "This is a standard transfer between accounts",
        "errorBalanceOrig": "There's a balance calculation error for the sender",
        "errorBalanceDest": "There's a balance calculation error for the receiver"
    }
    
    if feature in explanations and value == 1:
        return explanations[feature]
    elif feature == "amt_perc_orig":
        if value > 0.9:
            return f"Almost all ({value:.0%}) of the sender's balance is being transferred."
        elif value > 0.5:
            return f"A large portion of the sender's balance ({value:.0%}) is being transferred."
    elif feature == "amount":
        if value > 1_000_000:
            return f"The transaction amount (${value:,.0f}) is extremely high."
        elif value > 100_000:
            return f"The transaction amount (${value:,.0f}) is quite high."
    elif feature == "step":
        return f"This transaction was performed very early in the transaction sequence (Step {value})."
    elif feature == "newbalanceOrig":
        if value < 0:
            return "After this transaction, the sender's account would be overdrawn (negative balance)."
        elif value == 0:
            return "After this transaction, the sender's account balance would be zero."
    elif feature == "oldbalanceDest":
        if value < 1000:
            return "The receiver's account had a low balance before this transaction."
        elif value > 100_000:
            return "The receiver's account had a very high balance before this transaction."
    
    # For all "is" features, explain zero as "normal"
    if feature.startswith("is") and value == 0:
        if feature == "isOrigZero":
            return "The sender's account had a balance before this transaction."
        elif feature == "isOrigNeg":
            return "The sender's account did not go negative after this transaction."
        elif feature == "isDestZero":
            return "The receiver's account had a balance before this transaction."
        elif feature == "isDestNeg":
            return "The receiver's account did not go negative after this transaction."
    
    # Last fallback: convert raw variable to simple English
    
    label = feature.replace('_', ' ').capitalize()
    return f"{label}: {value}"    

def analyze_transaction(step, tx_type, amount, sender_balance, receiver_balance):
    """Main function to analyze transaction for fraud"""
    
    if not tx_type:
        return "⚠️ **Please select a transaction type**"
    
    if amount <= 0:
        return "⚠️ **Please enter a valid transaction amount**"
    
    try:
        # Calculate new balances
        new_sender_balance = sender_balance - amount
        new_receiver_balance = receiver_balance + amount
        
        # Create feature dictionary
        features = {
            'step': step,
            'amount': amount,
            'oldbalanceOrg': sender_balance,
            'newbalanceOrig': new_sender_balance,
            'oldbalanceDest': receiver_balance,
            'newbalanceDest': new_receiver_balance,
            'errorBalanceOrig': sender_balance - amount - new_sender_balance,
            'errorBalanceDest': receiver_balance + amount - new_receiver_balance,
            'isOrigZero': int(sender_balance == 0),
            'isDestZero': int(receiver_balance == 0),
            'isOrigNeg': int(new_sender_balance < 0),
            'isDestNeg': int(new_receiver_balance < 0),
            'amt_perc_orig': amount / (sender_balance + 1),
            'amt_perc_dest': amount / (receiver_balance + 1),
            'type_TRANSFER': int(tx_type == 'TRANSFER'),
            'type_CASH_OUT': int(tx_type == 'CASH_OUT'),
        }
        
        # Ensure all required features are present
        for col in feature_order:
            if col not in features:
                features[col] = 0
        
        # Create input DataFrame
        X_input = pd.DataFrame([features])[feature_order]
        
        # Get prediction
        fraud_probability = rf_pipeline.predict_proba(X_input)[0, 1]
        is_fraud = fraud_probability > 0.5
        
        # Get explanations
        explanations = get_fraud_explanations(features, fraud_probability, X_input)
        
        # Format results
        return format_results(is_fraud, fraud_probability, explanations, amount)
        
    except Exception as e:
        return f"❌ **Error during analysis**: {str(e)}\n\nPlease check your inputs and try again."

def get_fraud_explanations(features, probability, X_input):
    """Get explanations for the fraud prediction"""
    explanations = []
    
    try:
        # Try SHAP explanation
        shap_values = explainer.shap_values(X_input)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_row = shap_values[1]  # Positive class
            else:
                shap_row = shap_values[0]
        else:
            shap_row = shap_values
        
        if shap_row.ndim > 1:
            shap_row = shap_row[0]
        
        shap_row = np.array(shap_row).flatten()
        
        # Get top 3 most important features
        top_indices = np.argsort(np.abs(shap_row))[-3:][::-1]
        feature_names = list(X_input.columns)
        
        for idx in top_indices:
            if idx < len(feature_names) and abs(shap_row[idx]) > 1e-6:
                feature = feature_names[idx]
                value = X_input.iloc[0, idx]
                explanations.append(get_plain_english_explanation(feature, value))
                
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        # Fallback to rule-based explanations
        explanations = get_fallback_explanations(features)
    
    return explanations if explanations else ["Model detected suspicious patterns in the transaction"]

def get_fallback_explanations(features):
    """Fallback explanations when SHAP fails"""
    explanations = []
    
    if features['isOrigNeg'] == 1:
        explanations.append("Sender's account would go negative after this transaction")
    
    if features['amt_perc_orig'] > 0.9:
        explanations.append("Almost all of the sender's balance is being transferred")
    
    if features['amount'] > 1_000_000:
        explanations.append("Transaction amount is unusually high")
    
    if features['type_CASH_OUT'] == 1:
        explanations.append("Cash withdrawal transactions have higher fraud risk")
    
    if features['isOrigZero'] == 1:
        explanations.append("Sender's account has zero balance")
    
    if abs(features['errorBalanceOrig']) > 1e-3:
        explanations.append("Balance calculation error detected for sender")
    
    return explanations[:3]  # Return top 3

def format_results(is_fraud, probability, explanations, amount):
    """Format the results in a user-friendly way"""
    
    # Header with emoji and clear verdict
    if is_fraud:
        header = f"🚨 **FRAUD DETECTED** (Risk: {probability:.1%})"
        risk_level = "HIGH RISK"
        color_indicator = "🔴"
    else:
        header = f"✅ **TRANSACTION APPEARS SAFE** (Risk: {probability:.1%})"
        risk_level = "LOW RISK"
        color_indicator = "🟢"
    
    # Risk assessment
    if probability > 0.8:
        risk_desc = "VERY HIGH - Immediate review required"
    elif probability > 0.6:
        risk_desc = "HIGH - Manual verification recommended"
    elif probability > 0.4:
        risk_desc = "MODERATE - Additional monitoring suggested"
    elif probability > 0.2:
        risk_desc = "LOW - Normal processing acceptable"
    else:
        risk_desc = "VERY LOW - Standard transaction"
    
    # Build result string
    result = f"{header}\n\n"
    result += f"{color_indicator} **Risk Level:** {risk_level} ({risk_desc})\n"
    result += f"💰 **Transaction Amount:** ${amount:,.2f}\n\n"
    
    # Add explanations
    if explanations:
        if is_fraud:
            result += "**🔍 Why this might be fraudulent:**\n"
        else:
            result += "**✨ Why this appears legitimate:**\n"
        
        for i, explanation in enumerate(explanations, 1):
            result += f"{i}. {explanation}\n"
    
    result += "\n"
    
    # Add recommendations
    if is_fraud:
        result += "**⚠️ Recommended Actions:**\n"
        result += "• Flag transaction for manual review\n"
        result += "• Verify customer identity\n"
        result += "• Check transaction history\n"
        result += "• Consider temporary account restrictions\n"
    else:
        result += "**📋 Recommended Actions:**\n"
        result += "• Process transaction normally\n"
        result += "• Continue standard monitoring\n"
        result += "• No additional verification needed\n"
    
    return result

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="Fraud Detection System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .fraud-title {
            text-align: center;
            color: #1e40af;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        # Title and description
        gr.Markdown(
            """
            # 🛡️ Financial Fraud Detection System
            
            **Advanced AI-powered transaction analysis for fraud prevention**
            
            Enter transaction details below to assess fraud risk using machine learning algorithms.
            The system analyzes multiple factors including account balances, transaction patterns, and historical data.
            """,
            elem_classes=["fraud-title"]
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Transaction Information")
                
                step_input = gr.Number(
                    label="Time Period (Step)",
                    value=1,
                    info="When this transaction occurred (time step)",
                    minimum=1
                )
                
                type_input = gr.Dropdown(
                    choices=["TRANSFER", "CASH_OUT"],
                    label="Transaction Type",
                    info="Select the type of financial transaction"
                )
                
                amount_input = gr.Number(
                    label="Transaction Amount ($)",
                    value=0,
                    info="Amount of money being transferred",
                    minimum=0
                )
                
                sender_balance_input = gr.Number(
                    label="Sender's Current Balance ($)",
                    value=0,
                    info="Available balance in sender's account before transaction",
                    minimum=0
                )
                
                receiver_balance_input = gr.Number(
                    label="Receiver's Current Balance ($)",
                    value=0,
                    info="Available balance in receiver's account before transaction",
                    minimum=0
                )
                
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Analyze Transaction",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear Form",
                        variant="secondary"
                    )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Results")
                
                output = gr.Markdown(
                    """
                    **Ready for Analysis**
                    
                    Fill in the transaction details on the left and click "Analyze Transaction" to begin fraud detection analysis.
                    
                    The system will evaluate:
                    • Account balance patterns
                    • Transaction amount relative to account size
                    • Transaction type risk factors
                    • Historical fraud indicators
                    """,
                    elem_id="results-output"
                )
        
        # Button actions
        analyze_btn.click(
            fn=analyze_transaction,
            inputs=[step_input, type_input, amount_input, sender_balance_input, receiver_balance_input],
            outputs=output
        )
        
        def clear_form():
            return 1, None, 0, 0, 0, "**Ready for Analysis**\n\nForm cleared. Enter new transaction details to analyze."
        
        clear_btn.click(
            fn=clear_form,
            outputs=[step_input, type_input, amount_input, sender_balance_input, receiver_balance_input, output]
        )
        
        # Footer
#        gr.Markdown(
 #           """
  #          ---
  #          **Note:** This system is for demonstration purposes. In production environments, 
   #         additional security measures and human oversight should always be implemented.
    #        """,
     #       elem_classes=["footer-text"]
      #  )
    
    return interface

# Launch the application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_port=7860,
        show_error=True
    )

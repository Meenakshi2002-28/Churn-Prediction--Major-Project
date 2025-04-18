from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import psycopg2
from psycopg2.extras import DictCursor
import bcrypt
import joblib
import numpy as np
import pandas as pd
from flask import make_response
from flask import Response  


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Keep this consistent

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        dbname="churn_prediction",
        user="postgres",
        password="gowri",
        host="localhost",
        port="5432"
    )
    return conn

# Routes
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('signup_form'))

@app.route('/signup', methods=['GET', 'POST'])
def signup_form():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        organization = request.form['organization']  # New organization field
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'error')
            return render_template('signup.html', 
                                   name=name,
                                   username=username,
                                   email=email,
                                   organization=organization)

        # Hash the password
        hashed_password = bcrypt.hashpw(
            password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')  # Store as string in DB

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # Check for existing username
            cur.execute("SELECT * FROM login WHERE username = %s", (username,))
            if cur.fetchone():
                flash('Username already exists. Please choose another one.', 'error')
                return redirect(url_for('signup_form'))
            
            # Insert new user with organization
            cur.execute(
                "INSERT INTO login (name, username, email, organization, password) VALUES (%s, %s, %s, %s, %s)",
                (name, username, email, organization, hashed_password)
            )
            conn.commit()
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
        finally:
            cur.close()
            conn.close()

    return render_template('signup.html', name='', username='', email='', organization='')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        try:
            cur.execute("SELECT * FROM login WHERE username = %s", (username,))
            user = cur.fetchone()
            
            if user and bcrypt.checkpw(
                password.encode('utf-8'),
                user['password'].encode('utf-8')
            ):
                # Store user data in session
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['name'] = user['name']
                session['organization'] = user['organization']  # Add organization to session
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'error')
                return redirect(url_for('login'))
        finally:
            cur.close()
            conn.close()
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Render with no-cache headers
    response = make_response(render_template(
        'dashboard.html',
        name=session['name'],
        organization=session['organization']
    ))
    
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response

@app.route('/bulk_upload', methods=['GET', 'POST'])
def bulk_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in request', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        try:
            df = pd.read_csv(file)

            REQUIRED_COLUMNS = [
                'age', 'location', 'subscription_type', 'payment_plan', 'payment_method',
                'num_subscription_pauses', 'weekly_hours', 'average_session_length',
                'song_skip_rate', 'weekly_songs_played', 'weekly_unique_songs',
                'notifications_clicked', 'customer_service_inquiries', 'engagement_score',
                'skip_rate_per_session'
            ]

            # ✅ Check if required columns are present
            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                flash(f"Missing columns in uploaded file: {', '.join(missing)}", 'error')
                return redirect(request.url)

            # ✅ Convert signup_date to days_since_signup
            today = datetime.today()
            df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
            df['days_since_signup'] = (today - df['signup_date']).dt.days

            # ✅ Drop original signup_date
            df.drop(columns=['signup_date'], inplace=True)

            # ✅ One-hot encode categorical features
            categorical_features = ['location', 'subscription_type', 'payment_plan', 'payment_method']
            df_categorical = df[categorical_features]
            df_categorical = df_categorical.reindex(columns=onehot_encoder.feature_names_in_, fill_value="Unknown")
            df_encoded = pd.DataFrame(onehot_encoder.transform(df_categorical).toarray(),
                                      columns=onehot_encoder.get_feature_names_out())
            df_encoded = df_encoded.reindex(columns=onehot_encoder.get_feature_names_out(), fill_value=0)

            # ✅ Ordinal encode
            df[['customer_service_inquiries']] = ordinal_encoder.transform(df[['customer_service_inquiries']])

            # ✅ Drop original categorical
            df.drop(columns=categorical_features, inplace=True)

            # ✅ Combine encoded
            df = pd.concat([df, df_encoded], axis=1)

            # ✅ Reindex to match scaler input
            df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

            # ✅ Scale numeric data
            df_scaled = scaler.transform(df)
            df_scaled = pd.DataFrame(df_scaled, columns=scaler.feature_names_in_)

            # ✅ Rename to match model
            df_scaled.columns = model.feature_name_

            # ✅ Predict
            predictions = model.predict(df_scaled)
            df['Prediction'] = ['Churn' if p == 1 else 'Not Churn' for p in predictions]

            # ✅ Convert final result to CSV
            response = make_response(df.to_csv(index=False))
            response.headers['Content-Disposition'] = 'attachment; filename=bulk_predictions.csv'
            response.mimetype = 'text/csv'
            return response

        except Exception as e:
            print("❌ Error in bulk upload:", str(e))
            flash(f"Error processing file: {str(e)}", 'error')
            return redirect(request.url)

    return render_template('bulk_upload.html')


REQUIRED_COLUMNS = [
    'age', 'location', 'subscription_type', 'payment_plan', 'payment_method',
    'num_subscription_pauses', 'weekly_hours', 'average_session_length',
    'song_skip_rate', 'weekly_songs_played', 'weekly_unique_songs',
    'notifications_clicked', 'customer_service_inquiries',
    'engagement_score', 'skip_rate_per_session'
]

@app.route('/download_template')
def download_template():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    return Response(
        df.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=template.csv"}
    )

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

with open('predict.pkl', 'rb') as f:
    models = joblib.load(f)

model = models.get('lightgbm')

if model is None:
    print("❌ Error: LightGBM model not found in predict.pkl")
else:
    print("✅ LightGBM model loaded successfully with features:", model.feature_name_)


with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = joblib.load(f)

with open('ordinal_encoder.pkl', 'rb') as f:
    ordinal_encoder = joblib.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

from datetime import datetime


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input data
        input_data = {
            'age': request.form['age'],
            'location': request.form['location'],
            'subscription_type': request.form['subscription_type'],
            'payment_plan': request.form['payment_plan'],
            'payment_method': request.form['payment_method'],
            'num_subscription_pauses': request.form['num_subscription_pauses'],
            'weekly_hours': request.form['weekly_hours'],
            'average_session_length': request.form['average_session_length'],
            'song_skip_rate': request.form['song_skip_rate'],
            'weekly_songs_played': request.form['weekly_songs_played'],
            'weekly_unique_songs': request.form['weekly_unique_songs'],
            'notifications_clicked': request.form['notifications_clicked'],
            'customer_service_inquiries': request.form['customer_service_inquiries'],
            'engagement_score': request.form['engagement_score'],
            'skip_rate_per_session': request.form['skip_rate_per_session'],
            'signup_date': request.form['signup_date']  # Get signup date
        }

        # Convert signup date to days_since_signup
        today = datetime.today()
        signup_date = datetime.strptime(input_data['signup_date'], "%Y-%m-%d")  # Assuming format 'YYYY-MM-DD'
        input_data['days_since_signup'] = (today - signup_date).days  # Convert to integer

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Debugging: Print the received DataFrame columns
        print("Received DataFrame columns:", df.columns.tolist())

        # Define categorical features
        categorical_features = ['location', 'subscription_type', 'payment_plan', 'payment_method']

        # Check if categorical columns exist in DataFrame
        missing_cols = [col for col in categorical_features if col not in df.columns]
        if missing_cols:
            print("❌ Missing Categorical Columns:", missing_cols)
            return "Error: Missing required input fields", 400  # Return error message

        # One-hot encode categorical features (only if columns exist)
        df_categorical = df[categorical_features]

        # Ensure order matches fitted encoder
        df_categorical = df_categorical.reindex(columns=onehot_encoder.feature_names_in_, fill_value="Unknown")

        # Transform categorical features
        df_encoded = pd.DataFrame(onehot_encoder.transform(df_categorical).toarray(),
                          columns=onehot_encoder.get_feature_names_out())
        df_encoded = df_encoded.reindex(columns=onehot_encoder.get_feature_names_out(), fill_value=0)  


        # Encode ordinal feature
        if 'customer_service_inquiries' in df.columns:
            df[['customer_service_inquiries']] = ordinal_encoder.transform(df[['customer_service_inquiries']])
        else:
            print("❌ 'customer_service_inquiries' is missing!")

        # Drop original categorical columns
        df.drop(columns=categorical_features, errors='ignore', inplace=True)

        # Concatenate encoded categorical features
        df = pd.concat([df, df_encoded], axis=1)

        # Ensure feature order matches what was used during training
        df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # Scale numerical features
        df_scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=scaler.feature_names_in_)

        # Debugging: Check final feature names before renaming
        print("✅ Final Features Sent to Model:", df_scaled.columns.tolist())
        print("Processed DataFrame columns:", df_scaled.columns.tolist())
        print("Expected Model Columns:", model.feature_name_)

        # Rename processed DataFrame columns to match model expectations
        df_scaled.columns = model.feature_name_

        # Debugging: Confirm renaming is correct
        print("Renamed Processed DataFrame columns:", df_scaled.columns.tolist())

        # Make prediction
        prediction = model.predict(df_scaled)

        # Return result
        result = 'Churn' if prediction[0] == 1 else 'Not Churn'
        return render_template('prediction_result.html', prediction=result)
    
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
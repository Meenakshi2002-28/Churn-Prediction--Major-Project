from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import psycopg2
from psycopg2.extras import DictCursor
import bcrypt
import joblib
import numpy as np
import pandas as pd
from flask_mail import Mail, Message
from datetime import datetime
import re
from flask import make_response
from flask import Response  
from flask import Flask, render_template, request, flash, redirect, url_for, session
from psycopg2.extras import execute_values

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

import re  # make sure this is at the top

@app.route('/signup', methods=['GET', 'POST'])
def signup_form():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        organization = request.form['organization']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Name validation
        if not re.match(r'^[A-Za-z]+(?: [A-Za-z]+)*$', name):
            flash('Only alphabetic letters allowed', 'error')
            return render_template('signup.html', name=name, username=username, email=email, organization=organization)

        # Email validation (only @gmail.com allowed)
        if not re.match(r'^[\w\.-]+@gmail\.com$', email):
            flash('Enter a valid Gmail address (e.g., example@gmail.com).', 'error')
            return render_template('signup.html', name=name, username=username, email=email, organization=organization)

        # Password validation
        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*[\W_]).{8,}$', password):
            flash('Password must be at least 8 characters long and include 1 uppercase letter, 1 lowercase letter, and 1 special character.', 'error')
            return render_template('signup.html', name=name, username=username, email=email, organization=organization)

        # Confirm password match
        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'error')
            return render_template('signup.html', name=name, username=username, email=email, organization=organization)

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # Check for existing username
            cur.execute("SELECT * FROM login WHERE username = %s", (username,))
            if cur.fetchone():
                flash('Username already exists. Please choose another one.', 'error')
                return render_template('signup.html', name=name, username='', email=email, organization=organization)

            # Check for existing email
            cur.execute("SELECT * FROM login WHERE email = %s", (email,))
            if cur.fetchone():
                flash('Email already registered. Please use another one or log in.', 'error')
                return render_template('signup.html', name=name, username=username, email='', organization=organization)

            # Insert new user
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

# Initialize Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True  # Use TLS for secure connection
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'gparvathys04@gmail.com'  # Replace with your Gmail address
app.config['MAIL_PASSWORD'] = 'cvwq ftob aqdl gcem'  # Replace with your Gmail app password
app.config['MAIL_DEFAULT_SENDER'] = 'your_email@gmail.com'

mail = Mail(app)

# Forgot Password Route
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT * FROM login WHERE email = %s", (email,))
            user = cur.fetchone()
            
            if user:
                user_name = user[1]  # Assuming the 2nd column is the user's name

                # Generate a password reset link (you can add a secure token here)
                reset_link = url_for('reset_password', email=email, _external=True)

                # Create the email message
                msg = Message("Password Reset Request", recipients=[email])

                msg.html = f"""
                <p style="color: black;">Hi <strong>{user_name}</strong>,</p>
                <p style="color: black;">
                    We are sending you this email because you requested a password reset.
                </p>
                <p style="color: black;">
                    Click the button below to create a new password:
                </p>
                <p>
                    <a href="{reset_link}" style="background-color: #0000FF; color: white; padding: 10px 20px;
                        text-align: center; text-decoration: none; display: inline-block; border-radius: 8px; font-size: 14px; font-weight:550">
                       Set a New Password
                    </a>
                </p>
                <p style="color: black;">
                    If you did not request a password reset, you can safely ignore this email. Your password will not be changed.
                </p>
                """

                try:
                    mail.send(msg)
                    flash('Reset instructions have been sent to your email.', 'success')
                except Exception as e:
                    print("Email send failed:", e)
                    flash('Failed to send email. Please check your mail configuration.', 'error')
            else:
                flash('No account found with that email address.', 'error')
        finally:
            cur.close()
            conn.close()

        return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html')



import re  # Add at the top of your file if not already imported

# Reset Password Route
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        # Get email from hidden form input
        email = request.form.get('email')
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'error')
            return render_template('reset_password.html', email=email)

        # Password validation (without digit requirement)
        password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*[@$!%*?&])[A-Za-z@$!%*?&]{8,}$'
        if not re.match(password_pattern, password):
            flash('Password must be at least 8 characters, with 1 uppercase, 1 lowercase, and 1 special character.', 'error')
            return render_template('reset_password.html', email=email)

        # Hash the new password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Update the password in the database
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("UPDATE login SET password = %s WHERE email = %s", (hashed_password, email))
            conn.commit()
            flash('Password updated successfully!', 'success')
            return redirect(url_for('login'))
        finally:
            cur.close()
            conn.close()
    
    else:
        # For GET requests, get email from the URL
        email = request.args.get('email')
        if not email:
            flash("Invalid or missing email in reset link.", "error")
            return redirect(url_for('forgot_password'))
        return render_template('reset_password.html', email=email)


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

from psycopg2.extras import execute_values

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
            # 1) Read CSV
            df = pd.read_csv(file)

            REQUIRED_COLUMNS = [
                'age', 'location', 'subscription_type', 'payment_plan', 'payment_method',
                'num_subscription_pauses', 'weekly_hours', 'average_session_length',
                'song_skip_rate', 'weekly_songs_played', 'weekly_unique_songs',
                'notifications_clicked', 'customer_service_inquiries', 'engagement_score',
                'skip_rate_per_session', 'signup_date'
            ]
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                flash(f"Missing columns: {', '.join(missing)}", 'error')
                return redirect(request.url)

            # 2) Keep a copy for display + DB insert
            df_display = df.copy()

            # 3) Parse and reformat signup_date in df_display
            df_display['signup_date'] = pd.to_datetime(
                df_display['signup_date'], dayfirst=True, errors='coerce'
            )
            if df_display['signup_date'].isnull().any():
                flash("Some signup_date values could not be parsed.", 'error')
                return redirect(request.url)

            # Format as YYYY‑MM‑DD string for Postgres
            df_display['signup_date'] = df_display['signup_date'].dt.strftime('%Y-%m-%d')

            # 4) Calculate days_since_signup in a working df
            today = datetime.today()
            df['signup_date'] = pd.to_datetime(df_display['signup_date'])
            df['days_since_signup'] = (today - df['signup_date']).dt.days
            df.drop(columns=['signup_date'], inplace=True)

            # 5) One-hot & ordinal encode, scale, predict
            # (same as before)
            categorical_features = ['location', 'subscription_type', 'payment_plan', 'payment_method']
            df_cat = df[categorical_features]
            df_cat = df_cat.reindex(
                columns=onehot_encoder.feature_names_in_, fill_value="Unknown"
            )
            df_enc = pd.DataFrame(
                onehot_encoder.transform(df_cat).toarray(),
                columns=onehot_encoder.get_feature_names_out()
            ).reindex(columns=onehot_encoder.get_feature_names_out(), fill_value=0)

            df[['customer_service_inquiries']] = ordinal_encoder.transform(
                df[['customer_service_inquiries']]
            )
            df.drop(columns=categorical_features, inplace=True)
            df = pd.concat([df, df_enc], axis=1)
            df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
            df_scaled = scaler.transform(df)
            df_scaled = pd.DataFrame(df_scaled, columns=scaler.feature_names_in_)
            df_scaled.columns = model.feature_name_

            preds = model.predict(df_scaled)

            # 6) Attach predictions back to df_display
            df_display['prediction'] = ['Churn' if p==1 else 'Not Churn' for p in preds]

            # 7) Save in session for the results page
            session['prediction_results'] = df_display.to_json(orient='records')
            session['results_columns'] = df_display.columns.tolist()

            # 8) Get the file name and set source_type as 'dataset'
            dataset_name = file.filename
            source_type = 'dataset'
            upload_time = datetime.now()  # Upload timestamp

            # 9) Insert the data into the database
            columns_to_store = [
                'age', 'location', 'subscription_type', 'payment_plan', 'payment_method',
                'num_subscription_pauses', 'weekly_hours', 'average_session_length',
                'song_skip_rate', 'weekly_songs_played', 'weekly_unique_songs',
                'notifications_clicked', 'customer_service_inquiries', 'engagement_score',
                'skip_rate_per_session', 'signup_date', 'prediction', 'dataset_name', 'upload_time', 'source_type'
            ]
            df_display['dataset_name'] = dataset_name
            df_display['upload_time'] = upload_time
            df_display['source_type'] = source_type

            df_to_store = df_display[columns_to_store]
            records = [tuple(row) for row in df_to_store.to_numpy()]

            insert_query = """
                INSERT INTO data (
                  age, location, subscription_type, payment_plan, payment_method,
                  num_subscription_pauses, weekly_hours, average_session_length,
                  song_skip_rate, weekly_songs_played, weekly_unique_songs,
                  notifications_clicked, customer_service_inquiries, engagement_score,
                  skip_rate_per_session, signup_date, prediction,
                  dataset_name, upload_time, source_type
                ) VALUES %s
            """

            conn = get_db_connection()
            cur = conn.cursor()
            try:
                print(f"Attempting to insert {len(records)} records…")
                execute_values(cur, insert_query, records)
                conn.commit()
                print("✅ Insert successful!")
            except Exception as db_err:
                print("❌ DB insert error:", db_err)
                flash("Error saving to database", 'error')
            finally:
                cur.close()
                conn.close()

            # 10) Redirect to results
            return redirect(url_for('prediction_results'))

        except Exception as e:
            print("❌ Error in bulk_upload:", e)
            flash(f"Processing error: {e}", 'error')
            return redirect(request.url)

    return render_template('bulk_upload.html')





@app.route('/download_results')
def download_results():
    if 'prediction_results' not in session:
        flash('No results to download', 'error')
        return redirect(url_for('bulk_upload'))
    
    df = pd.read_json(session['prediction_results'], orient='records')
    
    response = make_response(df.to_csv(index=False))
    response.headers['Content-Disposition'] = 'attachment; filename=churn_predictions.csv'
    response.mimetype = 'text/csv'
    return response


REQUIRED_COLUMNS = [
    'age', 'location', 'subscription_type', 'payment_plan', 'payment_method',
    'num_subscription_pauses', 'weekly_hours', 'average_session_length',
    'song_skip_rate', 'weekly_songs_played', 'weekly_unique_songs',
    'notifications_clicked', 'customer_service_inquiries',
    'engagement_score', 'skip_rate_per_session'
]

@app.route('/prediction_results')
def prediction_results():
    if 'prediction_results' not in session:
        flash('No prediction results found', 'error')
        return redirect(url_for('bulk_upload'))
    
    try:
        # Get results from session
        results_json = session.get('prediction_results')
        columns = session.get('results_columns', [])
        
        # Convert back to DataFrame
        df = pd.read_json(results_json, orient='records')
        
        # Convert to HTML
        results_html = df.to_html(
            classes='table table-striped table-bordered',
            index=False
        )
        
        return render_template('prediction_results.html', 
                            results_table=results_html,
                            num_results=len(df))
        
    except Exception as e:
        flash('Error displaying results', 'error')
        return redirect(url_for('bulk_upload'))

@app.route('/download_template')
def download_template():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    return Response(
        df.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=template.csv"}
    )

@app.route('/past_predictions')
def past_predictions():
    conn = get_db_connection()
    cur = conn.cursor()

    # Modify the query to return unique dataset_name and upload_time combinations
    cur.execute("""
        SELECT DISTINCT dataset_name, upload_time, source_type 
        FROM data 
        ORDER BY upload_time DESC
    """)  # Using DISTINCT to fetch unique combinations of dataset_name and upload_time
    past_predictions = cur.fetchall()
    cur.close()
    conn.close()

    return render_template('past_predictions.html', predictions=past_predictions)

@app.route('/view_results/<dataset_name>')
def view_results(dataset_name):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Exclude dataset_name, upload_time, source_type from the SELECT
    cur.execute("""
        SELECT 
            age, location, subscription_type, payment_plan, payment_method,
            num_subscription_pauses, weekly_hours, average_session_length,
            song_skip_rate, weekly_songs_played, weekly_unique_songs,
            notifications_clicked, customer_service_inquiries, engagement_score,
            skip_rate_per_session, signup_date, prediction
        FROM data
        WHERE dataset_name = %s
    """, (dataset_name,))
    
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]

    cur.close()
    conn.close()

    return render_template("view_results.html", dataset_name=dataset_name, rows=rows, columns=colnames)

@app.route('/delete_dataset', methods=['POST'])
def delete_dataset():
    dataset_name = request.form.get('dataset_name')
    upload_time = request.form.get('upload_time')

    if not dataset_name or not upload_time:
        flash('Missing dataset name or upload time', 'error')
        return redirect(url_for('past_predictions'))

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            DELETE FROM data 
            WHERE dataset_name = %s AND upload_time = %s
        """, (dataset_name, upload_time))
        conn.commit()
        flash(f"Dataset '{dataset_name}' uploaded at '{upload_time}' deleted successfully!", 'success')
    except Exception as e:
        print("❌ Error deleting dataset:", e)
        flash("Error deleting dataset", 'error')
    finally:
        cur.close()
        conn.close()

    return redirect(url_for('past_predictions'))


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
from flask import Flask, render_template, request, redirect, url_for, flash, session
import psycopg2
from psycopg2.extras import DictCursor
import bcrypt

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

    return render_template('dashboard.html', name=session['name'], organization=session['organization'])

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

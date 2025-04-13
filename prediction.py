from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('final 3.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve data from the form
        age = request.form['age']
        location = request.form['location']
        subscription_type = request.form['subscription_type']
        payment_plan = request.form['payment_plan']
        payment_method = request.form['payment_method']
        num_subscription_pauses = request.form['num_subscription_pauses']
        signup_date = request.form['signup_date']
        weekly_hours = request.form['weekly_hours']
        average_session_length = request.form['average_session_length']
        song_skip_rate = request.form['song_skip_rate']
        weekly_songs_played = request.form['weekly_songs_played']
        weekly_unique_songs = request.form['weekly_unique_songs']
        notifications_clicked = request.form['notifications_clicked']
        customer_service_inquiries = request.form['customer_service_inquiries']
        engagement_score = request.form['engagement_score']
        skip_rate_per_session = request.form['skip_rate_per_session']

        # Process input data (convert values to appropriate formats for the model)
        input_data = [
            age, location, subscription_type, payment_plan, payment_method,
            num_subscription_pauses, signup_date, weekly_hours, average_session_length,
            song_skip_rate, weekly_songs_played, weekly_unique_songs, notifications_clicked,
            customer_service_inquiries, engagement_score, skip_rate_per_session
        ]

        # Prepare the input data for model prediction (you might need to do preprocessing here)
        # Example: You may need to convert categorical variables to numerical encoding

        # Make the prediction
        prediction = model.predict([input_data])

        # Return the prediction result to the user
        return render_template('prediction_result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

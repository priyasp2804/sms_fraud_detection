from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the form
    message = request.form['message']
    
    # Transform the message to match the model's input format
    message_tfidf = vectorizer.transform([message])
    
    # Make prediction
    prediction = model.predict(message_tfidf)[0]
    
    # Return the result
    result = "Spam" if prediction == 0 else "Not Spam"
    return render_template('index.html', prediction_text=f'This message is: {result}')

if __name__ == "__main__":
    app.run(debug=True)

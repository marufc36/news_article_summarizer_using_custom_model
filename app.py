from flask import Flask, render_template, request
from gradio_client import Client

app = Flask(__name__)

# Initialize Gradio client
client = Client("mmchowdhury/News_Summary_With_Custom_Model")

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        # Get input text from the form
        input_text = request.form["input_text"]
        
        # Make prediction using the Gradio client
        result = client.predict(input_text, api_name="/predict")
        
        # Check if result is a string
        if isinstance(result, str):
            summary = result
        else:
            # Extract the summary from the result
            summary = result["data"]["output_text"]
        
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
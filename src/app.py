from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

@app.route("/",methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        req = request.get_json()
        # Check mandatory key
        if "input" in req.keys():
            # Load model
            classifier = joblib.load("models/model.joblib")
            # Predict
            #wine_vars=[]
            #for ilist in req['input']:
            #    wine_vars.append([float(val) for val in ilist])

            prediction = classifier.predict(req['input'])
           #print("Input:",req["input"])
            # Return the result as JSON but first we need to transform the
            # result so as to be serializable by jsonify()
            #prediction = [str(prediction[0]),str(prediction[1])]
            #prediction="Hello"
            return jsonify([str(pred) for pred in prediction]), 200      
    return jsonify({"msg": "Error: not a JSON or no input key in your request"})




if __name__ == "__main__":
    app.run(debug=True)

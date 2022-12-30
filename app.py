# Importing Libraries
from main import *
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# Initializing the Flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():

    result = user_input(request.form["category"],
                        request.form["amt"],
                        request.form["zip"],
                        request.form["lat"],
                        request.form["long"],
                        request.form["city_pop"],
                        request.form["merch_lat"],
                        request.form["merch_long"],
                        request.form["age"],
                        request.form["hour"],
                        request.form["day"],
                        request.form["month"])

    trans_num = request.form["trans_num"]

    return render_template("home.html")

    # if int(result) == 1:
    #     return render_template("home.html", prediction=f"The transaction number {trans_num} is Fraud!")
    # else:
    #     return render_template("home.html", prediction=f"The transaction number {trans_num} is Genuine!")


if __name__ == '__main__':
    app.run(debug=True)

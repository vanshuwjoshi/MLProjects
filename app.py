# this fill will create a web application to fill out the details and get the predicted math score

from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


# Index page
@app.route("/")
def index():
    return render_template("index.html")


# Prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    # for GET method show the home page (without any predicted score)
    if request.method == "GET":
        return render_template("home.html")
    else:  # after submit button POST request is sent and we capture all the form data to predict the math score
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("writing_score")),
            writing_score=float(request.form.get("reading_score")),
        )
        # get the above captured data as a dataframe
        pred_df = data.get_data_as_frame()  # from predict_pipeline.py
        print(pred_df)

        # call to the PredictPipeline class in predict_pipeline.py
        predict_pipeline = PredictPipeline()
        # the below predict function will generate the predicted math score
        predictions = predict_pipeline.predict(pred_df)
        return render_template(
            "home.html", results=predictions[0]
        )  # show the predicted score in home page


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

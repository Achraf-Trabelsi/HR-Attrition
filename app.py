from flask import Flask, request
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

pickle_in = open("weights/HRattrition_model_RandomForest.pkl", "rb")
clf = pickle.load(pickle_in)
app = Flask(__name__)

@app.route('/')
def welcome():
    return 'this is my first flask app with HR attrition model'


@app.route('/predict_file', methods=['POST'])
def predict_on_test():
    df_test = pd.read_csv(request.files['file'])
    enc = OrdinalEncoder()
    df_test = enc.fit_transform(df_test)
    pred = clf.predict(df_test)
    return "The prediction is " + str(list(pred))


'''
@app.route("/", methods=["GET", "POST"])
def main():
    if fa.request.method == "GET":
        return fa.render_template("main.html")

    if fa.request.method == "POST":

        satisfaction_level = fa.request.form["satisfaction_level"]
        last_evaluation = fa.request.form["last_evaluation"]
        number_of_projects = fa.request.form["number_of_projects"]
        average_monthly_hours = fa.request.form["average_monthly_hours"]
        years_at_company = fa.request.form["years_at_company"]
        work_accident = fa.request.form["work_accident"]
        promotion_last_5years = fa.request.form["promotion_last_5years"]
        salary = fa.request.form["salary"]
        department = fa.request.form["department"]

        input_variables = pd.DataFrame(
            [
                [
                    satisfaction_level,
                    last_evaluation,
                    number_of_projects,
                    average_monthly_hours,
                    years_at_company,
                    work_accident,
                    promotion_last_5years,
                    department,
                    salary,
                ]
            ],
            columns=[
                "satisfaction_level",
                "last_evaluation",
                "number_of_projects",
                "average_monthly_hours",
                "years_at_company",
                "work_accident",
                "promotion_last_5years",
                "department",
                "salary",
            ],
            dtype=float,
        )
        prediction = model.predict(input_variables)[0]
        return f.render_template(
            "main.html",
            original_input={
                "satisfaction_level": satisfaction_level,
                "last_evaluation": last_evaluation,
                "number_of_projects": number_of_projects,
            },
            result=prediction,
        )

'''

if __name__ == "__main__":
    app.run(debug=True)

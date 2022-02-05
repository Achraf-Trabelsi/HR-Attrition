import flask as fa
import pickle 
import pandas as pd

with open(f'HRattrition_model_RandomForest.pkl', 'rb') as f:
    model = pickle.load(f) 

app = fa.Flask(__name__, template_folder='templates')

@app.route('/',methods=['GET', 'POST'])


def main():
    if fa.request.method == 'GET':
        return(fa.render_template('main.html'))

    if fa.request.method == 'POST':        
        
        satisfaction_level = fa.request.form['satisfaction_level']
        last_evaluation = fa.request.form['last_evaluation']
        number_of_projects = fa.request.form['number_of_projects']   
        average_monthly_hours= fa.request.form['average_monthly_hours']   
        years_at_company = fa.request.form['years_at_company']   
        work_accident = fa.request.form['work_accident']   
        promotion_last_5years = fa.request.form['promotion_last_5years']   
        salary = fa.request.form['salary']   
        department = fa.request.form['department']   
        

        input_variables = pd.DataFrame([[satisfaction_level, last_evaluation, number_of_projects,
        
                                        average_monthly_hours,years_at_company, work_accident,
                                        promotion_last_5years, department, salary]],
                                       columns=['satisfaction_level', 'last_evaluation',
                                        'number_of_projects','average_monthly_hours',
                                        'years_at_company','work_accident',
                                        'promotion_last_5years', 'department', 'salary'],
                                       dtype=float)        
        prediction = model.predict(input_variables)[0]        
        return f.render_template('main.html',
                                  original_input={'satisfaction_level':satisfaction_level,
                                                     'last_evaluation':last_evaluation,
                                                     'number_of_projects':number_of_projects},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run(debug=True)


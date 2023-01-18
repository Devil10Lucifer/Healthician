import pickle
import flask
import pandas as pd

with open('final_prediction.pkl','wb') as file:
    pickle.dump('final_prediction',file)

with open(f'model/final_prediction.pkl','rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__,template_folder='templates')

@app.route('/',methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':

        Symptom1 = flask.request.form['symptom1']
        Symptom2 = flask.request.form['symptom2']
        Symptom3 = flask.request.form['symptom3']

        input_variables = pd.DataFrame([[Symptom1,Symptom2,Symptom3]],columns=['symptom1','symptom2','symptom3'],dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',original_input={'Symptom1':Symptom1,'Symptom2':Symptom2,'Symptom3':Symptom3},result=prediction,)

if __name__ == '__main__':
    app.run()
    
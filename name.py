from flask import Flask, render_template, request
import pickle
import numpy as np
model  = open("model.pkl", "rb")
app = Flask(__name__)
@app.route('/')

def student():
    return render_template('student.html')

@app.route('/predicted',methods = ['POST', 'GET'])
def predict():
   if request.method == 'POST':
    results = [float(x) for x in request.form.values()]
    results.pop(2)
    final_var=[np.array(results)]
    pred=modal.predict(final_var)
    output=round(pred[0],2)
    if output<0.3:
        t="Chances of passing are very low:("
    else:
        t="Chances of passing are high"
    return render_template("student.html",prediction_text=t)


if __name__ == '__main__':
   print("running...")
   app.run(debug= True)
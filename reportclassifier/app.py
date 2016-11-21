from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
#from web

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
clf = pickle.load(
    open(os.path.join(cur_dir, 'pkl_objects/classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')


def classify(document):
    label = {0: 'single', 1: 'multiple'}
    x = vect.transform([document])
    y = clf.predict(x)[0]
    proba = np.max(clf.predict_proba(x))
    return label[y], proba


def train(document, y):
    x = vect.transform([document])
    clf.partial_fit(x, [y])


def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO reports_db (reports, category, date)"
              " VALUES(?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()


class ReportForm(Form):
    mrireport = TextAreaField('',
                              [validators.DataRequired(),
                               validators.length(min=60)])


@app.route('/')
def index():
    form = ReportForm(request.form)
    return render_template('reportform.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    form = ReportForm(request.form)
    if request.method == 'POST' and form.validate():
        report = request.form['mrireport']
        y, proba = classify(report)
        return render_template('results.html',
                               content=report,
                               prediction=y,
                               probability=round(proba * 100, 2))
    return render_template('reportform.html', form=form)


@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    report = request.form['report']
    prediction = request.form['prediction']

    inv_label = {'single': 0, 'multiple': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)

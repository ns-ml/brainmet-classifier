from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from sklearn.externals import joblib
import sqlite3
import numpy as np
import os
from vectorizer import tokenizer
import re

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
db = os.path.join(cur_dir, 'reviews.sqlite')
clf = joblib.load('pkl_objects/clf/logregression.pkl')

# def preprocessor(text):
#     text = re.sub('[\W]+', ' ', text.lower())
#     body_pattern = re.compile(r'findings (.*) (?=this report was)')
#     matched_text = body_pattern.search(text)

#     if matched_text is None:
#         body_pattern = re.compile(r'findings (.*) (?=radiologists signatures)')
#         matched_text = body_pattern.search(text)

#     if matched_text is None:
#         body_pattern = re.compile(r'comparison (.*) (?=radiologists signatures)')
#         matched_text = body_pattern.search(text)

#     if matched_text is None:
#         stripped_text = text
#     else:
#         stripped_text = matched_text.group(1).replace(
#             'i the teaching physician have reviewed the images and agree with the report as written',
#             '')
#     return stripped_text


def preprocessor(text):
    # Remove newlines
    text = text.replace(r'\n', '')

    # Remove date
    date_pattern = r'[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,}'
    text = re.sub(date_pattern, '', text.lower())

    # Remove whitespace
    text = ' '.join(text.split())

    # Remove punctution, keep decimal points
    text = re.sub(r'[\W]+(?!\d)', ' ', text)

    # Remove the signature at the end of the report

    if text.find(' i the teaching physician') != -1:
        body_pattern = re.compile(r'(.*) (?=i the teaching physician)')
        matched_text = body_pattern.search(text).group(1)

    elif text.find(' end of impression') != -1:
        body_pattern = re.compile(r'(.*) (?=end of impression)')
        matched_text = body_pattern.search(text).group(1)

    elif text.find(' radiologists signatures') != -1:
        body_pattern = re.compile(r'(.*) (?=radiologists signatures)')

        matched_text = body_pattern.search(text).group(1)

    else:
        matched_text = text

    return matched_text


def classify(document):
    text = preprocessor(document)
    label = {0: 'single', 1: 'multiple'}
    y = clf.predict([text])[0]
    proba = np.max(clf.predict_proba(text))
    return label[y], proba


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
        text = preprocessor(report)
        return render_template('results.html',
                               original_content=report,
                               clean_content=text,
                               prediction=y,
                               probability=round(proba * 100, 2))
    return render_template('reportform.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)

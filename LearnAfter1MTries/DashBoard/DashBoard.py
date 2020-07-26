# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd
import orjson


def ReadFromFolder(filePath):
    with open(filePath, 'rb') as log_file:
        readData = log_file.read()
        data = orjson.loads(readData)
    return data


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

path = '../TWEANN/Test_Tweann/Log/Log.log'





app.layout = html.Div(children=[
html.H1(children="Generation"),

html.Div(children='''
    Dash: A web application framework for Python.
'''),

dcc.Graph(
    id='GenData',

),
dcc.Dropdown(
        id='ChildDropdown',
        options=[
            {'label': 'BestEver', 'value': 'BestEver'},
            {'label': 'Best', 'value': 'Best'},
            {'label': 'Best and Best Ever', 'value': 'BestAndBestEver'},
            {'label': 'All', 'value': 'All'},

        ],
        value='All'
    ),
dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0)

])

@app.callback(
    Output('GenData', 'figure'),[Input('interval-component', 'n_intervals'),Input('ChildDropdown', 'value')])
def update_figure(n,childShow):
    y = ReadFromFolder(path)

    if childShow == 'BestEver':
        yShow =y[-2]
        xShow = [1]

        df = pd.DataFrame({
            "Child": xShow,
            "Fitness": yShow,

        })

        fig = px.bar(df, x="Child", y='Fitness')
    if childShow == 'Best':
        yShow =y[-1]
        xShow = [1]
        df = pd.DataFrame({
            "Child": xShow,
            "Fitness": yShow,

        })

        fig = px.bar(df, x="Child", y='Fitness')

    if childShow == 'BestAndBestEver':
        yShow = [y[-1], y[-2]]
        xShow = [1, 2]
        df = pd.DataFrame({
            "Child": xShow,
            "Fitness": yShow,

        })

        fig = px.bar(df, x="Child",y='Fitness')

    if childShow == 'All':
        yShow = y
        xShow = [x for x in range(1, len(y) + 1)]
        df = pd.DataFrame({
            "Child": xShow,
            "Fitness": yShow,

        })

        fig = px.histogram(df, x="Fitness")



    fig.update_layout(transition_duration=500)

    return fig



if __name__ == '__main__':
    app.run_server(debug=True)

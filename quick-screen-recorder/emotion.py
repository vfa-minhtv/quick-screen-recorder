import os
import pandas as pd
from dash import Dash, callback_context, no_update
from dash.dependencies import Input, Output
from dash_table import DataTable
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px

path = "bin/x64/Debug/output.csv"
df = pd.read_csv(path)
df = df.tail(150)
lastmt = os.stat(path).st_mtime
print(lastmt)

# fig = go.Figure()
# fig.add_trace(go.Scatter(df, x="date", y="happy", mode='lines', name='happy'))
# fig = px.line(df, x="date", y="happy")

fig = go.Figure()
# fig = fig.add_trace(go.Scatter(y=list(df["neutral"]), x=list(df.index), name="neutral"))
fig = fig.add_trace(go.Scatter(y=list(df["Happy"]), x=list(df.index), mode='lines+markers', name="Happy"))
fig.update_layout(showlegend=True, yaxis=dict(range=[0, 100]), legend = dict(font = dict(size = 30)))

app = Dash(__name__)
app.layout = html.Div(
    [
        # DataTable(
        #     id="table",
        #     columns=[{"name": i, "id": i} for i in df.columns],
        #     data=df.to_dict("records"),
        #     export_format="csv",
        # ),
        # dcc.Graph(
        #     id='example-graph',
        #     figure=fig
        # ),
        dcc.Graph(
          figure=fig,
          style={'height': 600},
          id='my-graph'
        ),
        dcc.Interval(id='interval', interval=1000, n_intervals=0)
    ]
)

@app.callback(Output('my-graph', 'figure'), [Input('interval', 'n_intervals')])
def trigger_by_modify(n):
    global lastmt
    if os.stat(path).st_mtime > lastmt:
        print("modified")
        lastmt = os.stat(path).st_mtime
        df = pd.read_csv(path)
        df = df.tail(150)

        fig = go.Figure()
        # fig = fig.add_trace(go.Scatter(y=list(df["neutral"]), x=list(df.index), name="neutral"))
        fig = fig.add_trace(go.Scatter(y=list(df["Happy"]), x=list(df.index), mode='lines+markers', name="happy"))
        fig.update_layout(transition_duration=100, yaxis=dict(range=[0, 100]), showlegend=True, legend = dict(font = dict(size = 30)))
        return fig

    return no_update

if __name__ == "__main__":
    app.run_server(debug=True)

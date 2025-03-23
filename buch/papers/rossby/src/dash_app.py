from dash import Dash, html, dcc, callback, Output, Input

def dash_app(fig):

    app = Dash()

    # Requires Dash 2.17.0 or later
    app.layout = [
        html.H1(children='Rossby Waves', style={'textAlign':'center'}),
        dcc.Graph(id='graph-content', figure=fig),
    ]
    return app

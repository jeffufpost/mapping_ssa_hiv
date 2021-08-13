# Navbar
from navbar import Navbar



##################################
##################################

def create_prob_series(xi, yi, yi5, yi10, title):
    return {
        'data': [
            {'name': 'Probability curve given current active cases',
             "x": xi,
             "y": yi,
             'type': 'scatter',
             'line': {'color':'royalblue', 'width':3}
            },
            {'name': 'Probability curve if active cases are underestimated by a factor of 5',
             "x": xi,
             "y": yi5,
             'type': 'scatter',
             'line': {'color':'firebrick', 'width':2, 'dash':'dot'}
            },
            {'name': 'Probability curve if active cases are underestimated by a factor of 10',
             "x": xi,
             "y": yi10,
             'type': 'scatter',
             'line': {'color':'black', 'width':2, 'dash':'dot'}
            }
        ],
        'layout': {
            'title': {'text': title},
            'xaxis': {'title': {'text': "Group size"}, 'type': 'log'},
            'yaxis': {'title': {'text': "Chance of at least 1 active case being present"}, 'type': 'log'}
        }
    }

def create_time_series(xc, yc, xd, yd, xr, yr, xi, yi, title):
    return {
        'data': [
            {'name': 'Active',
             "x": xi,
             "y": yi,
             'type': 'scatter',
             'line': {'color':'firebrick'}
            },
            {'name': 'Recoveries',
             "x": xr,
             "y": yr,
             'type': 'scatter',
             'line': {'color':'green'}
            },
            {'name': 'Deaths',
             "x": xd,
             "y": yd,
             'type': 'scatter',
             'line': {'color':'black'}
            },
            {'name': 'Cases since start of epidemic',
             "x": xc,
             "y": yc,
             'type': 'scatter',
             'line': {'color':'royalblue', 'width':4, 'dash':'dot'}
            }
        ],
        'layout': {
            #'height': 350,
            #'margin': {'l': 30, 'b': 30, 'r': 30, 't': 30},
            'title': {'text': title},
            'legend':{'traceorder': 'reversed'},
            'updatemenus': [{
                'active': 1,
                'buttons': [
                    {'args': [
                        {'visible': [True, True]},
                        {'yaxis': {'type': 'log'}}
                    ],
                    'label': 'Log Scale',
                    'method': 'update'
                    },

                    {'args': [
                        {'visible': [True, True]},
                        {'yaxis': {'type': 'linear'}}
                    ],
                     'label': 'Linear Scale',
                     'method': 'update'}
                ],
                'direction': 'down',
                'pad': {'r': 10, 't': 10},
                'showactive': True,
                'x': 0.1,
                'xanchor': 'left',
                'y': 1.1,
                'yanchor': 'top'
            }]
        }
    }

def create_bar_series(xc, yc, xd, yd, xr, yr, title):
    return {
        'data': [
            {'name': 'Recoveries',
             "x": xr,
             "y": yr,
             'marker': {'color':'green'},
             'type': 'bar'},
            {'name': 'Deaths',
             "x": xd,
             "y": yd,
             'marker': {'color':'black'},
             'type': 'bar'},
            {'name': 'New cases',
             "x": xc,
             "y": yc,
             'marker': {'color':'firebrick'},
             'type': 'bar',
             'barmode': 'group',}
        ],
        'layout': {
            #'height': 350,
            #'margin': {'l': 30, 'b': 30, 'r': 30, 't': 30},
            'title': {'text': title},
            'legend':{'traceorder': 'reversed'}
        }
    }

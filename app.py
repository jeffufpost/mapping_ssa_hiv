# Navbar
from navbar import Navbar



##################################
##################################

def create_time_series2(x, yrea, yrad, ydc, yhosp, ynum1, ynum2, title):
    return {
        'data': [
            {'name': 'Patients en Reanimation',
             "x": x,
             "y": yrea,
             'type': 'scatter',
             'line': {'color':'firebrick'}
            },
            {'name': 'Recoveries cumulative',
             "x": x,
             "y": yrad,
             'type': 'scatter',
             'visible':'legendonly',
             'line': {'color':'green'}
            },
            {'name': 'Deaths cumulative',
             "x": x,
             "y": ydc,
             'type': 'scatter',
             'visible':'legendonly',
             'line': {'color':'black'}
            },
            {'name': '# Lits réa + intensifs',
             "x": x,
             "y": ynum2,
             'type': 'scatter',
             'line': {'color':'darkmagenta', 'width':4, 'dash':'dot'}
            },
            {'name': '# Lits réa',
             "x": x,
             "y": ynum1,
             'type': 'scatter',
             'line': {'color':'firebrick', 'width':4, 'dash':'dot'}
            },
            {'name': 'Hospitalisations actuelles',
             "x": x,
             "y": yhosp,
             'type': 'scatter',
             'line': {'color':'royalblue'}
            }
        ],
        'layout': {
            #'height': 350,
            #'margin': {'l': 30, 'b': 30, 'r': 30, 't': 30},
            'title': {'text': title},
            'legend':{'traceorder': 'reversed'},
            'xaxis': {'rangeselector': {'buttons': [{'count': 7, 'label': '7D', 'step': 'day', 'stepmode': 'backward'},
                                            {'count': 14, 'label': '2W', 'step': 'day', 'stepmode': 'backward'},
                                            {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                            {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                            {'step': 'all'}]}
        }
    }
    }

def create_bar_series2(x, yc, yd, yr, yt, yh, yicu, title):
    return {
        'data': [
            {'name': 'Recoveries',
             "x": x,
             "y": yr,
             'visible':'legendonly',
             'marker': {'color':'green'},
             'type': 'bar'},
            {'name': 'Deaths',
             "x": x,
             "y": yd,
             'visible':'legendonly',
             'marker': {'color':'black'},
             'type': 'bar'},
            {'name': 'New cases - positive tests',
             "x": x,
             "y": yc,
             'marker': {'color':'firebrick'},
             'type': 'bar'},
            {'name': 'Total tests performed',
             "x": x,
             "y": yt,
             'visible':'legendonly',
             'marker': {'color':'blue'},
             'type': 'bar'},
            {'name': 'Hospitalizations',
             "x": x,
             "y": yh,
             'type': 'bar'},
            {'name': 'ICU admissions',
             "x": x,
             "y": yicu,
             'type': 'bar',
             'barmode': 'group',}
        ],
        'layout': {
            #'height': 350,
            #'margin': {'l': 30, 'b': 30, 'r': 30, 't': 30},
            'title': {'text': title},
            'legend':{'traceorder': 'reversed'},
            'xaxis': {'rangeselector': {'buttons': [{'count': 7, 'label': '7D', 'step': 'day', 'stepmode': 'backward'},
                                            {'count': 14, 'label': '2W', 'step': 'day', 'stepmode': 'backward'},
                                            {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                            {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                            {'step': 'all'}]}
        }
    }
    }

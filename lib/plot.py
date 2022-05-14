import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2)
import random
from ipywidgets import widgets
from IPython.display import display,HTML
from ipywidgets import HBox,VBox, Label

class DataPlotter:

    def __init__(self, input,output, param=None) -> None:
        self.input = input
        self.output = output
        self.param = param
    def plot(self):
        button = widgets.Button(description="Refresh")
        box_layout = widgets.Layout(display='flex',
                        flex_flow='column',
                        align_items='center',
                        width='24%')
        box = widgets.HBox(children=[button],layout=box_layout)


        if self.param == 'same':
            # plot only on one fig
            fig = go.FigureWidget(make_subplots(rows=1, cols=1))
            index = random.randint(0,len(self.input)-1)
            input = self.input[index]
            output = self.output[index]
            fig.add_trace(
                go.Scatter( y=input),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=output),
                row=1, col=1
            )
        else:
            fig = go.FigureWidget(make_subplots(rows=1, cols=2))
            index = random.randint(0,len(self.input)-1)
            input = self.input[index]
            output = self.output[index]
            fig.add_trace(
                go.Scatter( y=input),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=output),
                row=1, col=2
            )

        fig.update_layout(
            margin=dict(t=25),
        )
        output = widgets.Output()
        def response(b):     
            index = random.randint(0,len(self.input)-1)
            with fig.batch_update():
                fig.data[0].y = self.input[index]
                fig.data[1].y = self.output[index]
        button.on_click(response)
        return VBox([box,output,fig])
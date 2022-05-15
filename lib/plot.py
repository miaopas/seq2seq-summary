from numpy import SHIFT_DIVIDEBYZERO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from ipywidgets import widgets
from ipywidgets import VBox, HBox, Output, HTML
from math import exp
import numpy as np
from ml_collections import FrozenConfigDict
class DataPlotter:

    def __init__(self, data_name = None, num_of_plots=2) -> None:
        self.data_name = data_name
        self.num_of_plots = num_of_plots
        try:
            data_generator = getattr(self, f'_generate_data_{data_name.lower().strip()}')
        except:
            raise Exception('No such data name defined.')
        self.config = FrozenConfigDict({'shift': dict(LENGTH = 100,
                                    NUM = 20,
                                    SHIFT = 30)})
        self.input, self.output = data_generator()




    def _generate_gaussian(self, seq_length):
        
        def rbf_kernel(x1, x2, variance = 1):
            return exp(-1 * ((x1-x2) ** 2) / (2*variance))

        def gram_matrix(xs):
            return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]

        xs = np.arange(seq_length)*0.1
        mean = [0 for _ in xs]
        gram = gram_matrix(xs)
        ys = np.random.multivariate_normal(mean, gram)
        return ys

    def _generate_data_shift(self):
        # some parameters of the demo dataset
        

        input = []
        output = []
        for _ in range(self.config.shift.NUM):
            data = self._generate_gaussian(self.config.shift.LENGTH)
            input.append(data)
            output.append(np.concatenate((np.zeros(self.config.shift.SHIFT), data[:-self.config.shift.SHIFT])))
        return input, output

    def _get_descrip(self):
        if self.data_name == 'shift':
            return f'Shift the input to the right by {self.config.shift.SHIFT} timesteps.'
        else:
            return ''

    def _get_title(self):
        if self.data_name == 'shift':
            return f'Shift Seqeunce'
        else:
            return ''

    def plot(self):
        descrip = HTML(f' <font size="+1"><b>{self._get_title()}:</b> {self._get_descrip()} </font>')
        button = widgets.Button(description="Refresh")
        box_layout = widgets.Layout(display='flex',
                        flex_flow='column',
                        align_items='center',
                        width='24%')
        box = widgets.HBox(children=[button],layout=box_layout)

        # Having one plot or two
        if self.num_of_plots == 'same':
            second_col = 1
        else:
            second_col = 2
        
        plot_range = np.abs(np.array(self.input + self.output)).max()

        fig = go.FigureWidget(make_subplots(rows=1, cols=second_col, shared_yaxes=True,subplot_titles=("Input Sequence","Output Sequence")))
        index = random.randint(0,len(self.input)-1)
        input = self.input[index]
        output = self.output[index]

        fig.add_trace(
            go.Scatter(y=input, name='Input'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=output, name='Output'),
            row=1, col=second_col
        )    

        layout = go.Layout(
            margin=dict(t=25),
        )
        fig.update_yaxes(range=[-plot_range, plot_range], row=1,col=1)
        fig.update_yaxes(range=[-plot_range, plot_range], row=1,col=self.num_of_plots)
        fig.update_layout(layout)
        
        output = widgets.Output()
        def response(b):     
            index = random.randint(0,len(self.input)-1)
            with fig.batch_update():
                fig.data[0].y = self.input[index]
                fig.data[1].y = self.output[index]
        button.on_click(response)
        container = HBox([box, descrip])
        return VBox([container,output,fig])
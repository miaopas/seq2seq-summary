import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from ipywidgets import widgets
from ipywidgets import VBox, HBox, HTML
from math import exp
import numpy as np
from ml_collections import FrozenConfigDict
from pyparsing import ExceptionWordUnicode
from lib.demoGenerator import LorenzRandFGenerator
class DataPlotter:

    def __init__(self,input=None, output=None, data_name = None, num_of_plots=2) -> None:
        self.data_name = data_name
        self.num_of_plots = num_of_plots
        try:
            data_generator = getattr(self, f'_generate_data_{data_name.lower().strip()}')
        except:
            raise Exception('No such data name defined.')
        self.config = FrozenConfigDict({'shift': dict(LENGTH = 100,
                                    NUM = 20,
                                    SHIFT = 30),
                                    'lorentz': dict(NUM = 10, K=1, J=10, LENGTH=128 )})

        if input is None or output is None:
            self.input, self.output = data_generator()
        else:
            # If input and output data are provided.
            self.input, self.output = input, output

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
        input = []
        output = []
        for _ in range(self.config.shift.NUM):
            data = self._generate_gaussian(self.config.shift.LENGTH)
            input.append(data)
            output.append(np.concatenate((np.zeros(self.config.shift.SHIFT), data[:-self.config.shift.SHIFT])))
        return input, output

    def _generate_data_lorentz(self):
        def generate_inputs():
            input = []
            for _ in range(self.config.lorentz.NUM):
                data = self._generate_gaussian(self.config.shift.LENGTH)
                input.append(data[:,np.newaxis])
            return np.array(input)

        lorentz_generator = LorenzRandFGenerator({'n_init':self.config.lorentz.NUM, 
                                                    'K':self.config.lorentz.K, 
                                                    'J':self.config.lorentz.J,
                                                    'path_len':self.config.lorentz.LENGTH})
        lorentz_generator.generate_inputs = generate_inputs

        input, output = lorentz_generator.generate(scale=False)

        return input.squeeze(-1)[:,1:], output.squeeze(-1)[:,1:]

    def _get_descrip(self):
        if self.data_name == 'shift':
            return f'Shift the input to the right by {self.config.shift.SHIFT} timesteps.'
        elif self.data_name == 'lorentz':
            return f'The output is the response of input defined by the Lorentz96 system.'
        else:
            return ''

    def _get_title(self):
        if self.data_name == 'shift':
            return f'Shift Seqeunce'
        elif self.data_name == 'lorentz':
            return f'Lorentz System'
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

        # Having one plot or two, if have one plot then put input and output together.
        if self.num_of_plots == 1:
            second_col = 1
        elif self.num_of_plots == 2:
            second_col = 2
        else:
            raise NotImplementedError('Can only have 1 or 2 plots.')
        
        # Calculate range of plot.
        plot_range_in = np.abs(np.array(self.input)).max()+0.2
        plot_range_out = np.abs(np.array(self.output)).max()+0.2
        
        
        # Setup subplots, if only have 1 plots then input and output are together on col1.
        col1_name = 'Input Sequence' if self.num_of_plots == 2 else 'Intput/Output Sequences'
        fig = go.FigureWidget(make_subplots(rows=1, cols=second_col, shared_yaxes=False,subplot_titles=(col1_name,"Output Sequence")))
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

        fig.update_yaxes(range=[-plot_range_out, plot_range_out], row=1,col=self.num_of_plots)
        fig.update_yaxes(range=[-plot_range_in, plot_range_in], row=1,col=1)
        
        fig.update_layout(layout)
        
        if self.data_name == 'shift':
            # Add a rectangle to highlight the shifted interval
            x = self.config.shift.LENGTH-self.config.shift.SHIFT
            fig.add_shape(
                type='rect', xref='x', yref='y',
                x0=0, x1=x, y0=-plot_range_in+1, y1=plot_range_in-1, fillcolor="LightSkyBlue",opacity=0.3, line_color="LightSkyBlue"
            , row=1,col=1)
            fig.add_shape(
                type='rect', xref='x', yref='y',
                x0=self.config.shift.SHIFT, x1=self.config.shift.LENGTH, y0=-plot_range_out+1, y1=plot_range_out-1, fillcolor="LightSkyBlue",opacity=0.3, line_color="LightSkyBlue"
            , row=1,col=2)

        # Click event for button
        output = widgets.Output()
        def response(b):     
            index = random.randint(0,len(self.input)-1)
            with fig.batch_update():
                fig.data[0].y = self.input[index]
                fig.data[1].y = self.output[index]
        button.on_click(response)
        container = HBox([box, descrip])
        return VBox([container,output,fig])
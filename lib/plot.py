import imp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from ipywidgets import widgets
from ipywidgets import VBox, HBox, HTML
import numpy as np
import re
import ast
from ml_collections import FrozenConfigDict
from lib.demoGenerator import LorenzRandFGenerator
CONFIG = FrozenConfigDict({'shift': dict(LENGTH = 100,
                                                NUM = 20,
                                                SHIFT = 30),
                            'convo': dict(LENGTH = 100,
                                                NUM = 20,
                                                FILTER = [0.002, 0.022, 0.097, 0.159, 0.097, 0.022, 0.002]),
                                'lorentz': dict(NUM = 10, 
                                                K=1, J=10, 
                                                LENGTH=128 )})
class DataPlotter:

    """
    A class to create an interactive plot of sequential input, output pair. Have a button to randomly refresh different set of input, output pair.
    """

    def __init__(self,input, output, num_of_plots=2,title='', descrip='', subplot_title=()) -> None:

        """
        Args:
            input (_type_): The input have shape (num_of_data, length).
            output (_type_): The output have shape (num_of_data, length).
            num_of_plots (int, optional): The number of plots, if only have one then put input,output on same plot, otherwise put them side by side. Defaults to 2.
            title (str, optional): Title of the plot.
            descrip (str, optional): Description of the plot. 
            subplot_title (tuple, optional): A tuple consits of name of the subplots.
        """

        self.num_of_plots = num_of_plots
        self.input = input
        self.output = output
        self.descrip = descrip
        self.title = title
        self.subplot_title = subplot_title
        self.debug = widgets.Output() 
        if num_of_plots not in (1,2):
            raise Exception('Can only have 1 or 2 plots')
        

    def plot(self):
        descrip = HTML(f' <font size="+1"><b>{self.title}</b> {self.descrip} </font>')
        button = widgets.Button(description="Refresh")
        box_layout = widgets.Layout(display='flex',
                        flex_flow='column',
                        align_items='center',
                        width='24%')
        box = widgets.HBox(children=[button],layout=box_layout)

        # Calculate range of plot.
        plot_range_in = np.abs(np.array(self.input)).max()+0.2
        plot_range_out = np.abs(np.array(self.output)).max()+0.2
          
        # Setup subplots, if only have 1 plots then input and output are together on col1.
        self.fig = go.FigureWidget(make_subplots(rows=1, cols=self.num_of_plots, shared_yaxes=False,subplot_titles=self.subplot_title))
        
        # Randomly choose one input,output pair to plot.
        index = random.randint(0,len(self.input)-1)
        input, output = self.input[index], self.output[index] 

        self.fig.add_trace(
            go.Scatter(y=input, name='Input'),
            row=1, col=1
        )
        self.fig.add_trace(
            go.Scatter(y=output, name='Output'),
            row=1, col=self.num_of_plots
        )    
        layout = go.Layout(
            margin=dict(t=25),
        )
        self.fig.update_yaxes(range=[-plot_range_out, plot_range_out], row=1,col=self.num_of_plots)
        self.fig.update_yaxes(range=[-plot_range_in, plot_range_in], row=1,col=1)
        self.fig.update_layout(layout)
        

        # Click event for button
        
        button.on_click(self.response)
        container = HBox([box, descrip])
        return VBox([container,self.fig])

    def response(self, b):     
        index = random.randint(0,len(self.input)-1)
        with self.fig.batch_update():
            self.fig.data[0].y = self.input[index]
            self.fig.data[1].y = self.output[index]

class ShiftPlotter(DataPlotter):
    def __init__(self):
        input, output = self._generate_data()
        super().__init__(input=input, output=output, num_of_plots=2, 
                        title='Shift Seqeunce:',
                        descrip=f'Shift the input to the right by {CONFIG.shift.SHIFT} timesteps.',
                        subplot_title=('Input Sequence', 'Output Sequence'), )
    
    def _generate_gaussian(self, seq_length):
        def rbf_kernel(x1, x2, variance = 1):
            from math import exp
            return exp(-1 * ((x1-x2) ** 2) / (2*variance))
        def gram_matrix(xs):
            return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]
        xs = np.arange(seq_length)*0.1
        mean = [0 for _ in xs]
        gram = gram_matrix(xs)
        ys = np.random.multivariate_normal(mean, gram)
        return ys
    
    def plot(self):
        vbox = super().plot()
        fig = vbox.children[1]
        range = fig['layout'].yaxis.range[1]+0.2
        x = CONFIG.shift.LENGTH-CONFIG.shift.SHIFT
        fig.add_shape(
            type='rect', xref='x', yref='y',
            x0=0, x1=x, y0=-range+1, y1=range-1, fillcolor="LightSkyBlue",opacity=0.3, line_color="LightSkyBlue"
        , row=1,col=1)
        fig.add_shape(
            type='rect', xref='x', yref='y',
            x0=CONFIG.shift.SHIFT, x1=CONFIG.shift.LENGTH, y0=-range+1, y1=range-1, fillcolor="LightSkyBlue",opacity=0.3, line_color="LightSkyBlue"
        , row=1,col=2)
        return vbox

    def _generate_data(self):
        input = []
        output = []
        for _ in range(CONFIG.shift.NUM):
            data = self._generate_gaussian(CONFIG.shift.LENGTH)
            input.append(data)
            output.append(np.concatenate((np.zeros(CONFIG.shift.SHIFT), data[:-CONFIG.shift.SHIFT])))
        return input, output

class LorentzPlotter(DataPlotter):
    def __init__(self):
        input, output = self._generate_data()
        super().__init__(input=input, output=output, num_of_plots=1, 
                        title=f'Lorentz System:',
                        descrip=f'The output is the response of input defined by the Lorentz96 system.',
                        subplot_title=('Intput/Output Sequences',''), )
    
    def _generate_data(self):
        lorentz_generator = LorenzRandFGenerator({'n_init':CONFIG.lorentz.NUM, 
                                                    'K':CONFIG.lorentz.K, 
                                                    'J':CONFIG.lorentz.J,
                                                    'path_len':CONFIG.lorentz.LENGTH})

        input, output = lorentz_generator.generate(scale=False)

        return input.squeeze(-1)[:,1:], output.squeeze(-1)[:,1:]



class ConvoPlotter(DataPlotter):
    def __init__(self):
        self.filter = CONFIG.convo.FILTER
        input, output = self._generate_data()
        super().__init__(input=input, output=output, num_of_plots=2, 
                        title='&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Convolution of sequence with a filter',
                        descrip=f'',
                        subplot_title=('Input Sequence', 'Output Sequence'), )
    
    def _generate_gaussian(self, seq_length):
        def rbf_kernel(x1, x2, variance = 1):
            from math import exp
            return exp(-1 * ((x1-x2) ** 2) / (2*variance))
        def gram_matrix(xs):
            return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]
        xs = np.arange(seq_length)*0.1
        mean = [0 for _ in xs]
        gram = gram_matrix(xs)
        ys = np.random.multivariate_normal(mean, gram)
        return ys
    
    def plot(self):
        vbox = super().plot()
        hbox = vbox.children[0]
        range = self.fig['layout'].yaxis.range[1]+0.2
        self.filter_box = widgets.Text(
                        value=str(CONFIG.convo.FILTER)[1:-1],
                        placeholder='Enter a filter',
                        description='Filter:',
                        disabled=False,
                        layout = widgets.Layout(width='400px')
                    )
        button =  hbox.children[0].children[0]
        descp = hbox.children[1]
        new_hbox = HBox([self.filter_box,button, descp])
        new_vbox = VBox([new_hbox, self.fig,self.debug])
        return new_vbox

    def _generate_data(self):
        input = []
        output = []
        for _ in range(CONFIG.convo.NUM):
            data = self._generate_gaussian(CONFIG.convo.LENGTH)
            data = 2*np.random.random(CONFIG.convo.LENGTH)-1
            input.append(data)
            output.append(np.convolve(data, self.filter, mode='same'))
        return input, output

    def response(self, b):
        raw_filter = re.sub(r"[^0-9.-]+", ", ", self.filter_box.value) 

        
        #If filter changed then update data
        if list(self.filter) != ast.literal_eval('[' + raw_filter + ']'):
            self.filter = ast.literal_eval('[' + raw_filter + ']')
            self.filter_box.value = raw_filter
            self.input, self.output = self._generate_data()

        index = random.randint(0,len(self.input)-1)
        with self.fig.batch_update():
            self.fig.data[0].y = self.input[index]
            self.fig.data[1].y = self.output[index]
        
        plot_range_in = np.abs(np.array(self.input)).max()+0.2
        plot_range_out = np.abs(np.array(self.output)).max()+0.2
        self.fig.update_yaxes(range=[-plot_range_out, plot_range_out], row=1,col=self.num_of_plots)
        self.fig.update_yaxes(range=[-plot_range_in, plot_range_in], row=1,col=1)
       
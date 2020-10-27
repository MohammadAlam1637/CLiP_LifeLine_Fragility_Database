import pandas as pd     #(version 1.0.0)
import plotly           #(version 4.5.4) pip install plotly==4.5.4
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash             #(version 1.9.1) pip install dash==1.9.1
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from collections import defaultdict
import dash_bootstrap_components as dbc


import numpy as np
import scipy.stats as ss
import math 




app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)

#---------------------------------------------------------------
#Taken from https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases
df = pd.read_csv('CLiP_Database_Private.csv')
Infra_class    = df['Infrastructure_class'].unique()
Infra_subclass = df['Infrastructure_subclass'].unique()
Hazard         = df['Hazard_type'].unique()
Intensity      = df['IM'].unique()

# Create controls
Infra_class_options = [
    {'label': i ,'value': i } for i in Infra_class
 ]

Infra_subclass_options = [
    {'label': i ,'value': i } for i in Infra_subclass
 ]
Hazard_options = [
    {'label': i ,'value': i } for i in Hazard
 ]
Intensity_options = [
    {'label': i ,'value': i } for i in Intensity
 ]
 

############################################################
# Initial Figures, Pictures
############################################################

fig_single_fragility    = go.Figure()
fig_radar               = go.Figure()

#########################################################################
# App layout
#########################################################################

app.layout = html.Div([
    #-----------------------------------------------------------------
    # Banner
    #-----------------------------------------------------------------  
    html.Div([
            dbc.Row(
                [   dbc.Col(html.Div([
                    html.Img(
                        src= './assets/OSU_logo.png',
                        id="OSU-image",
                        style={
                                'height': '100px',
                                'width': '95%',
                                'margin-top':       '16px',
                                'margin-right':     '0px',
                                'margin-bottom':    '10px',
                                'margin-left':      '5px',
                               
                            },
                        )
                    ]
                    ), md =3),


                    dbc.Col(html.Div([
                        html.H3(
                            "Fragility Function Viewer",
                            style={
                                'textAlign': 'center',
                                'fontSize': 40,
                                'fontWeight':'bold',                 
                                'backgroundColor': 'white',

                                'padding': '5px',

                                "margin-top":       "16px",
                                'margin-right':     '5px',
                                "margin-bottom":    "0px",
                                'margin-left':      '0px',
                                
                                },
                                ),
                        html.H5(
                            "CLiP Lifeline Fragility Database", 
                            style={
                                    'textAlign':'center',
                                    'fontSize': 25,
                                    'fontWeight':'bold',
                                    'backgroundColor': 'white',

                                    'padding-bottom':'3px',

                                    'margin-top':    '0px',
                                    'margin-right':  '5px',
                                    'margin-bottom': '0px',
                                    'margin-left':   '0x',
                                   
                                    }
                                ),]
                    ), md = 6),

                    dbc.Col(html.Div([
                        html.Img(
                            src   ='./assets/CLiP.PNG',
                            id    ="CLiP-image",
                            style ={
                                "height": "100px",
                                "width": '95%',
                                # 'padding-left': '15px',
                            
                                'margin-top':       '16px',
                                'margin-right':     '0px',
                                'margin-bottom':    '30px',
                                'margin-left':      '5px'
                            },
                        
                        )]
                    ),md =3),

                ]
                ),
        ], id = 'Header', style= {"margin-bottom": "0px"},
       
        ),

         
    #-----------------------------------------------------------------
    # Keywords
    #-----------------------------------------------------------------
    html.Div(
      [
        html.P(
            children = ['Searchable Keywords'],
            style ={'textAlign':'center','fontWeight':'bold', 'fontSize': 16, 'padding': '0px', 'margin-right': '5px', 'margin-left': '5px','backgroundColor': 'tomato'}),

      ]),

    html.Div([
        dbc.Row(
            [   dbc.Col(html.Div([            
                html.H6('Infrastructure class', style ={'fontWeight':'bold', 'fontSize': 14, 'margin-right': '5px', 'margin-left': '5px'}),
                dcc.Dropdown(
                id          = 'Infrastructure_class_id',
                options     = Infra_class_options,
                value       = Infra_class[0],
                style       ={'fontWeight':'normal', 'fontSize': 12,'margin-right': '6px', 'margin-left': '2px'},
                multi       = False,
                clearable   = False,
            ),

            ]), md = 3),

                dbc.Col(html.Div([            
                html.H6('Infrastructure subclass', style ={'fontWeight':'bold', 'fontSize': 14,'margin-right': '5px', 'margin-left': '5px'}),
                dcc.Dropdown(
                id          = 'Infrastructure_subclass_id',
                options     = Infra_subclass_options,
                value       = Infra_subclass[0],
                style       ={'fontWeight':'normal', 'fontSize': 12,'margin-right': '6px', 'margin-left': '2px'},
                multi       = False,
                clearable   = False,
            ),

            ]), md = 3),
                dbc.Col(html.Div([            
                html.H6('Hazard type', style ={'fontWeight':'bold', 'fontSize': 14,'margin-right': '5px', 'margin-left': '5px'}),
                dcc.Dropdown(
                id          = 'Hazard_type_id',
                options     = Hazard_options,
                value       = Hazard[0],
                style       ={'fontWeight':'normal', 'fontSize': 12,'margin-right': '6px', 'margin-left': '2px'},
                multi       = False,
                clearable   = False,
            ),
            ]), md = 3),

                dbc.Col(html.Div([            
                html.H6('Intensity measure', style ={'fontWeight':'bold', 'fontSize': 14,'margin-right': '5px', 'margin-left': '5px'}),
                dcc.Dropdown(
                id          = 'Intensity_measure_id',
                options     = Intensity_options,
                value       = Intensity[0],
                style       ={'fontWeight':'normal', 'fontSize': 12,'margin-right': '6px', 'margin-left': '2px'},
                multi       = False,
                clearable   = False,
            ),
            ]), md = 3),

            ])
        ]),


    
    html.Div([
        html.H6('Search query', style ={'fontWeight':'bold', 'fontSize': 14, 'margin-right': '5px', 'margin-left': '5px'}),

        dcc.RadioItems(
            id      ='filter-query-read-write',
            options =[
                {'label': 'Display filter query', 'value': 'read'},
                {'label': 'Write to filter query', 'value': 'write'}
        ],
            value       ='read',
            style       = {'fontWeight':'normal', 'fontSize': 12,'margin-right': '5px', 'margin-left': '5px'}
        ),

        dbc.Row([
            dbc.Col(html.Div(
            [        
            dcc.Input(
                id='filter-query-input', 
                placeholder='Enter filter query', 
                ),
            ], 
            style = {'fontWeight':'normal', 'fontSize': 12,'margin-right': '5px', 'margin-left': '5px'}),
            md =12),
            ]),

        dbc.Row([
            dbc.Col(html.Div( 
            id='filter-query-output',
            style = {'fontWeight':'normal', 'fontSize': 12,'margin-right': '5px', 'margin-left': '15px'}),
            md =12),
            ]),
        ]),

    html.Br(),
   
    #-----------------------------------------------------------------
    # Database
    #-----------------------------------------------------------------    

    html.Div(
      [
        html.P(
            children =['Fragility Database'],
            style ={'textAlign':'center','fontWeight':'bold', 'fontSize': 16, 'padding': '0px', 'margin-right': '5px', 'margin-left': '5px','backgroundColor': 'tomato'}),

      ]
      ),

    dbc.Row([
        dbc.Col(html.Div(
        children =[
            dash_table.DataTable(
            id='datatable_id',
            data=df.to_dict('records'),
            columns=[
                {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
            ],
            
            fixed_rows ={'headers': False}, # True, False

            # Data Table Interactivity
            editable=False,               # Editable cells
            filter_action="native",       # Filtering by columns
            sort_action="native",         # Sorting by columns
            sort_mode="multi",            #
            # column_selectable="multi",  # Selecting columns
            row_selectable="multi",       # Selecting rows
            row_deletable=False,          # Deleting rows
            # selected_columns =[],
            selected_rows=[0],
            page_action= 'native',        # options: native and none; 'none' fits the whole table with vertical scroll
            
            page_size= 8,
            style_table={'height': '300px','overflowX': 'auto', 'margin': '5px'}, # fit within the page with horizontal scroll

            style_header={
                'textAlign': 'left',
                'backgroundColor': 'grey',
                'fontWeight': 'bold',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',

            },

            style_cell={
                'textAlign': 'left', 
                'padding': '5px',
                'width': '200px', 'minWidth': '120px', 'maxWidth': '200px',
                'overflow': 'hidden', 'textOverflow': 'ellipsis' ,
                'margin-right': '5px', 'margin-left': '5px'},

            style_data_conditional=[
            {   'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
            },
            ],
        ),
        ], 
        ), md =12)]),
    

    html.Br(), 

    html.Div(id = 'Output_container', children = [])  ,
    html.Br(),

    #-----------------------------------------------------------------
    # Figures(Picture, Plots)
    #-----------------------------------------------------------------   

    html.Div([
        dbc.Row(
            [ 
                dbc.Col(html.Div([
                    html.P(
                        children = ['Single Fragility Function'],
                        style ={'textAlign':'center','fontWeight':'bold', 'padding': '0px'},
                    ),
                    dcc.Graph(
                        id      =   'Single_Fragility_Function_id',
                        figure  =   fig_single_fragility,
                    ),
                    ], style = {'margin-left': '5px', 'margin-right':'5px'}), 
                    md = 4),

                dbc.Col(html.Div([                 
                    html.P(
                        children = ['Infrastructure Picture'],
                        style ={'textAlign':'center','fontWeight':'bold', 'padding': '0px'},
                    ),
                    html.Div(
                        id = 'Infrastructure_pic_id',
                        style = {'height': '450px'},
                        children = []  
                    ),
                    ], style = {'margin-left': '5px', 'margin-right':'5px'},),
                     md = 4),
                
                dbc.Col(html.Div([
                    html.P(
                        children = ['Quality Score'],
                        style ={'textAlign':'center','fontWeight':'bold', 'padding': '0px'},
                    ),
                    dcc.Graph(
                        id      =   'Quality_spider_plot_id',
                        figure  =   fig_radar,
                    ),
                    ], style = {'margin-left': '5px', 'margin-right':'5px'}), 
                    md = 4),

            ]
        )

        ]),

    html.Br(),


    #-----------------------------------------------------------------
    # Fragility Comparison Figure (1st row)
    #----------------------------------------------------------------- 
    html.Div(
      [
        html.P(
            children = ['Comparison of Fragility Functions'],
            style ={'textAlign':'center','fontWeight':'bold', 'fontSize': 16, 'padding': '0px', 'margin-right': '5px', 'margin-left': '5px','backgroundColor': 'tomato'}),

      ]),   

    html.Br(),

    # html.Div([
    #     dbc.Row([
    #         dbc.Col(html.Div(id = 'Output_Fragility_id_1', children = []), width = {'size': 6, 'offset':3}),
    #         ]),
    #     ]),

    html.Div(id = 'Output_Fragility_id_1', children = [], className = 'six_columns'), #'six_columns'

    html.Br(),

    #-----------------------------------------------------------------
    # Seleted Fragility Functions
    #-----------------------------------------------------------------    

    html.Div(
      [
        html.P(
            children = ['Fragility Functions for Exporting'],
            style ={'textAlign':'center','fontWeight':'bold', 'fontSize': 16, 'padding': '0px', 'margin-right': '5px', 'margin-left': '5px','backgroundColor': 'tomato'}),

      ]
      ), 

    html.Div([      
        html.Div([
        dash_table.DataTable(
            id='datatable_selected_id',
            data= [{}],
            columns=[
                {"name": i, "id": i, "deletable": False, "selectable": False} for i in df.columns
            ],
            
            fixed_rows ={'headers': False}, # True, False

            # Data Table Interactivity
            editable=False,                 # Editable cells
            # filter_action="native",       # Filtering by columns
            sort_action="native",           # Sorting by columns
            sort_mode="multi",              #
            row_selectable = 'multi',       # Selecting rows [options: 'single', 'multi', false]
            row_deletable =True,             # Deleting rows

            selected_rows=[0],

            export_columns = 'all',
            export_format  = 'csv',
            export_headers = 'names',

            # selected_row_ids = [],
            page_action= 'native',        # options: native and none; 'none' fits the whole table with vertical scroll
            
            page_size= 100,
            style_table={'height': '300px','overflowX': 'auto'}, # fit within the page with horizontal scroll
            style_header={
                'textAlign': 'left',
                'backgroundColor': 'grey',
                'fontWeight': 'bold',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },

            style_cell={
                'textAlign': 'left', 
                'padding': '5px',
                'width': '200px', 'minWidth': '120px', 'maxWidth': '200px',
                'overflow': 'hidden', 'textOverflow': 'ellipsis' },

            style_data_conditional=[
            {   'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
            },
            ],
        ),], className='twelve columns')                                        
    ],className='row'),

    html.Br(),




    ]
)

#-----------------------------------------------------------------
# Callbacks
#----------------------------------------------------------------- 

@app.callback(
    [Output('filter-query-input', 'style'),
     Output('filter-query-output', 'style')],
    [Input('filter-query-read-write', 'value')]
)
def query_input_output(val):
    input_style = {'width': '100%'}
    output_style = {}
    if val == 'read':
        input_style.update(display='none')
        output_style.update(display='inline-block')
    else:
        input_style.update(display='inline-block')
        output_style.update(display='none')
    return input_style, output_style


@app.callback(
    Output('datatable_id', 'filter_query'),
    [Input('filter-query-input', 'value')]
)
def write_query(query):
    if query is None:
        return ''
    return query


@app.callback(
    Output('filter-query-output', 'children'),
    [Input('datatable_id', 'filter_query')]
)
def read_query(query):
    if query is None:
        return "No filter query"
    return dcc.Markdown('`filter_query = "{}"`'.format(query))



@app.callback(
    Output(component_id = 'datatable_id', component_property = 'style_data_conditional'),
         
    [Input(component_id = 'datatable_id', component_property = 'selected_rows'),
     
    ]
    )


def update_styles(selected_rows):
    return [{
        'if': { 'row_index': i },
        'background_color': '#D2F3FF'
    } for i in selected_rows]


    

############################################################################################
# SINGLE FRAGILITY PLOT
############################################################################################
@app.callback(
    Output('Single_Fragility_Function_id','figure'),
    [Input ('datatable_id', 'selected_rows')]
    )
def update_Infra_Hazard_info(selected_rows):
    np.seterr(divide = 'ignore')

    Intensity_measure   = df.loc [selected_rows[-1], ['IM', 'IM_unit']]
    IM                  = Intensity_measure.IM
    IM_unit             = Intensity_measure.IM_unit
    xlabel = IM + " "+ "("+ IM_unit +")" 

    Fr_distribution = df.loc[selected_rows[-1], ['Fragility_distribution']]
    Fr_distribution = Fr_distribution.Fragility_distribution

    nods = df.loc[selected_rows[-1],['No_of_damage_state']]
    nods = nods.No_of_damage_state

    xlimit = 0

    #----------------------------------------------
    # Parametric Fragility
    #----------------------------------------------

    if (Fr_distribution == 'LogNormal') or (Fr_distribution == 'Lognormal') or (Fr_distribution == 'Normal'):
        # x-axis limit
        ls_max = 'Damage_state_med_' + str(nods)
        xlimit = df.loc [selected_rows[-1], [ls_max]]
        xlimit = round(4*float(xlimit[0]))

        if xlimit < float(1):
            xlimit = float(1)
        else:
            xlimit = xlimit

        x = np.linspace(0, xlimit, 100)


        # Fragility parameters for parametric fragility functions
        ds_f  = []
        mu_f        = []
        sigma_f     = []

        for i in range(1,nods+1):
            ds    = 'Damage_state_' + str(i)
            med   = 'Damage_state_med_' + str(i)
            std   = 'Damage_state_std_' + str(i)
            param = df.loc[selected_rows[-1], [ds, med, std, 'Dispersion_type']]
            ds_des = param[ds]
            Dispr_type = param.Dispersion_type
            mu     = float(param[med])            
            sigma  = float(param[std])

            ds_f.append(ds_des)
            mu_f.append(mu)
            sigma_f.append(sigma)


    # computing the CDFs
    y_cdf_lnorm  = []
    y_cdf_norm   = []

    for i in range(nods):
        if (Fr_distribution == 'LogNormal') or (Fr_distribution == 'Lognormal'):
            if Dispr_type == 'Logarithmic_standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x),np.log(mu_f[i]),sigma_f[i])
                y_cdf_lnorm.append(y_cdf)
            elif Dispr_type == 'Standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x),np.log(mu_f[i]),(sigma_f[i]/mu_f[i]))
                y_cdf_lnorm.append(y_cdf)

            

        elif Fr_distribution == 'Normal':           
            y_cdf = ss.norm.cdf(x,mu_f[i],sigma_f[i])
            y_cdf_norm.append(y_cdf)

  
    # Plotting the Fragility functions
    fig = go.Figure()

    if (Fr_distribution == 'LogNormal') or (Fr_distribution == 'Lognormal') or (Fr_distribution == 'Normal') or (Fr_distribution == 'Discrete'):
        fig.update_layout(#title = 'Single Fragility Function', 
            plot_bgcolor = '#FFF',    
            xaxis_title = xlabel,
            yaxis_title = 'Probability of exceedance', 
            showlegend=True,
            # equal_axes = True,
            # height = 400,
            # width  = 450, 

            # autosize = True,
            legend = dict(
                    orientation = 'h',
                    yanchor = 'bottom',
                    y = 1.02,
                    xanchor = 'right',
                    x = 1
                    ),          
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),          
            )
        fig.update_xaxes(
            range= [0, xlimit],
            rangemode = 'tozero',
            )
        fig.update_yaxes(
            range= [0, 1],
            rangemode = 'tozero',
            )


    if (Fr_distribution == 'LogNormal') or (Fr_distribution == 'Lognormal'):
            for i in range (nods):
                fig.add_trace(go.Scatter(x = x, y = y_cdf_lnorm[i], name = ds_f[i], mode = 'lines'))       
            return fig

    elif Fr_distribution == 'Normal':
            for i in range (nods):
                fig.add_trace(go.Scatter(x = x, y = y_cdf_norm[i], name = ds_f[i], mode = 'lines'))     
            return fig

    #----------------------------------------------------------
    # Discrete Fragility Functions
    #----------------------------------------------------------
    if Fr_distribution == 'Discrete':
        ds_f  = []
        x_f   = []
        y_f   = []

        for i in range(1,nods+1):
            ds    = 'Damage_state_' + str(i)
            med   = 'Damage_state_med_' + str(i)
            std   = 'Damage_state_std_' + str(i)
            param = df.loc[selected_rows[-1], [ds, med, std]]
            ds_des = param[ds]

            mu     = param[med]

            # print(type(mu))
            mu     = mu.split()
            mu     = [float(i) for i in mu[1:None]]
            mu.insert(0, 0.0)
            sigma  = param[std]
            sigma  = sigma.split()
            sigma  = [float(i) for i in sigma[1:None]]
            sigma.insert(0, 0.0)

            ds_f.append(ds_des)
            x_f.append(mu)
            y_f.append(sigma)


        xlimit = round(1.5*(x_f[nods-1][-1]))

        for i in range (nods):
            fig.add_trace(go.Scatter(x = x_f[i], y = y_f[i], name = ds_f[i], mode = 'lines+markers'))
            fig.update_xaxes(
                range = [0, xlimit]
                )
        return fig

    #----------------------------------------------------------
    # Polynomial Fragility Functions
    #----------------------------------------------------------

    if Fr_distribution == 'Polynomial':
        Poln = df.loc[selected_rows[-1],['Fragility_description', 'Fragility_polynomial']]
        Fr_des = Poln.Fragility_description
        Fr_eqn = Poln.Fragility_polynomial

        PGV_max_in_sec = 40   # in/sec
        PGV_max_cm_sec = 100  # cm/sec
        PGD_max_in     = 150  # 20 ft
        PGD_max_cm     = 350  #
        Dia_max_in     = 20   # in
        Dia_max_cm     = 50   # cm
        h_max_in       = 300  # 25 ft
        h_max_cm       = 760  # cm

        rep = {'exp': 'np.exp', 'log': 'np.log'}

        if ('exp' in Fr_eqn) or('log' in Fr_eqn):
            for x,y in rep.items():
                Fr_eqn = Fr_eqn.replace(x,y)

        if (IM_unit == 'in/s') or (IM_unit == 'in'):
            PGV = np.linspace(0.0001, PGV_max_in_sec, 100)
            D   = np.linspace(0.0001, Dia_max_in, 100)
            h   = np.linspace(0.0001, h_max_in, 100)
            PGD = np.linspace(0.0001, PGD_max_in,100)

        elif (IM_unit == 'cm/s') or (IM_unit == 'cm'):
            PGV = np.linspace(0.0001, PGV_max_cm_sec, 100)
            PGD = np.linspace(0.0001, PGD_max_cm, 100)
            D   = np.linspace(0.0001, Dia_max_cm, 100)
            h   = np.linspace(0.0001, h_max_cm, 100)

        fig.update_layout(#title = 'Single Fragility Function',
            plot_bgcolor = '#FFF',
            xaxis_title = xlabel,
            yaxis_title = Fr_des,
            # xaxis_type  = 'log',
            # yaxis_type  = 'log',
            showlegend=True, 
            # legend_title_text='Regression equation',
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1
                ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            )

        if IM == 'PGV':
            fig.add_trace(go.Scatter(x = PGV, y = eval(Fr_eqn), name = Fr_eqn, mode = 'lines'))
            fig.update_xaxes(
              range = [0, np.amax(PGV)],
              rangemode = 'tozero')
            fig.update_yaxes(
              range = [0, np.amax(eval(Fr_eqn))],
              rangemode = 'tozero')           
            return fig

        elif IM == 'PGD':   
            fig.add_trace(go.Scatter(x = PGD, y = eval(Fr_eqn), name = Fr_eqn, mode = 'lines'))
            fig.update_xaxes(
                range = [0, np.amax(PGD)],
                rangemode = 'tozero')
            fig.update_yaxes(
                range = [0, np.amax(eval(Fr_eqn))],
                rangemode = 'tozero')            
            return fig


############################################################################################
# FRAGILITY QUALITY PLOT
############################################################################################
@app.callback(
    Output('Quality_spider_plot_id','figure'),
    [Input ('datatable_id', 'selected_rows')]
    )
def quality_score_info(selected_rows):
    np.seterr(divide = 'ignore')

    categories          = ['Intensity measure score', 'Number of damage states score', 'Region score', 'Normalized score']
    Ranking_score       = df.loc[selected_rows[-1], ['Ranking_score_IM', 'Ranking_score_NDS', 'Ranking_score_Region', 'Ranking_score_Normalized']]
    IM_score            = Ranking_score.Ranking_score_IM
    NDS_score           = Ranking_score.Ranking_score_NDS
    Region_score        = Ranking_score.Ranking_score_Region
    Normalized_score    = Ranking_score.Ranking_score_Normalized

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar( 
        r = [IM_score, NDS_score, Region_score, Normalized_score], 
        theta = categories,
        fill = 'toself'
        ))

    fig.update_layout(#title = 'Quality Score Radar Plot', #font_size = 15,
        polar   =   dict(       
            radialaxis  =dict(
                visible =   True,
                range  =[0, 1]
                ),
            ),
        showlegend=False
        )

    return fig


############################################################################################
# Infrastructure Picture
############################################################################################
@app.callback(
    Output('Infrastructure_pic_id','children'),
    [Input ('datatable_id', 'selected_rows')]
    )
def update_infra_image(selected_rows):

    IM_loc   = df.loc[selected_rows[-1],['Infrastructure_image']]
    IM_loc   = IM_loc.Infrastructure_image

    Image_infra= html.Img(
        src = IM_loc,
        style = {
        'height': '100%',
        'width': '100%',
        # 'padding': '5px',
        }
        )

    return Image_infra


############################################################################################
# Seltected Fragility
############################################################################################
@app.callback(
    [Output('datatable_selected_id','data'),
     Output('datatable_selected_id','selected_rows'),
     ],
    [Input ('datatable_id', 'selected_rows'),
    ]
    )
def update_selected_fragility(selected_rows):
    selected_frag = df.iloc[selected_rows]
    selected_frag = selected_frag.to_dict('records')
    index_list_checkmark = list(range(0, len(selected_rows)))
    return selected_frag, index_list_checkmark



############################################################################################
# Removing and updating Fragility
############################################################################################
@app.callback(
    Output('datatable_id','selected_rows'),
    [Input ('datatable_selected_id', 'selected_rows'),
     Input ('datatable_selected_id', 'derived_virtual_data')
    ]
    )
def update_removed_fragility( selected_rows, derived_virtual_data ): 
    df_new = pd.DataFrame.from_dict(derived_virtual_data)
    Frag_no = df_new['Fragility_no'][selected_rows].values-1

    # index_rows = Frag_no-1

    return Frag_no

############################################################################################
# MULTIPLE FRAGILITY PLOT
############################################################################################
@app.callback(
    [Output('Output_Fragility_id_1','children'),
     # Output('Output_Fragility_id_2','children'),
     # Output('Output_Failure_Function_id','children'),  
     ],
    [Input ('datatable_id', 'selected_rows'),
     Input ('datatable_id', 'data'),
    ]
    )

def update_multi_fragility(selected_rows,data):
    np.seterr(divide = 'ignore')
    df_sel_frag = pd.DataFrame(data) # new dataframe with selected fragility

    IM_list = df_sel_frag.loc[selected_rows, ['IM']]
    IM_list = IM_list['IM'].values.tolist()
 
    Fr_dr_list = df_sel_frag.loc[selected_rows, ['Fragility_distribution']]
    Fr_dr_list = Fr_dr_list['Fragility_distribution'].values.tolist()

    nods_list = df_sel_frag.loc[selected_rows,['No_of_damage_state']]
    nods_list = nods_list['No_of_damage_state'].values.tolist()
   
    xlimit = 0
    IM_list_same = False

        

    fig_PGA     = go.Figure()
    fig_Sa      = go.Figure()
    fig_PGV     = go.Figure()
    fig_PGD     = go.Figure()
    fig_ASI     = go.Figure()
    fig_WSP     = go.Figure()
    fig_PGV_in  = go.Figure()
    fig_PGV_cm  = go.Figure()
    fig_PGD_in  = go.Figure()
  
    fig_PGA.update_layout( 
            plot_bgcolor = '#FFF',      
            xaxis_title = 'PGA (g)',
            yaxis_title = 'Probability of exceedance', 
            showlegend=True, 
            # height = '1',
            # width  = '2',
            # # dimension = 'ratio',
            # autosize = False

            # legend_title_text='Damage states', 
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1,
                font = dict(size = 10)
            ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
                
            )

    fig_Sa.update_layout( 
            plot_bgcolor = '#FFF',      
            xaxis_title = 'Sa(g)',
            yaxis_title = 'Probability of exceedance', 
            showlegend=True, 
            # legend_title_text='Damage states', 
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1,
                font = dict(size = 10)
            ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
                
            )  
    fig_PGV.update_layout( 
            plot_bgcolor = '#FFF',      
            xaxis_title = 'PGV (cm/s)',
            yaxis_title = 'Probability of exceedance', 
            showlegend=True, 
            # legend_title_text='Damage states', 
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1,
                font = dict(size = 10)
            ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
                
            )  
    fig_PGD.update_layout(
            plot_bgcolor = '#FFF',      
            xaxis_title = 'PGD (in)',
            yaxis_title = 'Probability of exceedance', 
            showlegend=True, 
            # legend_title_text='Damage states', 
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1,
                font = dict(size = 10)
            ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
                
            )  
    fig_ASI.update_layout(
            plot_bgcolor = '#FFF',      
            xaxis_title = 'ASI(g*s)',
            yaxis_title = 'Probability of exceedance', 
            showlegend=True, 
            # legend_title_text='Damage states', 
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1,
                font = dict(size = 10)

            ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
                
            )  

    fig_WSP.update_layout(
            plot_bgcolor = '#FFF',      
            xaxis_title = 'Wind speed (mph)',
            yaxis_title = 'Probability of exceedance', 
            showlegend=True, 
            # legend_title_text='Damage states', 
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1,
                font = dict(size = 10)
            ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
                
            )

    fig_PGV_in.update_layout(
            plot_bgcolor = '#FFF',      
            xaxis_title = 'PGV (in/s)',
            yaxis_title = 'Repairs/1000 ft', 
            showlegend=True, 
            # legend_title_text='Damage states', 
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1,
                font = dict(size = 10)
            ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
                
            ) 

    fig_PGV_cm.update_layout( 
            plot_bgcolor = '#FFF',      
            xaxis_title = 'PGV (cm/s)',
            yaxis_title = 'Repairs/Km', 
            showlegend=True, 
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1,
                font = dict(size = 10)
            ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
                
            )

    fig_PGD_in.update_layout(
            plot_bgcolor = '#FFF',      
            xaxis_title = 'PGD (in)',
            yaxis_title = 'Repairs/1000 ft', 
            showlegend=True, 
            # legend_title_text='Damage states', 
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.02,
                xanchor = 'right',
                x = 1,
                font = dict(size = 10)
            ),
            xaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
            yaxis = dict(
                linecolor = 'black',
                linewidth = 2,
                mirror = True,
                # showgrid = True,
                zeroline = True,
                ),
                
            ) 
                        
    mu_list_PGA    = []
    mu_list_Sa     = []
    mu_list_PGV    = []
    mu_list_PGD    = []
    mu_list_ASI    = []
    mu_list_WSP    = []

    sigma_list_PGA  = []   
    sigma_list_Sa   = []
    sigma_list_PGV  = []    
    sigma_list_PGD  = []   
    sigma_list_ASI  = []
    sigma_list_WSP  = []

    Dispr_type_list_PGA = []
    Dispr_type_list_Sa  = []
    Dispr_type_list_PGV = []
    Dispr_type_list_PGD = []
    Dispr_type_list_ASI = []
    Dispr_type_list_WSP = []        

    Fr_dist_list_PGA    = []
    Fr_dist_list_Sa     = []
    Fr_dist_list_PGV    = []
    Fr_dist_list_PGD    = []
    Fr_dist_list_ASI    = []
    Fr_dist_list_WSP    = []

    Ds_descrip_list_PGA = []
    Ds_descrip_list_Sa  = []
    Ds_descrip_list_PGV = []
    Ds_descrip_list_PGD = []
    Ds_descrip_list_ASI = []
    Ds_descrip_list_WSP = []
    Ds_descrip_list_PGV_in  = []
    Ds_descrip_list_PGV_cm = []
    Ds_descrip_list_PGD_in  = []



    y_cdf_list_PGA = []
    y_cdf_list_Sa  = [] 
    y_cdf_list_PGV = [] 
    y_cdf_list_PGD = [] 
    y_cdf_list_ASI = [] 
    y_cdf_list_WSP = [] 
   
    x_discrete_list = []
    y_discrete_list = []

    Fr_des_list_PGV_in =[]
    Fr_eqn_list_PGV_in =[]
    Fr_des_list_PGV_cm =[]
    Fr_eqn_list_PGV_cm =[]

    Fr_des_list_PGD_in =[]
    Fr_eqn_list_PGD_in =[]    
    Fr_des_list_PGD_cm =[]
    Fr_eqn_list_PGD_cm =[]




    for i in range(len(IM_list)):
        Fr_info         = df_sel_frag.loc[selected_rows[i], ['IM', 'IM_unit', 'Fragility_distribution','Fragility_description', 'Infrastructure_subclass']]
        
        IM              = Fr_info.IM
        IM_unit         = Fr_info.IM_unit
        Fr_distribution = Fr_info.Fragility_distribution
        Fr_description  = Fr_info.Fragility_description
        Infra_subclass  = Fr_info.Infrastructure_subclass

        nods_i          = nods_list[i]


        # Parametric fragility    
        if (Fr_distribution == 'LogNormal') or (Fr_distribution == 'Lognormal') or (Fr_distribution == 'Normal'):

            for j in range(1, nods_i+1):
                ds      = 'Damage_state_' + str(j)
                med     = 'Damage_state_med_' + str(j)
                std     = 'Damage_state_std_' + str(j)
                param   = df_sel_frag.loc[selected_rows[i], ['Fragility_no', ds, med, std, 'Fragility_distribution', 'Dispersion_type']]
                Dispr_type = param.Dispersion_type
                mu      = float(param[med])
                sigma   = float(param[std])
                dstrb   = param.Fragility_distribution
                ds_des  = 'Fragility_' + str(param.Fragility_no) + '_' + param[ds]

                if (IM == 'PGA') and (IM_unit == 'm/s2'):
                    mu_PGA      = mu/9.81
                    sigma_PGA   = (sigma/9.81)
                    mu_list_PGA.append(mu_PGA)
                    sigma_list_PGA.append(sigma_PGA)
                    Fr_dist_list_PGA.append(dstrb)
                    Dispr_type_list_PGA.append(Dispr_type)
                    Ds_descrip_list_PGA.append(ds_des) 


                elif (IM == 'PGA') and (IM_unit == 'cm/s2'):
                    mu_PGA      = mu/981
                    sigma_PGA   = (sigma/981)
                    mu_list_PGA.append(mu_PGA)
                    sigma_list_PGA.append(sigma_PGA)
                    Fr_dist_list_PGA.append(dstrb)
                    Dispr_type_list_PGA.append(Dispr_type)
                    Ds_descrip_list_PGA.append(ds_des) 

                elif (IM == 'PGA') and (IM_unit == 'g'):
                    mu_PGA      = mu
                    sigma_PGA   = sigma
                    # dstrb_PGA   = dstrb
                    mu_list_PGA.append(mu_PGA)
                    sigma_list_PGA.append(sigma_PGA)
                    Fr_dist_list_PGA.append(dstrb)
                    Dispr_type_list_PGA.append(Dispr_type)
                    Ds_descrip_list_PGA.append(ds_des) 

                elif ((IM == 'Sa') or (IM == 'Sa-gm') or (IM == 'Sa(1.0s)') or (IM == 'Sa(0.3s)') or (IM == 'Sa(0.4s)') or (IM == 'Sa(0.5s)') ) and (IM_unit == 'g'):
                    mu_Sa      = mu
                    sigma_Sa   = sigma
                    mu_list_Sa.append(mu_Sa)
                    sigma_list_Sa.append(sigma_Sa)
                    Fr_dist_list_Sa.append(dstrb)
                    Dispr_type_list_Sa.append(Dispr_type)
                    Ds_descrip_list_Sa.append(ds_des) 

                elif (IM == 'PGV') and (IM_unit == 'in/s'):
                    mu_PGV      = mu
                    sigma_PGV   = sigma
                    mu_list_PGV.append(mu_PGV)
                    sigma_list_PGV.append(sigma_PGV)
                    Fr_dist_list_PGV.append(dstrb)
                    Dispr_type_list_PGV.append(Dispr_type)
                    Ds_descrip_list_PGV.append(ds_des) 

                elif (IM == 'PGV') and (IM_unit == 'cm/s'):
                    mu_PGV      = mu
                    sigma_PGV   = sigma
                    mu_list_PGV.append(mu_PGV)
                    sigma_list_PGV.append(sigma_PGV)
                    Fr_dist_list_PGV.append(dstrb)
                    Dispr_type_list_PGV.append(Dispr_type)
                    Ds_descrip_list_PGV.append(ds_des) 

                elif (IM == 'PGD') and (IM_unit == 'in'):
                    mu_PGD      = mu
                    sigma_PGD   = sigma
                    mu_list_PGD.append(mu_PGD)
                    sigma_list_PGD.append(sigma_PGD)
                    Fr_dist_list_PGD.append(dstrb)
                    Dispr_type_list_PGD.append(Dispr_type)
                    Ds_descrip_list_PGD.append(ds_des) 

                elif (IM == 'PGD') and (IM_unit == 'cm'):
                    mu_PGD      = mu/2.54
                    sigma_PGD   = sigma/2.54
                    # sigma_PGD   = (sigma/2.54)
                    # sigma_PGD   = np.log(np.exp(sigma)/2.54)
                    mu_list_PGD.append(mu_PGD)
                    sigma_list_PGD.append(sigma_PGD)
                    Fr_dist_list_PGD.append(dstrb)
                    Dispr_type_list_PGD.append(Dispr_type)
                    Ds_descrip_list_PGD.append(ds_des) 

                elif (IM == 'PGD') and (IM_unit == 'm'):
                    mu_PGD      = mu*0.03937
                    sigma_PGD   = sigma*0.03937
                    # sigma_PGD    = sigma/(3.28*12)
                    # sigma_PGD   = np.log(np.exp(sigma)*3.28*12)
                    mu_list_PGD.append(mu_PGD)
                    sigma_list_PGD.append(sigma_PGD)
                    Fr_dist_list_PGD.append(dstrb)
                    Dispr_type_list_PGD.append(Dispr_type)
                    Ds_descrip_list_PGD.append(ds_des) 

                elif (IM == 'ASI') and (IM_unit == 'g*s'):
                    mu_ASI      = mu
                    sigma_ASI   = (sigma)
                    mu_list_ASI.append(mu_ASI)
                    sigma_list_ASI.append(sigma_ASI)
                    Fr_dist_list_ASI.append(dstrb)
                    Dispr_type_list_ASI.append(Dispr_type)
                    Ds_descrip_list_ASI.append(ds_des) 

                elif (IM == 'SI') and (IM_unit == 'cm/s'):
                    mu_ASI      = mu/981
                    sigma_ASI   = (sigma/981)
                    mu_list_ASI.append(mu_ASI)
                    sigma_list_ASI.append(sigma_ASI)
                    Fr_dist_list_ASI.append(dstrb)
                    Dispr_type_list_ASI.append(Dispr_type)
                    Ds_descrip_list_ASI.append(ds_des)                     

                elif (IM == 'Wind speed') and (IM_unit == 'mph'):
                    mu_WSP      = mu
                    sigma_WSP   = (sigma)
                    mu_list_WSP.append(mu_WSP)
                    sigma_list_WSP.append(sigma_WSP)
                    Fr_dist_list_WSP.append(dstrb)
                    Dispr_type_list_WSP.append(Dispr_type)
                    Ds_descrip_list_WSP.append(ds_des) 

        elif Fr_distribution == 'Discrete':

            for j in range(1, nods_i+1):
                ds    = 'Damage_state_' + str(j)
                med   = 'Damage_state_med_' + str(j)
                std   = 'Damage_state_std_' + str(j)
                param = df_sel_frag.loc[selected_rows[i], ['Fragility_no', ds, med, std]]
                ds_des  = 'Fragility_' + str(param.Fragility_no) + '_' + param[ds]
         

                mu     = param[med]
                # print (mu)
                mu     = mu.split()
                mu     = [float(i) for i in mu[1:None]]
                mu.insert(0, 0.0)

                sigma  = param[std]
                sigma  = sigma.split()
                sigma  = [float(i) for i in sigma[1:None]]
                sigma.insert(0, 0.0)

                if (IM == 'PGA') and (IM_unit == 'm/s2'):
                    mu = [x_val/9.81 for x_val in mu]
                elif (IM == 'PGA') and (IM_unit == 'g'):
                    mu = mu

                x_discrete_list.append(mu)
                y_discrete_list.append(sigma)
                Ds_descrip_list_PGA.append(ds_des) 


        elif Fr_distribution == 'Polynomial':
            Poln = df_sel_frag.loc[selected_rows[i],['Fragility_no', 'Fragility_description', 'Fragility_polynomial']]
            Fr_des = Poln.Fragility_description
            Fr_eqn = Poln.Fragility_polynomial
            ds_des  = 'Fragility_' + str(Poln.Fragility_no) + '_' + 'leaks or breaks'

            

            PGV_max_in_sec = 40   # in/sec
            PGV_max_cm_sec = 100  # cm/sec
            PGD_max_in     = 150  # 20 ft
            PGD_max_cm     = 350  #
            Dia_max_in     = 20   # in
            Dia_max_cm     = 50   # cm
            h_max_in       = 300  # 25 ft
            h_max_cm       = 760  # cm

            rep = {'exp': 'np.exp', 'log': 'np.log'}

            if (IM == 'PGV') and (IM_unit == 'in/s'):
                if ('exp' in Fr_eqn) or('log' in Fr_eqn):
                    for x,y in rep.items():
                        Fr_eqn = Fr_eqn.replace(x,y)
                Fr_eqn_list_PGV_in.append(Fr_eqn)
                Fr_des_list_PGV_in.append(Fr_des)
                Ds_descrip_list_PGV_in.append(ds_des) 


            elif (IM == 'PGV') and (IM_unit == 'cm/s'):
                if ('exp' in Fr_eqn) or('log' in Fr_eqn):
                    for x,y in rep.items():
                        Fr_eqn = Fr_eqn.replace(x,y)
                Fr_eqn_list_PGV_cm.append(Fr_eqn)
                Fr_des_list_PGV_cm.append(Fr_des)
                Ds_descrip_list_PGV_cm.append(ds_des)                 

            elif (IM == 'PGD') and (IM_unit == 'in'):
                if ('exp' in Fr_eqn) or('log' in Fr_eqn):
                    for x,y in rep.items():
                        Fr_eqn = Fr_eqn.replace(x,y)
                Fr_eqn_list_PGD_in.append(Fr_eqn)
                Fr_des_list_PGD_in.append(Fr_des)
                Ds_descrip_list_PGD_in.append(ds_des) 

            elif (IM == 'PGD') and (IM_unit == 'cm'):
                if ('exp' in Fr_eqn) or('log' in Fr_eqn):
                    for x,y in rep.items():
                        Fr_eqn = Fr_eqn.replace(x,y)
                Fr_eqn_list_PGD_cm.append(Fr_eqn)
                Fr_des_list_PGD_cm.append(Fr_des)
                Ds_descrip_list_PGD_cm.append(ds_des) 


            # print(Fr_eqn_list_PGV_in)

            if (IM_unit == 'in/s') or (IM_unit == 'in'):
                PGV = np.linspace(0.0001, PGV_max_in_sec, 100)
                D   = np.linspace(0.0001, Dia_max_in, 100)
                h   = np.linspace(0.0001, h_max_in, 100)
                PGD = np.linspace(0.0001, PGD_max_in,100)

            elif (IM_unit == 'cm/s') or (IM_unit == 'cm'):
                PGV = np.linspace(0.0001, PGV_max_cm_sec, 100)
                PGD = np.linspace(0.0001, PGD_max_cm, 100)
                D   = np.linspace(0.0001, Dia_max_cm, 100)
                h   = np.linspace(0.0001, h_max_cm, 100)


#------------------------------------------------------
# Subplots
#-----------------------------------------------------
    # IM = PGA          
    for k in range(len(mu_list_PGA)):
        xlimit_PGA = round(4*max(mu_list_PGA))

        if xlimit_PGA < float(1):
            xlimit_PGA = float(0.1)
        else:
            xlimit_PGA = xlimit_PGA


        x_PGA    = np.linspace(0, xlimit_PGA, 1000)

        if (Fr_dist_list_PGA[k] == 'LogNormal') or (Fr_dist_list_PGA[k] == 'Lognormal'):
            if Dispr_type_list_PGA[k] == 'Logarithmic_standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x_PGA),np.log(mu_list_PGA[k]),sigma_list_PGA[k])
            elif  Dispr_type_list_PGA[k] == 'Standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x_PGA),np.log(mu_list_PGA[k]),(sigma_list_PGA[k]/mu_list_PGA[k]))
        elif (Fr_dist_list_PGA[k] == 'Normal'):
            y_cdf = ss.norm.cdf(x_PGA,mu_list_PGA[k],sigma_list_PGA[k])

        y_cdf_list_PGA.append(y_cdf)

      
        fig_PGA.add_trace(go.Scatter(x = x_PGA , y = y_cdf_list_PGA[k], name = Ds_descrip_list_PGA[k] ))
        fig_PGA.update_xaxes(
            range= [0, xlimit_PGA],
            rangemode = 'tozero',
            )
        fig_PGA.update_yaxes(
            range= [0, 1],
            rangemode = 'tozero',
            )

    # IM = Sa
    for k in range(len(mu_list_Sa)):
        xlimit_Sa = round(4*max(mu_list_Sa))
        x_Sa    = np.linspace(0, xlimit_Sa, 1000)

        if (Fr_dist_list_Sa[k] == 'LogNormal') or (Fr_dist_list_Sa[k] == 'Lognormal'):
            if Dispr_type_list_Sa[k] == 'Logarithmic_standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x_Sa),np.log(mu_list_Sa[k]),sigma_list_Sa[k])
            elif Dispr_type_list_Sa[k] == 'Standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x_Sa),np.log(mu_list_Sa[k]),(sigma_list_Sa[k]/mu_list_Sa[k]))
        elif (Fr_dist_list_Sa[k] == 'Normal'):
            y_cdf = ss.norm.cdf(x_Sa,mu_list_Sa[k],sigma_list_Sa[k])
        y_cdf_list_Sa.append(y_cdf)


        fig_Sa.add_trace(go.Scatter(x = x_Sa , y = y_cdf_list_Sa[k], name = Ds_descrip_list_Sa[k] ))
        fig_Sa.update_xaxes(
            range= [0, xlimit_Sa],
            rangemode = 'tozero',
            )
        fig_Sa.update_yaxes(
            range= [0, 1],
            rangemode = 'tozero',
            )

    # IM = PGV
    for k in range(len(mu_list_PGV)):
        xlimit_PGV = round(4*max(mu_list_PGV))
        x_PGV    = np.linspace(0, xlimit_PGV, 1000)

        if (Fr_dist_list_PGV[k] == 'LogNormal') or (Fr_dist_list_PGV[k] == 'Lognormal'):
            if Dispr_type_list_PGV[k] == 'Logarithmic_standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x_PGV),np.log(mu_list_PGV[k]),sigma_list_PGV[k])
            elif Dispr_type_list_PGV[k] == 'Standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x_PGV),np.log(mu_list_PGV[k]), (sigma_list_PGV[k]/mu_list_PGV[k]))

        elif (Fr_dist_list_PGV[k] == 'Normal'):
            y_cdf = ss.norm.cdf(x_PGV,mu_list_PGV[k],sigma_list_PGV[k])
        y_cdf_list_PGV.append(y_cdf)


        fig_PGV.add_trace(go.Scatter(x = x_PGV , y = y_cdf_list_PGV[k], name = Ds_descrip_list_PGV[k] ))
        fig_PGV.update_xaxes(
            range= [0, xlimit_PGV],
            rangemode = 'tozero',
            )
        fig_PGV.update_yaxes(
            range= [0, 1],
            rangemode = 'tozero',
            )

    # IM = PGD
    for k in range(len(mu_list_PGD)):
        xlimit_PGD = round(4*max(mu_list_PGD))
        if xlimit_PGD < float(1):
            xlimit_PGD = float(0.1)
        else:
            xlimit_PGD = xlimit_PGD

        x_PGD    = np.linspace(0, xlimit_PGD, 1000)

        if (Fr_dist_list_PGD[k] == 'LogNormal') or (Fr_dist_list_PGD[k] == 'Lognormal'):
            if Dispr_type_list_PGD[k] == 'Logarithmic_standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x_PGD),np.log(mu_list_PGD[k]),sigma_list_PGD[k])
            elif Dispr_type_list_PGD[k] == 'Standard_deviation': 
                y_cdf = ss.norm.cdf(np.log(x_PGD),np.log(mu_list_PGD[k]), (sigma_list_PGD[k]/mu_list_PGD[k]))

        elif (Fr_dist_list_PGD[k] == 'Normal'):
            y_cdf = ss.norm.cdf(x_PGD,mu_list_PGD[k],sigma_list_PGD[k])
        y_cdf_list_PGD.append(y_cdf)


        fig_PGD.add_trace(go.Scatter(x = x_PGD , y = y_cdf_list_PGD[k], name = Ds_descrip_list_PGD[k] ))
        fig_PGD.update_layout(
            yaxis_title = 'Probability of exceedance'
            )
        fig_PGD.update_xaxes(
            range= [0, xlimit_PGD],
            rangemode = 'tozero',
            )
        fig_PGD.update_yaxes(
            range= [0, 1],
            rangemode = 'tozero',
            )

    # IM = ASI
    for k in range(len(mu_list_ASI)):
        xlimit_ASI = round(4*max(mu_list_ASI))
        x_ASI      = np.linspace(0, xlimit_ASI, 1000)

        if (Fr_dist_list_ASI[k] == 'LogNormal') or (Fr_dist_list_ASI[k] == 'Lognormal'):
            if Dispr_type_list_ASI[k] == 'Logarithmic_standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x_ASI),np.log(mu_list_ASI[k]),sigma_list_ASI[k])
            elif Dispr_type_list_ASI[k] == 'Standard_deviation': 
                y_cdf = ss.norm.cdf(np.log(x_ASI),np.log(mu_list_ASI[k]), (sigma_list_ASI[k]/mu_list_ASI[k]))

        elif (Fr_dist_list_ASI[k] == 'Normal'):
            y_cdf = ss.norm.cdf(x_ASI,mu_list_ASI[k],sigma_list_ASI[k])
        y_cdf_list_ASI.append(y_cdf)

        fig_ASI.add_trace(go.Scatter(x = x_ASI , y = y_cdf_list_ASI[k], name = Ds_descrip_list_ASI[k] ))
        fig_ASI.update_xaxes(
            range= [0, xlimit_ASI],
            rangemode = 'tozero',
            )
        fig_ASI.update_yaxes(
            range= [0, 1],
            rangemode = 'tozero',
            )

    # IM = WSP
    for k in range(len(mu_list_WSP)):
        xlimit_WSP = round(4*max(mu_list_WSP))
        x_WSP      = np.linspace(0, xlimit_WSP, 1000)

        if (Fr_dist_list_WSP[k] == 'LogNormal') or (Fr_dist_list_WSP[k] == 'Lognormal'):
            if Dispr_type_list_WSP[k] == 'Logarithmic_standard_deviation':
                y_cdf = ss.norm.cdf(np.log(x_WSP),np.log(mu_list_WSP[k]),sigma_list_WSP[k])
            elif Dispr_type_list_WSP[k] == 'Standard_deviation': 
                y_cdf = ss.norm.cdf(np.log(x_WSP),np.log(mu_list_WSP[k]), (sigma_list_WSP[k]/mu_list_WSP[k]))

        elif (Fr_dist_list_WSP[k] == 'Normal'):
            y_cdf = ss.norm.cdf(x_WSP,mu_list_WSP[k],sigma_list_WSP[k])
        y_cdf_list_WSP.append(y_cdf)


        fig_WSP.add_trace(go.Scatter(x = x_WSP , y = y_cdf_list_WSP[k], name = Ds_descrip_list_WSP[k] ))
        fig_WSP.update_xaxes(
            range= [0, xlimit_WSP],
            rangemode = 'tozero',
            )
        fig_WSP.update_yaxes(
            range= [0, 1],
            rangemode = 'tozero',
            )


    # Discrete
    for k in range(len(x_discrete_list)):
        xlimit_dis = (2*max(x_discrete_list))
      
        fig_PGA.add_trace(go.Scatter(x = x_discrete_list[k], y = y_discrete_list[k],  mode = 'lines+markers', name = Ds_descrip_list_PGA[k] ))
        fig_PGA.update_xaxes(
            range = [0, xlimit_dis],
            rangemode = 'tozero'
            )
        fig_PGA.update_yaxes(
            range = [0, 1],
            rangemode = 'tozero'
            )

    # Polynomial

    for k in range(len(Fr_eqn_list_PGV_in)):
        fig_PGV_in.add_trace(go.Scatter(x = PGV, y = eval(Fr_eqn_list_PGV_in[k]),  mode = 'lines', name = Ds_descrip_list_PGV_in[k] ))
        fig_PGV_in.update_xaxes(
              range = [0, np.amax(PGV)],
              rangemode = 'tozero')
        fig_PGV_in.update_yaxes(
              range = [0, np.amax(eval(Fr_eqn_list_PGV_in[k]))],
              rangemode = 'tozero') 
        fig_PGV_in.update_layout(
            xaxis_title = 'PGV (in/s)',
            yaxis_title = Fr_des_list_PGV_in[k]
            )          

    for k in range(len(Fr_eqn_list_PGV_cm)):
        fig_PGV_cm.add_trace(go.Scatter(x = PGV, y = eval(Fr_eqn_list_PGV_cm[k]),  mode = 'lines', name = Ds_descrip_list_PGV_cm[k] ))
        fig_PGV_cm.update_xaxes(
              range = [0, np.amax(PGV)],
              rangemode = 'tozero')
        fig_PGV_cm.update_yaxes(
              range = [0, np.amax(eval(Fr_eqn_list_PGV_cm[k]))],
              rangemode = 'tozero') 
        fig_PGV.update_layout(
            xaxis_title = 'PGV (cm/s)',
            yaxis_title = Fr_des_list_PGV_cm[k]
            ) 

    for k in range(len(Fr_eqn_list_PGD_in)):
        fig_PGD_in.add_trace(go.Scatter(x = PGD, y = eval(Fr_eqn_list_PGD_in[k]),  mode = 'lines', name = Ds_descrip_list_PGD_in[k] ))
        fig_PGD_in.update_xaxes(
              range = [0, np.amax(PGD)],
              rangemode = 'tozero')
        fig_PGD_in.update_yaxes(
              range = [0, np.amax(eval(Fr_eqn_list_PGD_in[k]))],
              rangemode = 'tozero') 
        fig_PGD_in.update_layout(
            xaxis_title = 'PGD (in)',
            yaxis_title = Fr_des_list_PGD_in[k]
            )



    

    # Return 
    unique_IM = np.array(IM_list)
    unique_IM = np.unique(unique_IM)
    unique_IM = unique_IM.tolist()

    fig_list_row_1 = []
    fig_id_row_1   = []
    fig_list_row_1f = []

    
    fig_list_row_2 = []
    fig_id_row_2   = []
    fig_list_row_2f = []


    fig_list_row_3 = []
    fig_id_row_3   = []
    fig_list_row_3f = []


  

    # if (Fr_distribution == 'Lognormal') or (Fr_distribution == 'Normal') or (Fr_distribution == 'Discrete'):


            
    for i in range(len(unique_IM)):

        if unique_IM[i] == 'PGA':
            fig_id_row_1.append('figure_id_{}'.format(unique_IM[i]))
            fig_list_row_1.append(fig_PGA)

        elif unique_IM[i] == 'Sa':
            fig_id_row_1.append('figure_id_{}'.format(unique_IM[i]))
            fig_list_row_1.append(fig_Sa)           

        elif ((unique_IM[i] == 'ASI') or (unique_IM[i] == 'SI')):
            fig_id_row_1.append('figure_id_{}'.format(unique_IM[i]))
            fig_list_row_1.append(fig_ASI) 

        elif (unique_IM[i] == 'PGV') and ((Fr_distribution == 'Lognormal') or (Fr_distribution == 'Normal') or (Fr_distribution == 'Discrete')):
            fig_id_row_1.append('figure_id_{}'.format(unique_IM[i]))
            fig_list_row_1.append(fig_PGV)

        elif (unique_IM[i] == 'PGD') and ((Fr_distribution == 'Lognormal') or (Fr_distribution == 'Normal') or (Fr_distribution == 'Discrete')):
            fig_id_row_1.append('figure_id_{}'.format(unique_IM[i]))
            fig_list_row_1.append(fig_PGD)

        elif (unique_IM[i] == 'Wind speed'):
            fig_id_row_1.append('figure_id_{}'.format(unique_IM[i]))
            fig_list_row_1.append(fig_WSP)

        elif (unique_IM[i] == 'PGV') and (Fr_distribution == 'Polynomial') and (IM_unit == 'in/s'):
            fig_id_row_1.append('figure_id_in_{}'.format(IM))
            fig_list_row_1.append(fig_PGV_in)

        elif (unique_IM[i] == 'PGV') and (Fr_distribution == 'Polynomial') and (IM_unit == 'cm/s'):
            fig_id_row_1.append('figure_id_cm_{}'.format(IM))
            fig_list_row_1.append(fig_PGV_cm) 

        elif (unique_IM[i]== 'PGD') and (Fr_distribution == 'Polynomial') :
            fig_id_row_1.append('figure_id_in_{}'.format(IM))
            fig_list_row_1.append(fig_PGD_in)
            
    # for i in range(len(unique_IM)):

    #     if (unique_IM[i] == 'PGV') and ((Fr_distribution == 'Lognormal') or (Fr_distribution == 'Normal') or (Fr_distribution == 'Discrete')):
    #         fig_id_row_2.append('figure_id_{}'.format(unique_IM[i]))
    #         fig_list_row_2.append(fig_PGV)
           

    #     elif (unique_IM[i] == 'PGD') and ((Fr_distribution == 'Lognormal') or (Fr_distribution == 'Normal') or (Fr_distribution == 'Discrete')):
    #         fig_id_row_2.append('figure_id_{}'.format(unique_IM[i]))
    #         fig_list_row_2.append(fig_PGD)



    #     elif (unique_IM[i] == 'Wind speed') and ((Fr_distribution == 'Lognormal') or (Fr_distribution == 'Normal') or (Fr_distribution == 'Discrete')):
    #         fig_id_row_2.append('figure_id_{}'.format(unique_IM[i]))
    #         fig_list_row_2.append(fig_WSP)


    # unique_IM_unit = ['in/s', 'cm/s', 'in']

     


    # for i in range(len(unique_IM_unit)):

        if (IM == 'PGV') and (Infra_subclass == 'buried pipeline') and (IM_unit == 'in/s'):
            # uniqe_IM_unit.append('in/s')
            fig_id_row_3.append('figure_id_in_{}'.format(IM))
            fig_list_row_3.append(fig_PGV_in)

        elif (IM == 'PGV') and (Infra_subclass == 'buried pipeline') and (IM_unit == 'cm/s'):
            # uniqe_IM_unit.append('cm/s')
            fig_id_row_3.append('figure_id_cm_{}'.format(IM))
            fig_list_row_3.append(fig_PGV_cm) 

        elif (IM == 'PGD') and (Infra_subclass == 'buried pipeline'):
            # uniqe_IM_unit.append('in')
            fig_id_row_3.append('figure_id_in_{}'.format(IM))
            fig_list_row_3.append(fig_PGD_in)         



    for i in range(len(fig_list_row_1)):
        fig_list_row_1f.append(html.Div( children =[dcc.Graph(id = fig_id_row_1[i], figure = fig_list_row_1[i]) ])) # className='container four columns'
    
    # for i in range(len(fig_list_row_2)):
    #     fig_list_row_2f.append(html.Div( children =[dcc.Graph(id =fig_id_row_2[i], figure = fig_list_row_2[i])],  className='container four columns')) # className='container four columns' style = {'height': '200px','width' : '200px'}
    
    # for i in range(len(fig_list_row_3)):
    #     fig_list_row_3f.append(html.Div( children =[dcc.Graph(id =fig_id_row_3[i], figure = fig_list_row_3[i])],  className='container four columns')) # className='container four columns' style = {'height': '200px','width' : '200px'}
    
    return [fig_list_row_1f]

    # return [fig_list_row_1f, fig_list_row_2f]

    # return [fig_list_row_1f, fig_list_row_2f, fig_list_row_3f]


    #----------------------------------------------------------------- 
if __name__ == '__main__':
    app.run_server(debug=False,dev_tools_ui=False,dev_tools_props_check=False)
    # app.run_server(debug=True)
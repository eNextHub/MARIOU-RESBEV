#%%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd

df = pd.read_excel('Final.xlsx')
df = df.fillna(method='ffill', axis=0)

df = df.sort_values(['Yearly mileage','Time Horizon'],ascending=[True,True])
df['Gasoline Price'] = df['Gasoline Price'].map({0.0756:1.4, 0.0864:1.6, 0.0972:1.8, 0.108:2, 0.1188:2.2})

# %%
time_horizon = 10

p1 = 'Time Horizon'
p2 = 'Yearly mileage'

TH = sorted(list(set(df[p1])))
YM = sorted(list(set(df[p2])))

titles = [f'Ownership time {int(y)} years' for y in TH]
title = f'NPC of household residential energy system by {p1} and {p2} [EUR]'                

fig = make_subplots(
    rows=len(TH), 
    cols=1, 
    shared_yaxes=True, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    subplot_titles=titles
    )

for y in TH:

    if TH.index(y)==0:
        show_leg = True
    else:
        show_leg = False
    
    fig.add_trace(
        go.Violin(
            y=df['Total Cost [€]'][ df['BEV installation'] == 0][df[p1] == y],
            x=df[p2][ df['BEV installation'] == 0][df[p1] == y],
            legendgroup='ICEV', 
            scalegroup='ICEV', 
            name='ICEV', 
            showlegend=show_leg,
            side='negative',
            line_color='#003566', 
            line_width=1.5
            ), 
        col=1, 
        row=TH.index(y)+1
        )
    fig.add_trace(
        go.Violin(
            y=df['Total Cost [€]'][ df['BEV installation'] == 1][df[p1] == y],
            x=df[p2][ df['BEV installation'] == 1][df[p1] == y],
            legendgroup='BEV', 
            scalegroup='BEV', 
            name='BEV', 
            showlegend=show_leg,
            side='positive', 
            line_color='#ffc300', 
            line_width=1.5
            ), 
        col=1, 
        row=TH.index(y)+1
        )

fig.update_traces(meanline_visible=True)
fig.update_xaxes(
    type='category',
    categoryorder='array', 
    categoryarray=YM, 
    )
fig.update_xaxes(
    title='Annual travelled distance [km/y]', 
    row=4,
    col=1
    )

fig.update_yaxes(
    title='NPC [EUR]', 
    )

fig.update_layout(
    font_family="HelveticaNeue", 
    template="seaborn", 
    title=title, 
    font_size=16, 
    violingap=0, 
    violingroupgap=0,
    width=1000,
    height=950,
    legend=dict(title='Preferred vehicle', x=0.5,y=-0.1,xanchor='center',yanchor='top',orientation='h'),
    )

fig.write_html(f"plots\{p1},{p2},NPC.html", auto_open=True)


# %%
time_horizon = 10

p1 = 'Time Horizon'
p2 = 'Yearly mileage'

TH = sorted(list(set(df[p1])))
YM = sorted(list(set(df[p2])))

titles = [f'{p1}={y}' for y in TH]
title = f'Lifecycle CO2 emissions of household residential energy system by {p1} and {p2} [kg]'                

fig = make_subplots(
    rows=len(TH), 
    cols=1, 
    shared_yaxes=True, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    subplot_titles=titles
    )

for y in TH:

    if TH.index(y)==0:
        show_leg = True
    else:
        show_leg = False
    
    fig.add_trace(
        go.Violin(
            y=df['Total emissions [KgCO2e]'][ df['BEV installation'] == 0][df[p1] == y],
            x=df[p2][ df['BEV installation'] == 0][df[p1] == y],
            legendgroup='ICEV', 
            scalegroup='ICEV', 
            name='ICEV', 
            showlegend=show_leg,
            side='negative',
            line_color='#003566', 
            line_width=1.5
            ), 
        col=1, 
        row=TH.index(y)+1
        )
    fig.add_trace(
        go.Violin(
            y=df['Total emissions [KgCO2e]'][ df['BEV installation'] == 1][df[p1] == y],
            x=df[p2][ df['BEV installation'] == 1][df[p1] == y],
            legendgroup='BEV', 
            scalegroup='BEV', 
            name='BEV', 
            showlegend=show_leg,
            side='positive', 
            line_color='#ffc300', 
            line_width=1.5
            ), 
        col=1, 
        row=TH.index(y)+1
        )

fig.update_traces(meanline_visible=True)
fig.update_xaxes(
    type='category',
    categoryorder='array', 
    categoryarray=YM, 
    )
fig.update_xaxes(
    title='Yearly mileage [km]', 
    row=4,
    col=1
    )

fig.update_layout(
    font_family="HelveticaNeue", 
    template="seaborn", 
    title=title, 
    font_size=16, 
    violingap=0, 
    violingroupgap=0,
    legend_title='Vehicle chosen'
    )

fig.write_html(f"plots\{p1},{p2},CO2.html", auto_open=True)
fig.show()

# %%
time_horizon = 6
travelled_distance = 15000

p1 = 'Electricity price'
p2 = 'Gasoline Price'

TH = sorted(list(set(df[p1])))
YM = sorted(list(set(df[p2])))
    
titles = [f'Electricity price: {y} €/kWh' for y in TH]
title = f'NPC by {p1} and {p2} (ownership time>{time_horizon}y, travelled distance>{travelled_distance}km/y) [EUR]'                

fig = make_subplots(
    rows=len(TH), 
    cols=1, 
    shared_yaxes=True, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    subplot_titles=titles
    )

for y in TH:

    if TH.index(y)==0:
        show_leg = True
    else:
        show_leg = False
    
    fig.add_trace(
        go.Violin(
            y=df['Total Cost [€]'][ df['BEV installation'] == 0][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance],
            x=df[p2][ df['BEV installation'] == 0][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance],
            legendgroup='ICEV', 
            scalegroup='ICEV', 
            name='ICEV', 
            showlegend=show_leg,
            side='negative',
            line_color='#003566', 
            line_width=1.5,
            ),
        col=1,
        row=TH.index(y)+1
        )
    fig.add_trace(
        go.Violin(
            y=df['Total Cost [€]'][ df['BEV installation'] == 1][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance],
            x=df[p2][ df['BEV installation'] == 1][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance],
            legendgroup='BEV',
            scalegroup='BEV', 
            name='BEV', 
            showlegend=show_leg,
            side='positive', 
            line_color='#ffc300', 
            line_width=1.5
            ),
        col=1, 
        row=TH.index(y)+1
        )

fig.update_traces(meanline_visible=True)

# fig.update_xaxes(
#     type='category',
#     categoryorder='array', 
#     categoryarray=YM, 
#     )

fig.update_xaxes(
    title='Gasoline price [€/litre]', 
    row=7,
    col=1
    )

fig.update_yaxes(
    title='NPC [EUR]', 
    )

fig.update_layout(
    font_family="HelveticaNeue", 
    template="seaborn", 
    title=title, 
    font_size=16, 
    violingap=0, 
    violingroupgap=0,
    width=1000,
    height=1000,
    legend=dict(title='Preferred vehicle', x=0.5,y=-0.1,xanchor='center',yanchor='top',orientation='h'),
    )

fig.write_html(f"plots\{p1},{p2},NPC.html", auto_open=True)
fig.show()


# %%
tech = 'Induction stove installation'

p1 = 'Time Horizon'
p2 = 'Yearly mileage'

TH = sorted(list(set(df[p1])))
YM = sorted(list(set(df[p2])))

titles = [f"Without {tech.split(' ')[0]}", f"With {tech.split(' ')[0]}",'','','','','','']
            
title = f'NPC of household residential energy system by {p1} and {p2} [EUR]'                

fig = make_subplots(
    rows=len(TH), 
    cols=2, 
    shared_yaxes=True, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    subplot_titles=titles
    )

for y in TH:

    if TH.index(y)==0:
        show_leg = True
    else:
        show_leg = False
    
    fig.add_trace(
        go.Violin(
            y=df['Total Cost [€]'][ df['BEV installation'] == 0][df[p1] == y][df[tech] == 0],
            x=df[p2][ df['BEV installation'] == 0][df[p1] == y][df[tech] == 0],
            legendgroup='ICEV', 
            scalegroup='ICEV', 
            name='ICEV', 
            showlegend=show_leg,
            side='negative',
            line_color='#003566', 
            line_width=1.5
            ), 
        col=1, 
        row=TH.index(y)+1
        )

    fig.add_trace(
        go.Violin(
            y=df['Total Cost [€]'][ df['BEV installation'] == 0][df[p1] == y][df[tech] > 0],
            x=df[p2][ df['BEV installation'] == 0][df[p1] == y][df[tech] > 0],
            legendgroup='ICEV', 
            scalegroup='ICEV', 
            name='ICEV', 
            showlegend=False,
            side='negative',
            line_color='#003566', 
            line_width=1.5
            ), 
        col=2, 
        row=TH.index(y)+1
        )
    
    fig.add_trace(
        go.Violin(
            y=df['Total Cost [€]'][ df['BEV installation'] == 1][df[p1] == y][df[tech] == 0],
            x=df[p2][ df['BEV installation'] == 1][df[p1] == y][df[tech] == 0],
            legendgroup='BEV', 
            scalegroup='BEV', 
            name='BEV', 
            showlegend=show_leg,
            side='positive', 
            line_color='#ffc300', 
            line_width=1.5
            ), 
        col=1, 
        row=TH.index(y)+1
        )

    fig.add_trace(
        go.Violin(
            y=df['Total Cost [€]'][ df['BEV installation'] == 1][df[p1] == y][df[tech] > 0],
            x=df[p2][ df['BEV installation'] == 1][df[p1] == y][df[tech] > 0],
            legendgroup='BEV', 
            scalegroup='BEV', 
            name='BEV', 
            showlegend=False,
            side='positive', 
            line_color='#ffc300', 
            line_width=1.5
            ), 
        col=2, 
        row=TH.index(y)+1
        )

fig.update_traces(meanline_visible=True)
fig.update_xaxes(
    type='category',
    categoryorder='array', 
    categoryarray=YM, 
    )

fig.update_xaxes(
    title='Yearly mileage [km]', 
    row=4,
    col=1
    )
fig.update_xaxes(
    title='Yearly mileage [km]', 
    row=4,
    col=2
    )

fig['layout']['yaxis']['title']=f'{TH[0]} years'
fig['layout']['yaxis3']['title']=f'{TH[1]} years'
fig['layout']['yaxis5']['title']=f'{TH[2]} years'
fig['layout']['yaxis7']['title']=f'{TH[3]} years'

fig.update_layout(
    font_family="HelveticaNeue", 
    template="seaborn", 
    title=title, 
    font_size=16, 
    violingap=0, 
    violingroupgap=0,
    legend_title='Vehicle chosen'
    )

fig.write_html(f"plots\{p1},{p2},{tech.split(' ')[0]}.html", auto_open=True)
fig.show()

# %%
time_horizon = 6
travelled_distance = 15000

p1 = 'Electricity price'
p2 = 'Gasoline Price'

TH = sorted(list(set(df[p1])))[3]
YM = sorted(list(set(df[p2])))[2]
    
title = f'Electricity price: {TH} €/kWh'

fig = make_subplots(
    rows=2, 
    cols=1, 
    shared_yaxes=True, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    subplot_titles=['PV<3kW', 'PV>=3kW']
    )

    
fig.add_trace(
    go.Violin(
        y=df['Total Cost [€]'][df['BEV installation'] == 0][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance][df['PV installation'] < 3][df[p2]==YM],
        x=df[p2][ df['BEV installation'] == 0][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance][df['PV installation'] < 3][df[p2]==YM],
        legendgroup='ICEV', 
        scalegroup='ICEV', 
        name='ICEV', 
        showlegend=True,
        side='negative',
        line_color='#003566', 
        line_width=1.5,
        ),
    col=1,
    row=1
    )

fig.add_trace(
    go.Violin(
        y=df['Total Cost [€]'][df['BEV installation'] == 0][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance][df['PV installation'] >= 3][df[p2]==YM],
        x=df[p2][ df['BEV installation'] == 0][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance][df['PV installation'] >= 3][df[p2]==YM],
        legendgroup='ICEV', 
        scalegroup='ICEV', 
        name='ICEV', 
        showlegend=True,
        side='negative',
        line_color='#003566', 
        line_width=1.5,
        ),
    col=1,
    row=2
    )

fig.add_trace(
    go.Violin(
        y=df['Total Cost [€]'][ df['BEV installation'] == 1][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance][df['PV installation'] < 3][df[p2]==YM],
        x=df[p2][ df['BEV installation'] == 1][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance][df['PV installation'] < 3][df[p2]==YM],
        legendgroup='BEV',
        scalegroup='BEV', 
        name='BEV', 
        showlegend=show_leg,
        side='positive', 
        line_color='#ffc300', 
        line_width=1.5
        ),
    col=1, 
    row=1
    )

fig.add_trace(
    go.Violin(
        y=df['Total Cost [€]'][ df['BEV installation'] == 1][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance][df['PV installation'] >= 3][df[p2]==YM],
        x=df[p2][ df['BEV installation'] == 1][df[p1] == y][df['Time Horizon'] > time_horizon][df['Yearly mileage'] > travelled_distance][df['PV installation'] >= 3][df[p2]==YM],
        legendgroup='BEV',
        scalegroup='BEV', 
        name='BEV', 
        showlegend=show_leg,
        side='positive', 
        line_color='#ffc300', 
        line_width=1.5
        ),
    col=1, 
    row=2
    )

fig.update_traces(meanline_visible=True)

fig.update_xaxes(
    title='Gasoline price [€/litre]', 
    row=2,
    col=1
    )

fig.update_yaxes(
    title='NPC [EUR]', 
    )

fig.update_layout(
    font_family="HelveticaNeue", 
    template="seaborn", 
    title=title, 
    font_size=16, 
    violingap=0, 
    violingroupgap=0,
    width=300,
    height=1000,
    legend=dict(title='Preferred vehicle', x=0.5,y=-0.1,xanchor='center',yanchor='top',orientation='h'),
    )

fig.write_html(f"plots\{p1},{p2},NPC,PV.html", auto_open=True)
fig.show()

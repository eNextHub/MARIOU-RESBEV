#%%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import copy 

df = pd.read_excel('Final.xlsx')
df = df.fillna(method='ffill', axis=0)

df = df.sort_values(['Yearly mileage','Time Horizon'],ascending=[True,True])
df['Gasoline Price'] = df['Gasoline Price'].map({0.0756:1.4, 0.0864:1.6, 0.0972:1.8, 0.108:2, 0.1188:2.2})

#%% Figure 3 - BEV preferability
df_bevp = df.loc[:,['User','Location','Yearly mileage','Time Horizon','Electricity price','Gasoline Price',' BEV Incentives','PV installation','BEV installation','Total Cost [€]','Total emissions [KgCO2e]']]
df_bevp.columns = ['House size', 'Solar radiation', 'Annual travelled distance','Ownership time', 'National grid electricity price', 'National gasoline price', 'BEV purchase subsidies','PV capacity','BEV installation', 'NPC', 'CO2']

map_values = {
    'House size': {'A': 60, 'B':120, 'C':180, 'D':240},
    'Solar radiation': {'Milano':0.12, 'Roma': 0.14, 'Siracusa':0.16}
    }
for k,v in map_values.items():
    df_bevp[k] = df_bevp[k].map(v)


synth_results = {}
for sp in df_bevp.columns:
    
    if sp not in ["BEV installation","NPC","CO2"]:
        synth_results[sp] = pd.DataFrame()
        
        for value in sorted(list(set(df_bevp[sp]))):
            BP = df_bevp.query(f"`{sp}`=={value} & `BEV installation`==1").shape[0]/df_bevp.query(f"`{sp}`=={value}").shape[0]

            synth_results[sp] = pd.concat([
                synth_results[sp],
                pd.DataFrame(
                    BP, 
                    index=[value],
                    columns=['BEV preferability'],
                    ),
                ], axis=0)

        synth_results[sp] = synth_results[sp].fillna(0)
        synth_results[sp] = synth_results[sp].groupby(level=0,axis=0).sum()
        
synth_results['PV capacity'].index = [0,1,2,3,4,5,"+6","+6","+6","+6","+6","+6","+6","+6","+6","+6","+6"]
synth_results['PV capacity'] = synth_results['PV capacity'].groupby(level=0).mean()

ad_results = copy.deepcopy(synth_results) 

ad_results['Annual travelled distance'].index = ['V0','V2','V4','V6']
ad_results['BEV purchase subsidies'].index = ['V0','V2','V3','V4','V6']
ad_results['House size'].index = ['V0','V2','V4','V6']
ad_results['National gasoline price'].index = ['V0','V2','V3','V4','V6']
ad_results['National grid electricity price'].index = ['V6','V5','V4','V3','V2','V1','V0']
ad_results['PV capacity'].index = ['V0','V1','V2','V3','V4','V5','V6']
ad_results['Solar radiation'].index = ['V0','V3','V6']
ad_results['Ownership time'].index = ['V0','V2','V4','V6']

colors = {
    'Annual travelled distance': '#ff595e',
    'BEV purchase subsidies': '#ff924c',
    'House size': '#ffca3a',
    'National gasoline price': '#8ac926',
    'National grid electricity price': '#52a675',
    'Ownership time': '#1982c4',
    'PV capacity': '#4267ac',
    'Solar radiation': '#6a4c93',
    }

fig = go.Figure()
for sp in sorted(list(ad_results.keys())):
    df = ad_results[sp]
    
    if sp == 'PV capacity':
        dashed = 'dot'
    else:
        dashed = 'solid'
        
    fig.add_trace(
        go.Scatter(
            x = list(df.index), 
            y = df['BEV preferability'],
            name = sp,
            marker_color = colors[sp],
            line_color = colors[sp],
            marker_symbol='diamond',
            marker_size=10,
            line_dash=dashed,            
            )
        )
fig.update_xaxes(categoryorder='array', categoryarray= ['V0','V1','V2','V3','V4','V5','V6'])
fig.update_yaxes(title='BEV preferability')

fig.update_layout(
    template='seaborn',
    font_family="HelveticaNeue Light",
    font_size=15,
    # legend=dict(x=0.5,y=-0.1,xanchor='center',yanchor='top',orientation='h'),
    yaxis_range=[-0.05,1.05],
    )
fig.update_xaxes(type='category')

fig.write_html("plots\Figure 3 - BEVp.html",auto_open=True)

# %% Figure 4 - km vs years
time_horizon = 10


df = pd.read_excel('Final.xlsx')
df = df.fillna(method='ffill', axis=0)

df = df.sort_values(['Yearly mileage','Time Horizon'],ascending=[True,True])
df['Gasoline Price'] = df['Gasoline Price'].map({0.0756:1.4, 0.0864:1.6, 0.0972:1.8, 0.108:2, 0.1188:2.2})


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

fig.write_html(f"plots\Figure 4 - {p1},{p2},NPC.html", auto_open=True)


# %% Figure 5a - EE vs Gasoline prices
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

fig.write_html(f"plots\Figure 5a - {p1},{p2},NPC.html", auto_open=True)
fig.show()


# %% Figure 5b - EE vs Gasoline prices + PV
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

fig.write_html(f"plots\Figure 5b - {p1},{p2},NPC,PV.html", auto_open=True)
fig.show()

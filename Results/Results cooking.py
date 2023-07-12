#%%
import pandas as pd

df = pd.read_excel('Final.xlsx')
df = df.fillna(method='ffill', axis=0)
# %%
# Plot with plotly df a boxplot of Total Cost [€] by BEV installation

import plotly.express as px

df = df.sort_values(by=['Electricity price','Yearly mileage','Time Horizon'])
fig = px.violin(df,
             x="BEV installation", color= "BEV installation",
             y="Total Cost [€]", box=True,
             facet_col="Yearly mileage", 
             facet_row="Time Horizon",  hover_data=df.columns
             )

fig.update_layout(font_family="HelveticaNeue", template="plotly_white")
fig.show()
fig.write_html("Total costs.html")

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd


TH = sorted(list(set(df['Time Horizon'])))
YM = sorted(list(set(df['Yearly mileage'])))

titles = [f'Owning the vehicle {round(y)} years' for y in TH]
title = 'Total cost of ownership by yearly mileage and time horizon [EUR]'

fig = make_subplots(rows=len(TH), cols=1, shared_yaxes=True, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=titles)

for y in TH:

    if TH.index(y)==0:
        show_leg = True
    else:
        show_leg = False
    
    fig.add_trace(go.Violin(y=df['Total Cost [€]'][ df['BEV installation'] == 1][df['Time Horizon'] == y],
                        x=df['Yearly mileage'][ df['BEV installation'] == 1][df['Time Horizon'] == y],
                        legendgroup='BEV', scalegroup='BEV', name='BEV', showlegend=show_leg,
                        side='negative', hovertext=y,
                        line_color='orange'), col=1, row=TH.index(y)+1
             )
    fig.add_trace(go.Violin(y=df['Total Cost [€]'][ df['BEV installation'] == 0][df['Time Horizon'] == y],
                        x=df['Yearly mileage'][ df['BEV installation'] == 0][df['Time Horizon'] == y],
                        legendgroup='ICEV', scalegroup='ICEV', name='ICEV', showlegend=show_leg,
                        side='positive', hovertext=y,
                        line_color='red'), col=1, row=TH.index(y)+1
             )
fig.update_traces(meanline_visible=True)
# fig.update_layout(violingap=0, violinmode='overlay')

fig.update_xaxes(type='category', categoryorder='array', categoryarray=YM, title='Yearly mileage [km]')
fig.update_layout(font_family="HelveticaNeue", template="plotly_white", title=title)

# update xaxes so that values from 1250 and 17250 are not shown
# fig.update_xaxes(range=[0, 20000])

fig.write_html("Gilardio.html")
fig.show()

# %%
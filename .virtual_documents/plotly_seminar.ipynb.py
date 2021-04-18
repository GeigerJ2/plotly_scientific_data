# Relevant imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode  # Some settings for usage of Plotly in jupyter
from plotly.offline import plot
from plotly.subplots import make_subplots

from sklearn.datasets import load_wine
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

# Some further jupyter-notebook related settings
init_notebook_mode(connected=True)             # Some settings for usage of Plotly in jupyter
from IPython.display import IFrame, Image         # Display of local html-files as images
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


empty_fig = go.Figure()
print(type(empty_fig))
_ = empty_fig.show()


scatter_data = np.random.rand(10)
print(scatter_data[:])

# Create empty Figure
scatter_fig = go.Figure()

# Add Scatter-Plot
_ = scatter_fig.add_trace(     # Assignment to _ similar to redirection to /dev/null in bash (add_trace returns Figure object)
    go.Scatter(
        y = scatter_data,
#       x = list(range(len(scatter_data))),
        mode = 'markers',
    )
)

# Show Figure
scatter_fig.show()


x = np.arange(0, 4*np.pi, np.pi/100)
y = np.sin(x)

sine_fig = go.Figure()

_ = sine_fig.add_trace(
    go.Scatter(
        x = x,
        y = y,
        mode = 'lines',
    )
)

sine_fig.show()


# Define Figure layout
sine_layout = go.Layout(
    width=800, height=400,
    font=dict(size=10),
    title=dict(text='Simple sine function', x=0.55,
               font_size=20     # Magic underscore notation
              ),
    margin=go.layout.Margin(l=80, r=0, b=0, t=50,),
    xaxis=dict(title=str("x")),
    yaxis_title="sin(x)",       # Magic underscore notation
)

# Apply layout to Figure
_ = sine_fig.update_layout(sine_layout)

sine_fig.show()


# Load data
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

wine_df.head()
print("Shape of wine-dataframe: {}".format(wine_df.shape))


# Create empty Figure
wine_fig = go.Figure()

# Add Bar-chart with wine-data
_ = wine_fig.add_trace(
    go.Bar(
        y = wine_df.sort_values(by='alcohol', ascending=False).iloc[:10]['alcohol'],
    )
)

# Show Figure
wine_fig.show()


# Update layout
wine_layout = go.Layout(
    
    width=1000, height=400,
    font_size=10,
    title=dict(text='Alcohol content of 10 strongest wines in dataset', x=0.55,
               font_size=20),
#   margin=go.layout.Margin(l=80, r=0, b=0, t=50,),
    margin=dict(l=80, r=0, b=0, t=50,),
    
    xaxis=dict(
        title='rank', title_font=dict(size=18,),
        tickmode = 'linear', tick0 = 1, dtick = 1,
        tickfont=dict(size=18,),
    ),
    
    yaxis=dict(
        title=dict(
            text='% Alcohol', font=dict(size=18, color='red')
        ),
        range=[14, 15],
        tickfont=dict(size=18,),
    )
)

_ = wine_fig.update_layout(wine_layout)

print(wine_layout)

wine_fig.show()


heat_data = np.random.randn(10, 10)

heat_fig = go.Figure()

_ = heat_fig.add_trace(go.Heatmap(
    z = heat_data,),
)

heat_layout = go.Layout(
    width=600, height=500,
    title=dict(text='Basic heatmap', x=0.5, font_size=20,),
    margin=go.layout.Margin(l=80, r=0, b=0, t=50,),
)

_ = heat_fig.update_layout(heat_layout)
heat_fig.show()


# Let's create some random sample data
metals = ["Ni", "Pd", "Pt"]
energy_data = np.random.uniform(low=0, high=3, size=(10,3))
metal_colors = ["red", "green", "blue"]

# Let's set our ground states
energy_data[0,:] = 0

# Let's introduce some outliers
energy_data[-1, 0] = energy_data[-1, 0]+6
energy_data[-1, 1] = energy_data[-1, 1]+4
energy_data[-1, 2] = energy_data[-1, 2]-4

# DataFrame creation
energy_df = pd.DataFrame(data=energy_data, columns=metals)

# Define hovertexts
scatter_texts = [["ground state"]+["all good"]*8+["messed up POTCAR"],
                 ["ground state"]+["all good"]*8+["not converged"],
                 ["ground state"]+["all good"]*8+["wrong k-point sampling"]]

energy_df.shape
energy_df.head(10)


energy_fig = go.Figure()

# Zero-reference line
_ = energy_fig.add_trace(
        go.Scatter(
            x = [0, 2],
            y = [0, 0],
            mode='lines', line_color='grey', line_width=3,
            showlegend=False,
        )
)

# Iteration over metals for more individual control
for imetal, metal in enumerate(metals):
    _ = energy_fig.add_trace(
            go.Scatter(
                x = [imetal]*len(energy_df[metal]),
                y = energy_df[metal],
                mode = 'markers', marker_size=8, name=metal,
                marker_color = metal_colors[imetal],
                text = scatter_texts[imetal], 
                hoverinfo='x+y+text',
            ),
    )

# Update Figure Layout
energy_layout = go.Layout(
    width=700, height=500,
    font_size=10,
    title=dict(text='Some calculated energies', x=0.5, font_size=20),
    margin=dict(l=80, r=0, b=0, t=100,),
    
    xaxis=dict(
        title='Metal', title_font=dict(size=18,),
        tickmode = 'array', tick0 = 1, dtick = 1,
        tickfont=dict(size=18,),
        tickvals=[0, 1, 2],
        ticktext=metals,
    ),
    
    yaxis=dict(
        title=dict(text='DFT energy / eV', font_size=18),
        tickfont=dict(size=18,),
    )
)

_ = energy_fig.update_layout(energy_layout)

# Plot and save figure
_ = plot(energy_fig, filename='./dft_energies.html', auto_open=False)
_ = energy_fig.show()


md_fig = go.Figure()

# Let's create some sample MD data
md_mgn = np.random.uniform(low=0, high=0.3, size=(500,))
md_mgn[:100] += 0
md_mgn[100:200] += 1
md_mgn[200:220] += 1.5
md_mgn[220:300] += 2
md_mgn[300:400] += 1
md_mgn[400:450] += 0.5
md_mgn[450:500] += 0

# Transform magnetization into color values for plotting
mgn_min, mgn_max, mgn_mean = np.min(md_mgn), np.max(md_mgn), np.mean(md_mgn)
mgn_colors = [np.log(mgn+1) for mgn in md_mgn]
mgn_colors = [(mgn-mgn_min)/(mgn_max-mgn_min) for mgn in mgn_colors]

# Iterate over data
for imgn, mgn in enumerate(md_mgn):
    _ = md_fig.add_trace(go.Scatter(
            x=[imgn, imgn],
            y=[mgn_min, mgn_max],
            mode='lines', text=str(mgn),
            line=dict(color='rgba({}, {}, {}, {})'.format(0, 0, 255-mgn_colors[imgn]*255, 1), width=3),
            hoverinfo='x+text', showlegend=False,
            )
    )

# Add magnetization line
_ = md_fig.add_trace(go.Scatter(
        x=list(range(len(md_mgn))),
        y=md_mgn,
        mode='lines', text=md_mgn,
        line=dict(color='white', width=2, shape='spline'),
        hoverinfo='skip', showlegend=False,
        )
    )

# Add heatmap legend
color_array = np.linspace(min(mgn_colors), max(mgn_colors), 51)
color_list = ['rgba({}, {}, {}, {})'.format(0, 0, 255-color_val*255, 1) for color_val in color_array]
_ = md_fig.add_trace(go.Heatmap(
        z=[np.linspace(mgn_min, mgn_max, 51)],
        colorscale=[color for color in color_list],
        )
    )

# Add metal annotation
_ = md_fig.add_annotation(
    xanchor="left", yanchor="top",
    x=0.02*len(md_mgn), y=0.99*max(md_mgn),
    text="Pt @ 600K",
    font=dict(color='white', size=24),
    showarrow=False,
)

# Update global Figure layout
md_layout = go.Layout(
    title=dict(text="Pt-OS during MD simulation", font_size=18, x=0.5,),
    width=900, height=450, margin=dict(l=80, r=0, b=0, t=50,),
    hoverlabel = {'namelength': -1}, hovermode='x unified',
    autosize=True,
    yaxis=dict(title='OS', side='left', range=[mgn_min, mgn_max]),
    xaxis=dict(title='MD step', range=[0, len(md_mgn)],)
)

_ = md_fig.update_layout(md_layout)

# Show and save figure
_ = md_fig.show()
_ = plot(md_fig, filename='./md_os.html', auto_open=False)

del md_fig


# Let's define random data for some ML application
feature_data = np.random.uniform(low=0, high=3, size=(100,8))
feature_names = ['feature {}'.format(i) for i in range(8)]
ml_df = pd.DataFrame(data=feature_data, columns=feature_names)

ml_df['target'] = ml_df["feature 0"] - \
                2*ml_df["feature 1"] + \
                3*np.sqrt(ml_df["feature 2"]) - \
              0.1*np.exp(ml_df["feature 3"])

feature_data.shape
ml_df.head()


x = feature_data
y = ml_df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=False)

train_scaler = pp.StandardScaler().fit(x_train)
x_train_scaled = train_scaler.transform(x_train)
x_test_scaled  = train_scaler.transform(x_test)
alphas = np.linspace(start=float('1e-5'), stop=float('2'), num=1000)
coefs = []

for alpha in alphas:
    model = Lasso(alpha)
    _ = model.fit(x_train_scaled, y_train)
    coefs.append(model.coef_)

lasso_desc_fig = go.Figure()

# Plot course of descriptors
for idesc, desc in enumerate(feature_names):
    _ = lasso_desc_fig.add_trace(
    go.Scatter(
        x=alphas,
        y=[coef[idesc] for coef in coefs],
        mode='lines', line=dict(width=3), showlegend=True, name=desc,
    ),
)

lasso_desc_layout = go.Layout(
    width=1000, height=400, font=dict(size=12), margin=dict(l=0, r=0, b=0, t=50,),
    title=dict(text='LASSO - Descriptors', x=0.47,),
    hoverlabel = {'namelength': -1}, hovermode='x unified',
    xaxis=dict(title=r'$\alpha$', range=[0, max(alphas)],),
    yaxis=dict(title='Weights',),
)

_ = lasso_desc_fig.update_layout(lasso_desc_layout)

# Save and plot Figure
_ = plot(lasso_desc_fig, filename='./lasso_desc.html', auto_open=False)
_ = lasso_desc_fig.show()


train_scaler = pp.StandardScaler().fit(x)
x_scaled = train_scaler.transform(x)

# Run PCA and add results to prep_calc_df (if done for full metal set)
n_components=4
pca = PCA(n_components=n_components)
x_transformed = pca.fit_transform(x_scaled)

pca_headers = ['PC {}'.format(pci) for pci in range(x_transformed.shape[1])]

# Plot PC weights as heatmap
pca_df = pd.DataFrame(
    data=pca.components_.transpose(),
    columns=pca_headers,
).abs()


pca_fig = make_subplots(rows=2, subplot_titles=["Cumulative variance", "Weights > 0.5"],
                        vertical_spacing=0.1, row_heights=[0.1, 0.9], shared_xaxes=True,)

cumulative_variance = list(np.cumsum(pca.explained_variance_ratio_))
_ = pca_fig.add_trace(
    go.Bar(
        x=['PC '+ str(i) for i in range(n_components)],
        y=cumulative_variance,
        showlegend=False, text=[round(y,3) for y in cumulative_variance],
        textposition='auto',
    ), 
    row=1, col=1,
)

# Add weight heatmap to weight plot
_ = pca_fig.add_trace(
    go.Heatmap(
    x=['PC '+ str(i) for i in range(n_components)],
    y=feature_names,
    z=pca_df[pca_df > 0.4].values,
    xgap=1, ygap=1, colorbar=dict(thickness=20, ticklen=3),
    ),
    row=2, col=1,
)

# # Update PCA-weight layout
pca_layout = go.Layout(
    title=dict(text='Principal component analysis (PCA)', x=0.5),
    width=800, height=600, font=dict(size=12), margin=dict(l=0, r=0, b=0, t=100,),
    xaxis=dict(showgrid=True), yaxis1=dict(ticks=""),
)

_ = pca_fig.update_layout(pca_layout)

pca_fig.show()


print(scatter_fig)
scatter_fig.show("json")
# print('='*45)
# print('='*45)
# empty_fig.show("json")


print(sine_fig)

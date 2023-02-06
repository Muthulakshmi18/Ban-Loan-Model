# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:23:03 2021

@author: z027242
"""
#===============import libraies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#=============dash
import dash             #(version 1.9.1) pip install dash==1.9.1
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pathlib
# get relative data folder
#==========Read the data
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
dataset = pd.read_excel(DATA_PATH.joinpath("Bank Loan Data.xlsx"))


#=============split data
X=dataset.iloc[:,1:-1]   #==================independent variables
Y=dataset.iloc[:,-1]  #=============dependent vqariable

#====== Splitting the data - 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X , Y, test_size = 0.2, random_state = 42)

dtree = DecisionTreeClassifier(max_depth=3,min_samples_leaf=5)  

dtree.fit(X_train,y_train)

# Predicting the values of test data
dtree_y_pred = dtree.predict_proba(X_test)
dtree_y_pred1 = dtree.predict(X_test)
#=============== decision
dtree_confusion_mat=confusion_matrix(y_test,dtree_y_pred1)

dt=pd.DataFrame(dtree_confusion_mat, index=["Default","Non Default"], columns=["Default","Non Default"])
dt["Model"]="Decision Tree"
#============accuracy
dec_Accuracy=metrics.accuracy_score(y_test, dtree_y_pred1)

#=======================Random Forest
Random = RandomForestClassifier(n_estimators=250, random_state=0)

Random.fit(X_train,y_train)

#=== Predicting the values of test data
Random_y_pred = Random.predict_proba(X_test)
Random_y_pred1 = Random.predict(X_test)


#=============== Random confusion matrix
Random_confusion_mat=confusion_matrix(y_test,Random_y_pred1)

RF=pd.DataFrame(Random_confusion_mat, index=["Default","Non Default"], columns=["Default","Non Default"])
RF["Model"]="Random Forest"
#============ Random accuracy
Random_Accuracy=metrics.accuracy_score(y_test, Random_y_pred1)

log_model=LogisticRegression()
log_model.fit(X_train,y_train)

#========= Predicting the values of test data
log_y_pred = log_model.predict_proba(X_test)
log_y_pred1=log_model.predict(X_test)


#=============== Logistics accuracy
log_confusion_mat=confusion_matrix(y_test,log_y_pred1)

LR=pd.DataFrame(log_confusion_mat, index=["Default","Non Default"], columns=["Default","Non Default"])
LR["Model"]="Logistics Regression"
#============accuracy
log_Accuracy=metrics.accuracy_score(y_test, log_y_pred1)

#============confusion matrix all models
Final_con=pd.concat([dt,RF,LR])

#=================ROC curve


fpr1 , tpr1, thresholds1 = roc_curve(y_test, dtree_y_pred[:,1])  #============roc curve for decision tree

auc1 = roc_auc_score(y_test, dtree_y_pred[:,1])

fpr2 , tpr2, thresholds2 = roc_curve(y_test, Random_y_pred[:,1]) #============roc curve for Random forest

auc2 = roc_auc_score(y_test, Random_y_pred[:,1])

fpr3 , tpr3, thresholds3 = roc_curve(y_test, log_y_pred[:,1])
auc3 = roc_auc_score(y_test, log_y_pred[:,1])


 #============roc curve for Logistics Regression
auc = roc_auc_score(y_test, log_y_pred1)
#=========================dash
#==========create app
app = dash.Dash(__name__)
server=app.server
#==================title
app.layout = html.Div([
         html.H1(children="Data Visualization",  #============heading for graph
            style={
                
                'textAlign' : 'center',   #===========align the header text
                
                'color': '#0C4590'  #==============color of header text
                
                }
            
            
            
            ),
    
    
    
    html.Br(),html.Br(),
    
#=====================drop down for model    
    html.Div([
        html.Pre(children="Select the model", style={"fontSize":"150%",'font-weight': 'bold'}),
        html.Br(),
        dcc.Dropdown(
            id='my_dropdown',   #===========drop down name
            options=[
                     {'label': 'Logistics Regression', 'value': 'Logistics Regression'},   #===========value refer to column name of data frame
                     {'label': 'Random Forest', 'value': 'Random Forest'},
                     {'label': 'Decision Tree', 'value': 'Decision Tree'},
                     {'label': 'All', 'value': 'All models'},

            ],
            value='Logistics Regression',  #==========when web page loaded shows default pie chart with column of 'Animal class'
            multi=False,
            clearable=False,
            style={"width": "50%"}
        ),
    ]),
    html.Br(),html.Br(),
    html.Div([
        dcc.Graph(id='confusion_mat')   #=============graph name 
    ]),
    html.Br(),html.Br(),
    
    html.Div([
        dcc.Graph(id='the_graph')   #=============graph name 
    ]),

])

#============run Dash
#================call dash for graph
@app.callback(
 Output(component_id='the_graph', component_property='figure'),  #=========component id refers graph name
    [Input(component_id='my_dropdown', component_property='value')]  #=========component id drop down name and component_property refers "value" in drop down options
  #=========component id drop down name and component_property refers "value" in drop down options
)

def update_graph(mydropdown):  #===============refers input component id "Input(component_id='my_dropdown')"
    trace1 = go.Scatter(
    x=pd.DataFrame(fpr3)[0].to_list(),
    y=pd.DataFrame(tpr3)[0].to_list(),
    mode='lines',
    name="Logistics Regression",
     #hovertemplate=['<b>('+i+', '+str(j)+'%)</b><extra></extra>' for i,j in zip(table[column].to_list(),table["Non Default%"].to_list())],
     #text=[str(i)+"%" for i in table["Non Default%"].to_list()],
     #textposition="inside",
     #textfont=dict(
         #family="sans serif",
         #size=18,
        # color='rgb(255,255,255)'
    # )
    )   #=====================first bar
        




    trace2 = go.Scatter(
    x=pd.DataFrame(fpr2)[0].to_list(),
    y=pd.DataFrame(tpr2)[0].to_list(),
    mode='lines',
    name="Random Forest",
     #hovertemplate=['<b>('+i+', '+str(j)+'%)</b><extra></extra>' for i,j in zip(table[column].to_list(),table["Non Default%"].to_list())],
     #text=[str(i)+"%" for i in table["Non Default%"].to_list()],
     #textposition="inside",
     #textfont=dict(
         #family="sans serif",
         #size=18,
        # color='rgb(255,255,255)'
    # )
     )   #=====================first bar

    trace3 = go.Scatter(
    x=pd.DataFrame(fpr1)[0].to_list(),
    y=pd.DataFrame(tpr1)[0].to_list(),
    mode='lines',
    name="Decision Tree",
     #hovertemplate=['<b>('+i+', '+str(j)+'%)</b><extra></extra>' for i,j in zip(table[column].to_list(),table["Non Default%"].to_list())],
     #text=[str(i)+"%" for i in table["Non Default%"].to_list()],
     #textposition="inside",
     #textfont=dict(
         #family="sans serif",
         #size=18,
        # color='rgb(255,255,255)'
    # )
    )   #=====================first bar
    if (len(str(mydropdown))==0):
        data=[trace1]
        x_ax=0.5
        y_ax=1
        text="<b>AUC Score "+str(round(auc3,3))+"</b>"
        a=[dict(x=x_ax, y=y_ax,
        text=text,
        showarrow=True,
        arrowhead=1)]

    elif mydropdown=="Logistics Regression":

        data=[trace1]
        x_ax=0.5
        y_ax=1
        text="<b>AUC Score "+str(round(auc3,3))+"</b>"
        a=[dict(x=x_ax, y=y_ax,
        text=text,
        showarrow=True,
        arrowhead=1)]

    elif mydropdown=="Random Forest":

        data=[trace2]
        x_ax=0.5
        y_ax=1
        text="<b>AUC Score "+str(round(auc2,3))+"</b>"
        a=[dict(x=x_ax, y=y_ax,
        text=text,
        showarrow=True,
        arrowhead=1)]

    elif mydropdown=="Decision Tree":
        

        data=[trace3]
        x_ax=0.5
        y_ax=1
        text="<b>AUC Score "+str(round(auc1,3))+"</b>"
        a=[dict(x=x_ax, y=y_ax,
                text=text,
                showarrow=True,
                arrowhead=1)]
    else:
        
        data=[trace1,trace2,trace3]
        a=[dict(x=0.61, y=0.97,
        text="<b>AUC Score "+str(round(auc3,3))+"</b>",
        showarrow=True,
        arrowhead=1),dict(x=0.43, y=0.86,
        text="<b>AUC Score "+str(round(auc2,3))+"</b>",
        showarrow=True,
        arrowhead=1),dict(x=0.22, y=0.57,
        text="<b>AUC Score "+str(round(auc1,3))+"</b>",
        showarrow=True,
        arrowhead=1)]
    

    fig=go.Figure(
                              data=data,  #============append first and second bar
                              

                              
                              )
                          
        
    fig.update_layout(    #=========Layout
                    #==============align the title text center     
                  title={
                    'y':1,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                  
                  #============bold italics title text
                  title_text="<b><i>Bank Loan Data</i></b>",
                  
                  
                  title_font_color='rgb(169, 50, 38)',  #============title font color
                  xaxis_title_text="<b>Specificity</b>",  #===========bold xasis title
                  yaxis_title_text="<b>Sensitivity</b>",  #=========bold y axis title
                  plot_bgcolor='rgb(255, 255, 255)',  #===========graph background color
                  paper_bgcolor ='rgb(255, 255, 255)', #==============back ground color
                  font={'size': 20, 'family': 'Courier'},  #=========font size of all titles
                  font_color='rgb(23, 32, 42)',  #==============x axis and y axis color
                  
                  annotations=a
                  
                  
                  
                  
                  )
        
    fig.update_xaxes(showgrid=False)  #==============remove grid lines of x axis
    fig.update_yaxes(showgrid=False) #==============remove grid lines of x axis
        
    return (fig)

#================dynamic data table

@app.callback(
    Output(component_id='confusion_mat', component_property='figure'),  #=========component id refers graph name
     [Input(component_id='my_dropdown', component_property='value')]  #=========component id drop down name and component_property refers "value" in drop down options
)

def update_con_mat(mydropdown):
    if len(str(mydropdown))==0:
        z=log_confusion_mat
    elif mydropdown=='Logistics Regression':
        z=log_confusion_mat        
    elif mydropdown=='Random Forest':
        z=Random_confusion_mat 
    elif mydropdown=='Decision Tree':
        z=dtree_confusion_mat
    else:
        z1=log_confusion_mat
        z2=Random_confusion_mat
        z3=dtree_confusion_mat

    x=["Default","Non Default"]
    y=["Default","Non Default"]



    if (len(str(mydropdown))==0) | (mydropdown in ["Logistics Regression","Random Forest","Decision Tree"])  :
        z_text = [[str(y) for y in x] for x in z]
    
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    
        
        # add title
        fig.update_layout(title_text='<i><b>'+mydropdown+'</b></i>',
                          #xaxis = dict(title='x'),
                          #yaxis = dict(title='x')
                         )
        
        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        
        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.3,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        
        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200),height=550,width=700,)
        
        # add colorbar
        fig['data'][0]['showscale'] = True


    else:
        trace1=ff.create_annotated_heatmap(z1, x=x, y=y,  colorscale='Viridis')
        trace2=ff.create_annotated_heatmap(z2, x=x, y=y, colorscale='Viridis')
        trace3=ff.create_annotated_heatmap(z3, x=x, y=y, colorscale='Viridis')
        fig = make_subplots(rows=1, cols=3,horizontal_spacing=0.12)
    
        fig.add_trace(trace1.data[0], row=1, col=1)
        fig.add_trace(trace2.data[0], row=1, col=2)
        fig.add_trace(trace2.data[0], row=1, col=3)
        
        annot1 = list(trace1.layout.annotations)
        annot2 = list(trace2.layout.annotations)
        annot3 = list(trace3.layout.annotations)
        for k  in range(len(annot2)):
            annot2[k]['xref'] = 'x2'
            annot2[k]['yref'] = 'y2'
        for k  in range(len(annot3)):
            annot3[k]['xref'] = 'x3'
            annot3[k]['yref'] = 'y3'
                   
    
        # add title
        fig.update_layout(annotations=annot1+annot2+annot3)
        
    #=======================x title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.1,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.95,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
    
        fig.add_annotation(dict(font=dict(color="black",size=18),
                                x=0.05,
                                y=1.12,
                                showarrow=False,
                                text="Logistics Regression",
                                xref="paper",
                                yref="paper"))
        fig.add_annotation(dict(font=dict(color="black",size=18),
                                x=0.5,
                                y=1.12,
                                showarrow=False,
                                text="Random Forest",
                                xref="paper",
                                yref="paper"))
        fig.add_annotation(dict(font=dict(color="black",size=18),
                                x=0.95,
                                y=1.12,
                                showarrow=False,
                                text="Decision Tree",
                                xref="paper",
                                yref="paper"))    
    #===============================y axis
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.1,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        
            # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.3,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
            # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.7,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        
        
        
        
        
        
        
        
        
        
        
        
        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200),height=450,width=1500,)
        
        # add colorbar
        fig['data'][0]['showscale'] = True





    return(fig)

#==============display the graph

if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False)


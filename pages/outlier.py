####################################
# imports
####################################
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List
from clarifai.client.auth import create_stub
from clarifai_grpc.grpc.api.resources_pb2 import  Input
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from google.protobuf.struct_pb2 import Struct
from clarifai.client.input import Inputs
from clarifai.client.model import Model
from clarifai.client.app import App
import requests
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import umap.umap_ as umap
from stqdm import stqdm
import os

####################################
# read in auth/stub/etc from session_state
####################################

auth = ClarifaiAuthHelper.from_streamlit(st)
os.environ['CLARIFAI_PAT']=st.secrets.CLARIFAI_PAT
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
query_params = st.experimental_get_query_params()
user_id=query_params.get("user_id", [])[0]
app_id=query_params.get("app_id", [])[0]
MODEL_URL="https://clarifai.com/clarifai/main/models/multimodal-clip-embed" 

if 'user_input' not in st.session_state:
    st.session_state.user_input = None

if "user_input_2" in st.session_state:
    st.session_state.user_input_2 = None

####################################
# Functions
####################################

def list_dataset(app_id : str, user_id : str) -> List[str]:
   app = App(app_id=app_id, user_id=user_id)
   dataset_list=[]
   for dataset in list(app.list_datasets()):
      dataset_list.append(dataset.id)
   return dataset_list

def get_all_inputs(user_id : str, app_id : str, dataset_id : str) -> List[Input]:
  """Fetches all the inputs of the app.
  
  Args:
    user_id: The user id of the app.
    app_id: The app id.
    
  Returns:
    A list of inputs.
  """
  input_obj=Inputs(app_id=app_id,user_id=user_id)
  input_response=list(input_obj.list_inputs(input_type='text',
                                            dataset_id=dataset_id))
  return (input_response)

def get_embeddings_for_inputs(input_urls : list[str] ,embed_model_name : str
                              ) ->List[List[float]]:
    """Fetches the embeddings for the given inputs.

    Args:
      input_urls: A list of input urls.
      embed_model_name: The name of the embedding model.
      
    Returns:
      A list of embeddings.
    """
    input_obj=Inputs()
    model_obj=Model(url_init=embed_model_name)
    batch_size = 32
    embeddings = []
    try:
        for i in stqdm(range(0, len(input_urls), batch_size), desc="Embeddings In Progress"):
            batch = input_urls[i : i + batch_size]
            input_batch=[input_obj.get_input_from_url(input_id=str(id), text_url=inp) for id,inp in enumerate(batch)]
            predict_response = model_obj.predict(input_batch) 
            embeddings.extend(
            [
                list(output.data.embeddings[0].vector)
                for output in predict_response.outputs
            ]
        )

    except Exception as e:
        st.error(f"Predict failed, exception: {e}")

    return embeddings

def get_concepts_list(input_response_list: List[Input]) -> List[List[str]]:
    """Fetches the concepts/labels for the given inputs.

    Args:
      input_response_list: A list of input response objects.
      
    Returns:
      A list of concepts.
    """
    concepts_list=[]
    for input in input_response_list:
        concepts_list.append([concept.name for concept in input.data.concepts])
    return concepts_list

def get_umap_embedding(embedding_list : List[List[float]] , n_neighbors : int) -> np.ndarray :
    """Reduces the dimensionality of the embeddings into 3D using UMAP.

    Args:
      embedding_list: A list of embeddings.
      n_neighbors: The number of neighbors to consider.
      
    Returns:
      A ndarray of 3D embeddings.
    """
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, n_components=3, min_dist=0.2)
    reducer.fit(embedding_list)
    embedding = reducer.embedding_
    return embedding

def get_outlier_text(input_url : str) -> str:
    """Get the input text for the given outlier inputs URL.
    Args:
      input_url: The input url of texts.
    
    Returns:
      A string of the input text.
    """
    headers = {"Authorization": f"Bearer {os.environ['CLARIFAI_PAT']}"}
    response = requests.get(input_url,headers=headers )
    text = response.text
    return text

def update_inputs_metadata( input_ids : List[str], metadata : List[str],
                           user_id : str, app_id : str ) -> None:
    
    """Updates the metadata of the inputs. Based on the cluster ranking

    Args:
      input_ids: A list of all input ids.
      metadata: A list of metadata.
      user_id: The user id of the app.
      app_id: The app id.
    
    Returns:
      None
    """
    endpoint = f"https://api.clarifai.com/v2/users/{user_id}/apps/{app_id}/inputs"
    inputs = []
    for i in range(len(input_ids)):
        current_input = {
            "id": input_ids[i],
            "data": {
                "Cluster": metadata[i]
            }
        }
        inputs.append(current_input)
      
    data = {"inputs": inputs}
    headers = {
        "Authorization": f"Key {os.environ['CLARIFAI_PAT']}",
        "Content-Type": "application/json",
    }
    # Make the PATCH request
    try:
      response = requests.patch(endpoint, json=data, headers=headers)

    except Exception as e:
      st.error(f"Failed to update inputs. Exception: {e}")
  
def update_inputs_metadata_grpc( input_ids : List[str], metadata : List[str],auth) -> None:
    """Updates the metadata of the inputs. Based on the cluster ranking

    Args:
      input_ids: A list of all input ids.
      metadata: A list of metadata.
    
    Returns:
      None
    """
    stub = create_stub(auth)
    userDataObject = auth.get_user_app_id_proto()
    # Create a list of inputs
    inputs_list = []
    for i in range(len(input_ids)):
       meta=Struct()
       meta.update({"Cluster": "outlier"})
       current_input = resources_pb2.Input(
            id=input_ids[i],
            data=resources_pb2.Data(
                metadata=meta)
            )
  
       inputs_list.append(current_input)
    # Create the request
    try:
      request = stub.PatchInputs(service_pb2.PatchInputsRequest(
          user_app_id=userDataObject,
          action="overwrite",
          inputs=inputs_list
      ),
      metadata=auth.metadata
    ) 
    except Exception as e:  
      st.error(f"Failed to update inputs metadata. Exception: {e}")

####################################
# Main program
####################################

with st.form(key='clusters-app'):
    st.title('Visualize and detect outliers in your app inputs.')
    dataset_id=st.selectbox("**Select the dataset**",list_dataset(app_id=app_id ,user_id=user_id),key="dataset")
    umap_n_neighbours=st.slider('No of neighbours (Recommended above 50 for large datasets):', 2, 100)
    cluster_min_distance = st.text_input('Epsilon (min distance ):', 0.5)
    cluster_min_samples = st.slider('Min Samples:', 3, 100, 5)
    submitted = st.form_submit_button('Begin!')

if submitted:
    df=pd.DataFrame()
    ####################################
    # Get inputs and embeddings
    ####################################
    with st.spinner('Fetching inputs...'):
        input_list=get_all_inputs(user_id, app_id, dataset_id=dataset_id)
        st.text('Inputs loaded.')
    concepts=get_concepts_list(input_list)
    input_url=[input.data.text.url for input in input_list]
    input_id=[input.id for input in input_list]
    with st.spinner('Fetching embeddings...'):
        embeddings= stqdm(get_embeddings_for_inputs(input_url, MODEL_URL), desc="Embeddings In Progress")
        st.text('got embeddings')
        df=pd.DataFrame()
        df['input_id']=input_id
        df['input_url']=input_url
        df['embedding']=embeddings
        df['concepts']=concepts

    ####################################
    # get umap embeddings and convert vector into 3D vector
    ####################################
    with st.spinner('Reducing dimensions with UMAP...'):
        #st.write("umap n neighbours:", umap_n_neighbours)
        reduced_dim_list=get_umap_embedding(df["embedding"].tolist(), int(umap_n_neighbours))
        df_reduced_dim=pd.DataFrame(reduced_dim_list,columns=['x','y','z'])
        df['x']=df_reduced_dim.x.values
        df['y']=df_reduced_dim.y.values
        df['z']=df_reduced_dim.z.values

    ####################################
    # Fit DBSCAN algorithm to find the outliers
    ####################################
    with st.spinner('clustering with DBSCAN...'):
        X = StandardScaler().fit_transform(reduced_dim_list)
        dbscan = DBSCAN(eps=float(cluster_min_distance), min_samples=int(cluster_min_samples))
        df['cluster'] = dbscan.fit_predict(X)
        outliers = df[df['cluster'] == -1]
        outliers['data']=[get_outlier_text(text_url) for text_url in outliers['input_url'].tolist()]
        st.header("Outliers found")
        st.write(outliers[['input_id','data','concepts']])
        st.session_state.user_input = outliers
        
####################################
# plot the 3D scatter plot
####################################

    st.header("3D Scatter Plot")
    labels = list(np.unique(df['cluster']))
    labels=['outliers' if x == -1 else f"cluster {x}" for x in labels]
    fig = go.Figure()
    for label in labels:
        scatter = go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            mode='markers',
            marker=dict(
                size=6,
                color=df['cluster'],
                colorscale='Viridis',
                opacity=0.8
            ),
            name=label
        )
        # Add scatter plot to the figure
        fig.add_trace(scatter)
    # Customize the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis'),
        ),
        scene_camera=dict(eye=dict(x=1.87, y=0.88, z=-0.64)),
        height=1000,  # Adjust the height
        width=1200
    )
    st.plotly_chart(fig)
    st.success('Done!')

if st.button('Update metadata'):
    outliers=st.session_state.user_input
    update_inputs_metadata_grpc(input_ids=outliers['input_id'].tolist(), metadata=outliers['cluster'].tolist(), auth=auth)
    st.success("Metadata updated ")
    





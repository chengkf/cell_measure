import pickle

import cv2
import numpy as np
from dash import html, Input, Output, callback, dcc, State
from plotly.subplots import make_subplots

from segment.omnipose_segment import omnipose_segment
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from server import app
import base64
import zipfile
import io

global df

base_segment_html = html.Div([
    html.H1("图片基础分割"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            '将文件拖放到此处或',
            html.A('选取文件')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Loading(
        id="loading-1",
        children=[html.Div([
                            html.Div(id='output-image-upload',style={
                                'width': '50%',
                                'float': 'left'
                            }),
                            html.Div(id='output-outline-upload',style={
                                'width': '50%',
                                'float': 'right'
                            }),

                        ]),
            dbc.Button('处理', id='handle-photo', n_clicks=0, color='primary'),
                  ],
        type="circle",
    ),
    dcc.Loading(
        id="loading-2",
        children=[html.Div(id='handle-photo-out')],
        type="circle",
    ),
])

def parse_image(contents, filename):
    return html.Div([
        html.H5(filename, id='filename'),
        dbc.Card(
            [
                dbc.CardImg(id='out-photo', src=contents, top=True),
            ],
            #style={"width": "50%"},
        ),

    ])

def parse_outline(contents, filename):
    return html.Div([
        html.H5(filename.split(".")[0]+'_outline.jpg', id='outline_name'),
        dbc.Card(
            [
                dbc.CardImg(id='out-outline', src=contents, top=True),
            ],
            #style={"width": "50%"},
        ),

    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              prevent_initial_call=True,
          )
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_image(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children
    else:
        return html.Div([
            '请上传图片'
        ])

@app.callback(Output('handle-photo-out', 'children'),
              Output('output-outline-upload', 'children'),
              Input('handle-photo', 'n_clicks'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              prevent_initial_call=True,
          )
def handle_photo(n_clicks, list_of_contents, list_of_names):
    global df
    image = []
    if list_of_contents is not None:
        for index, list_of_contents in enumerate(list_of_contents):

            # base64_to_img
            head, context = list_of_contents.split(",")
            data = base64.b64decode(context)
            image = np.asarray(bytearray(data), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if n_clicks != 0:
        seg = omnipose_segment()
        df, imgout = seg.base_segment(image)
        #df.to_excel(data_path, index=False)
        fig = seg.draw_scatterchart(df)
        y1 = df["细胞面积"]
        y2 = df["细胞周长"]
        y3 = df["细胞长度"]
        y4 = df["细胞宽度"]
        fig_box = make_subplots(rows=1, cols=4)
        fig_box.add_trace(go.Box(y=y1, name="细胞面积"), row=1, col=1)
        fig_box.add_trace(go.Box(y=y2, name="细胞周长"), row=1, col=2)
        fig_box.add_trace(go.Box(y=y3, name="细胞长度"), row=1, col=3)
        fig_box.add_trace(go.Box(y=y4, name="细胞宽度"), row=1, col=4)

        #fig_box = seg.draw_boxchart(df, files[3], "细胞面积")
        children1 = html.Div([
            dcc.Graph(
                id="fig_image",
                figure=fig,
                style={"width": "100%"},
            ),
            dcc.Graph(
                id="fig_box",
                figure=fig_box,
                style={"width": "100%"},
            ),
            dbc.Button("下载文件", id="btn_image", n_clicks=0, color='primary'),
            dcc.Download(id="download-image"),
        ])

        # img_to_base64
        img_array = cv2.cvtColor(imgout, cv2.COLOR_RGB2BGR)
        encode_image = cv2.imencode(".jpg", img_array)[1]
        byte_data = encode_image.tobytes()
        outlines = base64.b64encode(byte_data).decode()
        outlines = 'data:image/png;base64,{}'.format(outlines)
        children2 = [
            parse_outline(c, n) for c, n in
            zip([outlines], list_of_names)]
        return children1, children2


@app.callback(
    Output("download-image", "data"),
    Input("btn_image", "n_clicks"),
    Input("filename", "children"),
    Input("out-photo", "src"),
    Input("out-outline", "src"),
    prevent_initial_call=True,
)
def func(n_clicks, filename, image, outline):
    global df
    if n_clicks != 0:
        data1 = base64.decodebytes(image.encode("utf8").split(b";base64,")[1])
        data2 = base64.decodebytes(outline.encode("utf8").split(b";base64,")[1])

        file = io.BytesIO()
        with zipfile.ZipFile(file, 'w') as myzip:
            myzip.writestr(filename, data1)
            myzip.writestr(filename.split(".")[0]+'_outline.jpg', data2)
            myzip.writestr(filename.split(".")[0]+'.csv', bytes(df.to_csv().encode('utf-8')))
            #myzip.writestr(filename.split(".")[0]+'.xlsx', pickle.dumps(df))

        zip_data = file.getvalue()

        return dcc.send_bytes(zip_data, filename.split(".")[0]+'.zip')#, dcc.send_bytes(base64.decodebytes(data2), filename.split(".")[0]+'_outline.jpg'),




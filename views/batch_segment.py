import base64
import io
import os

import cv2
import dash_bootstrap_components as dbc
import numpy as np
from dash import html, Input, Output, dcc, State
from segment.omnipose_segment import omnipose_segment
from server import app
import zipfile

batch_segment_html = html.Div([
    html.H1("图片批量处理"),
    dcc.Upload(
        id='batch-upload-data',
        children=html.Div([
            '上传压缩文件夹',
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
        multiple=True,  # 支持上传多个文件
        accept='.zip'
    ),
    dcc.Loading(
        id="loading-3",
        children=[html.Div(id='batch-output-data-upload'),
                  dbc.Button('处理', id='batch-handle-photo', n_clicks=0, color='primary'),],
        type="circle",
    ),
    dcc.Loading(
        id="loading-4",
        children=[html.Div(id='batch-handle-images-out')],
        type="default",
    ),

])

def parse_contents(contents, filename):
    return html.Div([
        html.H5(filename),
        dbc.Card(
            [
                dbc.CardImg(id='batch-out-photo', src='assets/img.png', top=True),
            ],
            style={"width": "10%"},
        ),

    ])


def read_zipfiles(path, folder=''):
    images = []
    filenames = []
    for member in path.iterdir():

        filename = os.path.join(folder, member.name)

        if member.is_file():
            image = np.asarray(bytearray(member.read_bytes()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #print(filename, ':', member.read_bytes())  # member.read_bytes()
            filenames.append(filename)
            images.append(image)

        else:
            image, filename = read_zipfiles(member, filename)
            images += image
            filenames += filename
    return images, filenames


@app.callback(Output('batch-output-data-upload', 'children'),
              Input('batch-upload-data', 'contents'),
              State('batch-upload-data', 'filename'),

              )
def batch_update_output(list_of_contents, list_of_names):
    global savepath
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]

        return children
    else:
        return html.Div([
            '请上传压缩文件夹'
        ])

@app.callback(Output('batch-handle-images-out', 'children'),
              Input('batch-handle-photo', 'n_clicks'),
              Input('batch-upload-data', 'contents'),
              State('batch-upload-data', 'filename'),
              prevent_initial_call=True,
           )
def batch_handle_photo(n_clicks, list_of_contents, list_of_names):
    global files
    if n_clicks != 0:
        seg = omnipose_segment()
        if list_of_contents is not None:
            for index, list_of_contents in enumerate(list_of_contents):
                data = base64.decodebytes(list_of_contents.encode("utf8").split(b";base64,")[1])
                #data = list_of_contents.encode("utf8").split(b";base64,")[1]

                with zipfile.ZipFile(io.BytesIO(data)) as zip_file:
                    images, filenames = read_zipfiles(zipfile.Path(zip_file))



        mean_sum, min_sum, max_sum, files = seg.batch_segment(images, filenames)
        fig_mean = seg.draw_scatterchart(mean_sum)
        fig_min = seg.draw_scatterchart(min_sum)
        fig_max = seg.draw_scatterchart(max_sum)
        fig_area = seg.draw_boxchart(files[0], files[3], "细胞面积")
        fig_per = seg.draw_boxchart(files[0], files[3], "细胞周长")
        fig_len = seg.draw_boxchart(files[0], files[3], "细胞长度")
        fig_wid = seg.draw_boxchart(files[0], files[3], "细胞宽度")
        children = html.Div([html.P('处理完成'),
                             html.P('平均值'),
                             dcc.Graph(
                                 figure=fig_mean,

                             ),
                             html.P('最小值'),
                             dcc.Graph(
                                 figure=fig_min,

                             ),
                             html.P('最大值'),
                             dcc.Graph(
                                 figure=fig_max,

                             ),
                             html.P('细胞面积'),
                             dcc.Graph(
                                 figure=fig_area,
                             ),
                             html.P('细胞周长'),
                             dcc.Graph(
                                 figure=fig_area,
                             ),
                             html.P('细胞长度'),
                             dcc.Graph(
                                 figure=fig_area,
                             ),
                             html.P('细胞宽度'),
                             dcc.Graph(
                                 figure=fig_area,
                             ),
                             dbc.Button("下载文件", id="out_zip", n_clicks=0, color='primary'),
                             dcc.Download(id="download-zip")
                             ])

        return children


@app.callback(
    Output("download-zip", "data"),
    Input("out_zip", "n_clicks"),
)
def func(n_clicks):
    global files
    if n_clicks != 0:
        file = io.BytesIO()
        with zipfile.ZipFile(file, 'w') as myzip:
            for idx in range(len(files[0])):
                filename = files[0][idx]
                image = files[1][idx]
                outline = files[2][idx]
                df = files[3][idx]

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.imencode(".jpg", image)[1]
                image = image.tobytes()

                outline = cv2.cvtColor(outline, cv2.COLOR_RGB2BGR)
                outline = cv2.imencode(".jpg", outline)[1]
                outline = outline.tobytes()


                myzip.writestr(filename, image)
                myzip.writestr(filename.split(".")[0] + '_outline.jpg', outline)
                myzip.writestr(filename.split(".")[0] + '.csv', bytes(df.to_csv().encode('utf-8')))
            mean_dir = files[4]
            min_dir = files[5]
            max_dir = files[6]
            myzip.writestr('平均值.csv', bytes(mean_dir.to_csv().encode('utf-8')))
            myzip.writestr('最小值.csv', bytes(min_dir.to_csv().encode('utf-8')))
            myzip.writestr('最大值.csv', bytes(max_dir.to_csv().encode('utf-8')))

        zip_data = file.getvalue()
        return dcc.send_bytes(zip_data, 'data.zip')




from dash import html
import dash_bootstrap_components as dbc
from server import app

home_html = html.Div(
    [
        html.P(
            [
                html.Span('欢迎来到细胞分割工具'),

                html.Span('此网站基于Omnipose算法'),

            ]
        ),
        html.P(
            [
                html.Hr(),

                html.Span('   细胞分割是指将一张包含多个细胞的图像分离出其中每个细胞的过程。 '
                          '通过细胞分割，可以获取每个细胞的形态和结构信息，帮助我们更好地了解细胞的组织结构、器官分布和生理功能。'
                          '基于Omnipose可有效地分割现有细胞图片中的细胞，但是对于一些突变的细胞分割成功率较低，本工具可先对细胞图片进行一些处理提高其分割正确率，'
                          '同时本工具可以对分割结果进行分析统计，计算分割出来的每个细胞的大小，长宽，在一定程度上识别其形态，并进行统计，为合成生物学中分析细胞突变提供数据支持。'),
            ]
        ),html.P(
            [
                html.Hr(),
            ]
        ),


        #html.Img(src=app.get_asset_url('img/百家号.png'), width=370, height=470),
        # html.Img(src=app.get_asset_url('img/1.jpg'), width=370, height=470),
    ]
)
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from views.home import home_html
from views.base_segment import base_segment_html
from views.batch_segment import batch_segment_html
from server import app

# 左侧侧边栏
left_sidebar = html.Div(
    [
        html.H2(children='细胞分割', className='display-5'),
        html.Hr(),
        dbc.Nav(
            children=[
                dbc.NavLink('工具介绍', href='/', active="exact"),
                dbc.NavLink('基础分割', href='/base_segment', active="exact", loading_state=True),
                dbc.NavLink('批量处理', href='/batch_segment', active="exact", loading_state=True),
            ],
            vertical=True,
            pills=True
        ),
    ],
    style={"background-color": "rgb(245,245,245)"},
)

# 右侧页面区域
right_content = html.Div(
    [
        html.Br(),
        html.Div(id='right-page-content'),
    ]
)

# 整体布局
app.layout = html.Div(
    [
        dcc.Location(id='url'),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(left_sidebar, width=2),
                        dbc.Col(right_content, width=10),
                    ]
                ),
            ],
            fluid=True,
        ),
    ]
)

@app.callback(Output('right-page-content', 'children'), [Input('url', 'pathname')])
def render_page_content(pathname):
    if pathname == '/':
        return home_html
    elif pathname == '/base_segment':
        return base_segment_html
    elif pathname == '/batch_segment':
        return batch_segment_html
    return html.Div(
        children=[
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className='p-3 bg-light rounded-3'
    )

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=5001, threaded=False, debug=False)

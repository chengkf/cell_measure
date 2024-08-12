from dash import Dash
import dash_bootstrap_components as dbc

'''
通过选择不同的Bootstrap和Bootswatch主题来定制您的应用程序。
dash-bootstrap-components不包含CSS。这是为了让您可以自由地使用您所选择的任何Bootstrap v5样式表, 
因此您可以在应用程序中实现您想要的外观。
选择下面的主题，查看应用程序样式的不同选项，有关主题可参考如下解释
https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/explorer/
'''

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    #external_stylesheets=[dbc.themes.CERULEAN, dbc.icons.FONT_AWESOME]
    external_stylesheets=['assets/css/bootstrap.min.css', 'assets/css/all.css']
)

# 设置网页title
app.title = 'Omnipose细胞分割'

server = app.server
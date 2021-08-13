import dash_bootstrap_components as dbc


def Navbar():
     navbar = dbc.NavbarSimple(
           children=[
#              dbc.NavItem(dbc.NavLink("France tracker", href="/france")),
              dbc.DropdownMenu(
                 nav=True,
                 in_navbar=True,
                 label="Plots",
                 children=[
                    dbc.DropdownMenuItem(dbc.NavLink("Box", href="/box")),
                    dbc.DropdownMenuItem(dbc.NavLink("PCA map", href="/pcamap"))
                          ],
                      ),
                    ],
          brand="Main page",
          brand_href="/",
          sticky="top",
        )
     return navbar

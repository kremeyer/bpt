from datetime import datetime
from datetime import timedelta
from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express.colors as pcolors
import mysql.connector
import numpy as np
from scipy.interpolate import interp1d


# LAYOUT VARIABLES
HEADER_HEIGHT = 125  # px
PLOT_BG_COLOR = "#EEEEEE"

app = Dash("Siwick Lab Info")
app.title = "Siwick Lab Info"

# DATA IMPORT
# -----------
def load_data():
    global data_lab, data_weather, dates_lab, date_set_lab, dates_weather, date_set_weather, temp_stds, temp_mins, temp_maxs
    connection = mysql.connector.connect(
        host="__", user="__", password="__"
    )
    cursor = connection.cursor()
    cursor.execute("USE logging")
    cursor.execute("SELECT * FROM AMBIENT")
    data_lab = np.array(cursor.fetchall())
    data_lab[:, 0] = [
        timestamp + datetime.now().astimezone().utcoffset()
        for timestamp in data_lab[:, 0]
    ]
    cursor.execute("SELECT * FROM WEATHER")
    data_weather = np.array(cursor.fetchall())
    data_weather[:, 0] = [
        timestamp + datetime.now().astimezone().utcoffset()
        for timestamp in data_weather[:, 0]
    ]

    dates_lab = np.array([timestamp.date() for timestamp in data_lab[:, 0]])
    date_set_lab = sorted(set(dates_lab))
    dates_weather = np.array([timestamp.date() for timestamp in data_weather[:, 0]])
    date_set_weather = sorted(set(dates_weather))

    temp_stds_lab = []
    temp_mins_lab = []
    temp_maxs_lab = []
    temp_stds_weather = []
    temp_mins_weather = []
    temp_maxs_weather = []
    for date in date_set_lab:
        temp_stds_lab.append(np.nanstd(data_lab[:, 2][dates_lab == date]))
        temp_mins_lab.append(np.nanmin(data_lab[:, 2][dates_lab == date]))
        temp_maxs_lab.append(np.nanmax(data_lab[:, 2][dates_lab == date]))
        if len(data_weather[:, 1][dates_weather == date]):
            temp_stds_weather.append(
                np.nanstd(data_weather[:, 1][dates_weather == date])
            )
            temp_mins_weather.append(
                np.nanmin(data_weather[:, 1][dates_weather == date])
            )
            temp_maxs_weather.append(
                np.nanmax(data_weather[:, 1][dates_weather == date])
            )
        else:
            temp_stds_weather.append(np.NaN)
            temp_mins_weather.append(np.NaN)
            temp_maxs_weather.append(np.NaN)
    temp_stds = np.array([temp_stds_lab, temp_stds_weather])
    temp_mins = np.array([temp_mins_lab, temp_mins_weather])
    temp_maxs = np.array([temp_maxs_lab, temp_maxs_weather])

    connection.close()


load_data()


def abs_humidity(temperature, rel_humidity):
    """calculate absolute humidity[g/m^2] from temperature[deg C] and relative humidity[%]
    from Bolton 1980: https://doi.org/10.1175/1520-0493(1980)108%3C1046:TCOEPT%3E2.0.CO;2"""
    temperature = temperature.astype(float)
    rel_humidity = rel_humidity.astype(float)
    return (
        6.112
        * np.exp((17.67 * temperature) / (243.5 + temperature))
        * rel_humidity
        * 2.1674
        / (temperature + 273.15)
    )


# SENSOR TIME SERIES PLOTS
# ------------------------
fig_lab = go.Figure()
fig_weather = go.Figure()
fig_abs_humidity = go.Figure()

# CALENDAR PLOT
# -------------
def gen_calendar():
    fig_calendar = go.Figure(
        data=go.Heatmap(
            z=temp_stds / np.array((1, 5))[:, None],
            x=date_set_lab,
            y=["Lab", "Weather"],
            zmin=0.0,
            zmax=0.5,
            customdata=np.stack((temp_stds, temp_mins, temp_maxs), axis=-1),
            colorscale="BuPu",
            showscale=False,
            hovertemplate="%{x|%A %d.%m.%Y}<br>Temperature STD: %{customdata[0]:.2f}°C<br>Min: %{customdata[1]:.1f}°C, Max: %{customdata[2]:.1f}°C<extra></extra>",
        )
    )
    fig_calendar.update_layout(
        height=HEADER_HEIGHT,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        xaxis={"fixedrange": True},
        yaxis={"fixedrange": True},
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor="white",
    )
    return fig_calendar


def gen_weekly_temp_correlation():
    last_week = date_set_lab[-7:]
    fig_weekly_temp_correlation = make_subplots(
        rows=1,
        cols=7,
        subplot_titles=[datetime.strftime(day, "%a %d.%m.%Y") for day in last_week],
    )

    for i, day in enumerate(last_week):
        mask_weather = day == dates_weather
        mask_lab = day == dates_lab
        midnight = datetime.combine(datetime.now().date() + timedelta(days=-(6-i)), datetime.min.time())

        seconds = np.array(
            [(t - midnight).total_seconds() for t in data_lab[mask_lab, 0]]
        )

        temp_weather_interp = interp1d(
            [
                (t - midnight).total_seconds()
                for t in data_weather[mask_weather, 0]
            ],
            data_weather[mask_weather, 1],
            kind="cubic",
            bounds_error=False,
            fill_value=np.NaN,
        )(seconds)

        fig_weekly_temp_correlation.add_trace(
            go.Scatter(
                x=temp_weather_interp,
                y=data_lab[mask_lab, 2],
                mode="markers",
                showlegend=False,
                marker=dict(
                    # size=,
                    cmin=0,
                    cmax=24 * 60 * 60,
                    color=seconds,
                    colorbar=dict(
                        title="time",
                        tickmode="array",
                        tickvals=np.array([0, 6, 12, 18, 24]) * 60 * 60,
                        ticktext=["00:00", "06:00", "12:00", "18:00", "00:00"],
                    ),
                    colorscale="sunsetdark",
                ),
                customdata=data_lab[mask_lab, 0],
                hovertemplate="%{customdata|%H:%M}<extra></extra><br>%{x:.2f}°C, %{y:.2f}°C",
            ),
            row=1,
            col=i + 1,
        )

        fig_weekly_temp_correlation.update_layout(
            title="last week inside/outside temperature correlation",
            yaxis_title="lab temperature [°C]",
            plot_bgcolor=PLOT_BG_COLOR,
        )
        fig_weekly_temp_correlation["layout"]["xaxis4"][
            "title"
        ] = "burnside temperature [°C]"

    return fig_weekly_temp_correlation


# APP LAYOUT AND CALLBACKS
# ------------------------
def serve_layout():
    load_data()
    fig_calendar = gen_calendar()
    fig_weekly_temp_correlation = gen_weekly_temp_correlation()
    return html.Div(
        [
            html.Div(
                [
                    dcc.DatePickerRange(
                        id="date_picker",
                        start_date=datetime.now().date() - timedelta(days=1),
                        end_date=datetime.now().date(),
                        min_date_allowed=date_set_lab[0],
                        max_date_allowed=date_set_lab[-1],
                        display_format="DD.MM.YYYY",
                        minimum_nights=0,
                        first_day_of_week=1,
                    ),
                ],
                style={
                    "width": "20%",
                    "display": "inline-block",
                    "text-align": "center",
                    "vertical-align": f"{int(HEADER_HEIGHT)/2}px",
                },
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="graph_calendar",
                        figure=fig_calendar,
                        config={"displayModeBar": False},
                    ),
                ],
                style={
                    "width": "79%",
                    "display": "inline-block",
                    "height": f"{HEADER_HEIGHT}px",
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="graph_lab", figure=fig_lab),
                    dcc.Graph(id="graph_weather", figure=fig_weather),
                    dcc.Graph(id="graph_abs_humidity", figure=fig_abs_humidity),
                ]
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="weekly_temp_correlation", figure=fig_weekly_temp_correlation
                    )
                ]
            ),
        ]
    )


app.layout = serve_layout()

# each output can only be triggered by a single callback; that's why we can't have nice code... enjoy!
@app.callback(
    [
        Output("graph_lab", "figure"),
        Output("graph_weather", "figure"),
        Output("graph_abs_humidity", "figure"),
    ],
    [Input("date_picker", "start_date"), Input("date_picker", "end_date")],
    Input("graph_calendar", "clickData"),
)
def update_time_range(*args):
    ctx = callback_context
    start_date, end_date, calendar_click = args
    load_data()
    if ctx.triggered_id == "date_picker":
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    elif ctx.triggered_id == "graph_calendar":  # calendar plot was clicked
        start_date = datetime.strptime(
            calendar_click["points"][0]["x"], "%Y-%m-%d"
        ).date()
        end_date = start_date
    else:
        start_date = datetime.now().date() - timedelta(days=1)
        end_date = datetime.now().date()

    date_mask_lab = (dates_lab >= start_date) & (dates_lab <= end_date)
    date_mask_weather = (dates_weather >= start_date) & (dates_weather <= end_date)

    # max datapoints to display = 2000; thus figure out step divider
    step_div_lab = int(len(data_lab[date_mask_lab]) / 4000) + 1
    step_div_weather = int(len(data_weather[date_mask_weather]) / 4000) + 1

    fig_lab = go.Figure()
    fig_lab = make_subplots(specs=[[{"secondary_y": True}]])
    fig_lab.add_trace(
        go.Scatter(
            x=data_lab[date_mask_lab, 0][::step_div_lab],
            y=data_lab[date_mask_lab, 1][::step_div_lab],
            name="PCB temperature",
            mode="lines+markers",
            line_color="black",
            visible="legendonly",
        )
    )
    fig_lab.add_trace(
        go.Scatter(
            x=data_lab[date_mask_lab, 0][::step_div_lab],
            y=data_lab[date_mask_lab, 2][::step_div_lab],
            name="finger temperature",
            mode="lines+markers",
            line_color="crimson",
        )
    )
    fig_lab.add_trace(
        go.Scatter(
            x=data_lab[date_mask_lab, 0][::step_div_lab],
            y=data_lab[date_mask_lab, 3][::step_div_lab],
            name="humidity",
            mode="lines+markers",
            line_color="darkcyan",
        ),
        secondary_y=True,
    )
    fig_lab.update_layout(
        title="OMC027", xaxis_title="timestamp", yaxis_title="temperature [°C]",
        plot_bgcolor=PLOT_BG_COLOR,
    )
    fig_lab.update_yaxes(title_text="relative humidity [%]", secondary_y=True)

    fig_weather = go.Figure()
    fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
    fig_weather.add_trace(
        go.Scatter(
            x=data_weather[date_mask_weather, 0][::step_div_weather],
            y=data_weather[date_mask_weather, 1][::step_div_weather],
            name="temperature",
            mode="lines+markers",
            line_color="crimson",
        )
    )
    fig_weather.add_trace(
        go.Scatter(
            x=data_weather[date_mask_weather, 0][::step_div_weather],
            y=data_weather[date_mask_weather, 2][::step_div_weather],
            name="humidity",
            mode="lines+markers",
            line_color="darkcyan",
        ),
        secondary_y=True,
    )
    fig_weather.update_layout(
        title="burnside weather station",
        xaxis_title="timestamp",
        yaxis_title="temperature [°C]",
        plot_bgcolor=PLOT_BG_COLOR,
    )
    fig_weather.update_yaxes(title_text="relative humidity [%]", secondary_y=True)

    fig_abs_humidity = go.Figure()
    fig_abs_humidity.add_trace(
        go.Scatter(
            x=data_lab[date_mask_lab, 0][::step_div_lab],
            y=abs_humidity(
                data_lab[date_mask_lab, 2][::step_div_lab],
                data_lab[date_mask_lab, 3][::step_div_lab],
            ),
            name="lab",
            mode="lines+markers",
            line_color="crimson",
        )
    )
    fig_abs_humidity.add_trace(
        go.Scatter(
            x=data_weather[date_mask_weather, 0][::step_div_weather],
            y=abs_humidity(
                data_weather[date_mask_weather, 1][::step_div_weather],
                data_weather[date_mask_weather, 2][::step_div_weather],
            ),
            name="weather",
            mode="lines+markers",
            line_color="darkcyan",
        )
    )
    fig_abs_humidity.update_layout(
        title="absolute humidity",
        xaxis_title="timestamp",
        yaxis_title="absolute humidity [g/m^3]",
        plot_bgcolor=PLOT_BG_COLOR,
    )

    return fig_lab, fig_weather, fig_abs_humidity


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)

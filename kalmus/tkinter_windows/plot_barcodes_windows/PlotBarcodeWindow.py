"""
PlotBarcodeWindow Class
ColorHistogramWindow Class
RGBColorCubeWindow Class
OutputCSVWindow Class
"""

import os
import tkinter
import tkinter.filedialog
from tkinter.messagebox import showinfo, showerror

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from skimage.color import rgb2hsv

from kalmus.tkinter_windows.gui_utils import resource_path, update_hist
from kalmus.utils.visualization_utils import show_colors_in_cube, show_colors_in_hue_light_scatter_plot, \
    show_colors_in_hue_light_3d_bar_plot


class PlotBarcodeWindow():
    """
    PlotBarcodeWindow Class
    GUI window of plotting the barcode for user to inspect in details
    """

    def __init__(self, barcode, figsize=(6, 4), dpi=100):
        """
        Initialize

        :param barcode: The barcode that will be inspected
        :param figsize: The size of the plotted figure
        :param dpi: The dpi of the figure
        """
        self.barcode = barcode

        # Initialize the window
        self.plot_window = tkinter.Tk()
        self.plot_window.wm_title("Inspect Barcode")
        self.plot_window.iconbitmap(resource_path("kalmus_icon.ico"))

        # Set up the plotted figure
        self.fig = plt.figure(figsize=figsize, dpi=dpi)

        # Use the correct color map based on the input barcode's type
        if barcode.barcode_type == "Brightness":
            plt.imshow(barcode.get_barcode().astype("uint8"), cmap="gray")
        else:
            plt.imshow(barcode.get_barcode().astype("uint8"))
        plt.axis("off")
        plt.tight_layout()

        # Set up the canvas for the figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_window)  # A tk.DrawingArea.
        self.canvas.draw()

        # Dynamic layout based on the type of the inspected barcode
        if barcode.barcode_type == "color":
            column_span = 5
        else:
            column_span = 2
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=column_span, pady=3)

        # Use tkinter Frame to organize the figure widget
        toolbarFrame = tkinter.Frame(master=self.plot_window)
        toolbarFrame.grid(row=2, column=0, columnspan=column_span, sticky=tkinter.W, pady=6)

        # Initialize the plotting tool bar
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        self.toolbar.update()

        # Button to output the data in the barcode to a csv file
        self.button_output_csv = tkinter.Button(master=self.plot_window, text="Output CSV",
                                                command=self.output_csv)
        self.button_output_csv.grid(row=1, column=0, padx=6)

        # Button to check the histogram distribution of the barcode's hue/brightness value
        self.button_hist = tkinter.Button(master=self.plot_window, text="Hue Histogram",
                                          command=self.show_color_histogram)
        self.button_hist.grid(row=1, column=1, padx=6)

        # If the barcode is a color barcode, allow user to inspect the RGB color distribution in a RGB cube
        if barcode.barcode_type == "color":
            self.button_cube = tkinter.Button(master=self.plot_window, text="Colors in RGB Cube",
                                              command=self.show_RGB_color_in_cube)
            self.button_cube.grid(row=1, column=2)

            self.button_scatter = tkinter.Button(master=self.plot_window, text="Hue vs. Light Scatter",
                                                 command=self.show_color_in_scatter)
            self.button_scatter.grid(row=1, column=3)

            self.button_bar = tkinter.Button(master=self.plot_window, text="Hue vs. Light Bar Plot",
                                             command=self.show_color_in_bar)
            self.button_bar.grid(row=1, column=4)

    def show_RGB_color_in_cube(self):
        """
        Instantiate the RGBColorCubeWindow if user press the show color in RGB cube button
        """
        RGBColorCubeWindow(self.barcode)

    def show_color_in_scatter(self):
        """
        Instantiate the LightHueScatterPlotWindow if user press the show color in hue light scatter button
        """
        HueLightScatterPlotWindow(self.barcode)

    def show_color_in_bar(self):
        """
        Instantiate the HueBrightness3DBarPlotWindow if user press the show color in hue light scatter button
        """
        HueLight3DBarPlotWindow(self.barcode)

    def show_color_histogram(self):
        """
        Instantiate the ColorHistogramWindow if user press the show histogram button
        """
        ColorHistogramWindow(self.barcode)

    def output_csv(self):
        OutputCSVWindow(self.barcode)


class ColorHistogramWindow():
    """
    ColorHistogramWindow Class
    GUI window that show the distribution of the barcode's hue[0, 360]/brightness[0, 255] value
    """

    def __init__(self, barcode):
        """
        Initialize

        :param barcode: The input barcode
        """
        # Set up the window
        self.window = tkinter.Tk()
        self.window.wm_title("Histogram Distribution")
        self.window.iconbitmap(resource_path("kalmus_icon.ico"))

        # Set up the plotted figure
        fig, ax = plt.subplots(figsize=(9, 5))

        update_hist(barcode, ax=ax, bin_step=5)

        # Plot the histogram based on the barcode's type
        if barcode.barcode_type == "color":
            ax.set_xticks(np.arange(0, 361, 30))
            ax.set_xlabel("Color Hue (0 - 360)")
            ax.set_ylabel("Number of frames")
        else:
            ax.set_xticks(np.arange(0, 255, 15))
            ax.set_xlabel("Brightness (0 - 255)")
            ax.set_ylabel("Number of frames")

        # Set up the canvas of the figure
        canvas = FigureCanvasTkAgg(fig, master=self.window)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Set up the tool bar of the figure
        toolbar = NavigationToolbar2Tk(canvas, self.window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


class RGBColorCubeWindow():
    """
    RGBColorCubeWindow Class
    GUI window that shows the distribution of the barcode's RGB color in a RGB cube
    range in [0, 255] for all three channels
    """

    def __init__(self, barcode):
        """
        Initialize

        :param barcode: The input barcode
        """
        self.barcode = barcode

        # Set up the window
        self.window = tkinter.Tk()
        self.window.wm_title("Colors in RGB cube")
        self.window.iconbitmap(resource_path("kalmus_icon.ico"))

        # Set up the plotted figure
        sampling = 6000
        if sampling > self.barcode.colors.shape[0]:
            sampling = self.barcode.colors.shape[0]
        fig, ax = show_colors_in_cube(self.barcode.colors, return_figure=True,
                                      figure_size=(6.5, 6.5), sampling=sampling)

        # Set up the canvas
        canvas = FigureCanvasTkAgg(fig, master=self.window)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Allow mouse events on 3D figure
        ax.mouse_init()

        # Set up the tool bar of the figure
        toolbar = NavigationToolbar2Tk(canvas, self.window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


class HueLight3DBarPlotWindow():
    """
    HueLight3DBarPlotWindow Class
    GUI window that shows the distribution of the barcode's color in a Hue (x-axis) vs. Light (y-axis)
    vs. Counts/Frequency (z-axis) 3D bar plot. The color of the barcode will be converted from RGB to
    HSV/HSL color space. Hue ranges from 0 to 360 degree and Light range from 0 to 1 (darkest to the brightest)
    """

    def __init__(self, barcode, figure_size=(6.5, 6.5)):
        """
        Initialize

        :param barcode: The input barcode
        :param figure_size: The size of the plotted figure
        """
        self.barcode = barcode

        # Set up the window
        self.window = tkinter.Tk()
        self.window.wm_title("Colors in Hue Light 3D Bar Plot")
        self.window.iconbitmap(resource_path("kalmus_icon.ico"))

        self.figure_size = figure_size

        # Set up the plotted figure
        self.fig, self.ax = show_colors_in_hue_light_3d_bar_plot(self.barcode.colors,
                                                                 figure_size=self.figure_size,
                                                                 return_figure=True,
                                                                 shaded=False,
                                                                 invert_light_axis=True)

        # Set up the canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=1, pady=3, rowspan=14)

        # Use tkinter Frame to organize the figure widget
        toolbarFrame = tkinter.Frame(master=self.window)
        toolbarFrame.grid(row=14, column=0, columnspan=1, sticky=tkinter.W, pady=6, rowspan=1)

        # Initialize the plotting tool bar
        toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        toolbar.update()

        # Allow mouse events on 3D figure
        self.ax.mouse_init()

        # Shaded/Unshaded option
        self.config_shade = tkinter.IntVar(self.window)
        # self.config_shade.set("Unshaded")  # initialize
        self.config_shade.set(0)

        label_config = tkinter.Label(self.window, text="Plot Configurations:")
        label_config.grid(row=0, column=1, columnspan=2)

        self.checkbox_shade_object = tkinter.Checkbutton(self.window, text="Shade 3D Bar Objects",
                                                         variable=self.config_shade,
                                                         onvalue=1, offvalue=0, command=self.shade_bar_plot)
        self.checkbox_shade_object.grid(row=1, column=1, sticky="nw", padx=(15, 0), columnspan=2)

        # Grid on/off option
        self.config_grid = tkinter.IntVar(self.window)
        # self.config_shade.set("Unshaded")  # initialize
        self.config_grid.set(0)

        self.checkbox_grid = tkinter.Checkbutton(self.window, text="Turn on Grid",
                                                 variable=self.config_grid,
                                                 onvalue=1, offvalue=0, command=self.turn_on_plot_grid)
        self.checkbox_grid.grid(row=2, column=1, sticky="nw", padx=(15, 0), columnspan=2)

        # Axis on/off option
        self.config_axis = tkinter.IntVar(self.window)
        # self.config_shade.set("Unshaded")  # initialize
        self.config_axis.set(1)

        self.checkbox_axis = tkinter.Checkbutton(self.window, text="Turn on Axis",
                                                 variable=self.config_axis,
                                                 onvalue=1, offvalue=0, command=self.turn_on_plot_axis)
        self.checkbox_axis.grid(row=3, column=1, sticky="nw", padx=(15, 0), columnspan=2)

        label_resolution = tkinter.Label(self.window, text="Plot Resolutions:\n"
                                                           "Changes to resolution will be effective\n"
                                                           " after clicking the Submit Changes Button")
        label_resolution.grid(row=4, column=1, sticky="w", padx=5, columnspan=2)

        self.label_hue_resolution = tkinter.Label(self.window, text="Bar width on Hue axis:")
        self.label_hue_resolution.grid(row=5, column=1, sticky="nw", padx=5)

        # Variable that stores the user's choice of Hue resolution
        self.var_hue_resolution = tkinter.StringVar(self.window)
        self.var_hue_resolution.set("10")
        self.cur_hue_res = 10

        # Dropdown menu for the Hue resolution selection
        dropdown_hue_type = tkinter.OptionMenu(self.window, self.var_hue_resolution, "5", "10", "15", "30")
        dropdown_hue_type.grid(row=5, column=2, sticky="nw", padx=5)

        self.label_hue_resolution = tkinter.Label(self.window, text="Bar width on Light axis:")
        self.label_hue_resolution.grid(row=6, column=1, sticky="nw", padx=5)

        # Variable that stores the user's choice of Brightness resolution
        self.var_bri_resolution = tkinter.StringVar(self.window)
        self.var_bri_resolution.set("0.02")
        self.cur_bri_res = 0.02

        # Dropdown menu for the Light resolution selection
        dropdown_bri_type = tkinter.OptionMenu(self.window, self.var_bri_resolution, "0.01", "0.02", "0.05", "0.10")
        dropdown_bri_type.grid(row=6, column=2, sticky="nw", padx=5)

        # Button to submit the changes
        self.button_submit = tkinter.Button(master=self.window, text="Submit Changes",
                                            command=self.generate_new_plot_with_changes)
        self.button_submit.grid(row=7, column=1, columnspan=2, sticky="n")

        label_camera = tkinter.Label(self.window, text="Camera Views:")
        label_camera.grid(row=8, column=1, columnspan=2)

        # Variable that stores the Camera view selected
        self.var_view_option = tkinter.StringVar(self.window)
        self.var_view_option.set("Diag View 1")  # initialize

        # Stores the original x, y, z limits
        self.original_xlim = self.ax.get_xlim3d()
        self.original_ylim = self.ax.get_ylim3d()
        self.original_zlim = self.ax.get_zlim3d()

        # Radio button for the Camera view selection
        self.radio_hue_light_view_1 = tkinter.Radiobutton(self.window, text="Diag View 1",
                                                          variable=self.var_view_option,
                                                          value="Diag View 1",
                                                          command=self.change_view)
        self.radio_hue_light_view_1.grid(row=9, column=1, sticky="nw")
        self.radio_hue_light_view_1.select()

        self.radio_hue_light_view_2 = tkinter.Radiobutton(self.window, text="Diag View 2",
                                                          variable=self.var_view_option,
                                                          value="Diag View 2",
                                                          command=self.change_view)
        self.radio_hue_light_view_2.grid(row=9, column=2, sticky="nw")

        self.radio_hue_view_1 = tkinter.Radiobutton(self.window, text="Hue View 1",
                                                    variable=self.var_view_option,
                                                    value="Hue View 1",
                                                    command=self.change_view)
        self.radio_hue_view_1.grid(row=10, column=1, sticky="nw")

        self.radio_hue_view_2 = tkinter.Radiobutton(self.window, text="Hue View 2",
                                                    variable=self.var_view_option,
                                                    value="Hue View 2",
                                                    command=self.change_view)
        self.radio_hue_view_2.grid(row=10, column=2, sticky="nw")

        self.radio_light_view_1 = tkinter.Radiobutton(self.window, text="Light View 1",
                                                      variable=self.var_view_option,
                                                      value="Light View 1",
                                                      command=self.change_view)
        self.radio_light_view_1.grid(row=11, column=1, sticky="nw")

        self.radio_light_view_2 = tkinter.Radiobutton(self.window, text="Light View 2",
                                                      variable=self.var_view_option,
                                                      value="Light View 2",
                                                      command=self.change_view)
        self.radio_light_view_2.grid(row=11, column=2, sticky="nw")

        self.radio_top_view = tkinter.Radiobutton(self.window, text="Top View",
                                                  variable=self.var_view_option,
                                                  value="Top View",
                                                  command=self.change_view)
        self.radio_top_view.grid(row=12, column=1, columnspan=2, sticky="n")

        label_rotation_speed = tkinter.Label(self.window, text="Rotation Speed (Degrees):")
        label_rotation_speed.grid(row=13, column=1, columnspan=2, sticky="s")

        self.var_rotation_speed = tkinter.IntVar(self.window)
        self.var_rotation_speed.set(5)
        self.rotation_speed = self.var_rotation_speed.get()
        self.slider_rotation_speed = tkinter.Scale(
            self.window,
            from_=1,
            to=30,
            orient='horizontal',
            variable=self.var_rotation_speed,
            command=self.update_rotation_speed,
        )
        self.slider_rotation_speed.grid(row=14, column=1, columnspan=2, sticky="n")

        self.window.bind("<Left>", self.rotate_left)
        self.window.bind("<Right>", self.rotate_right)
        self.window.bind("<Up>", self.rotate_up)
        self.window.bind("<Down>", self.rotate_down)

        self.window.lift()

    def update_rotation_speed(self, event):
        self.rotation_speed = self.var_rotation_speed.get()

    def rotate_left(self, event):
        cur_elev = self.ax.elev
        cur_azim = self.ax.azim

        cur_azim += self.rotation_speed
        if cur_azim >= 180:
            cur_azim -= 360

        self.ax.view_init(elev=cur_elev, azim=cur_azim)
        self.canvas.draw()

    def rotate_right(self, event):
        cur_elev = self.ax.elev
        cur_azim = self.ax.azim

        cur_azim -= self.rotation_speed
        if cur_azim < -180:
            cur_azim += 360

        self.ax.view_init(elev=cur_elev, azim=cur_azim)
        self.canvas.draw()

    def rotate_up(self, event):
        cur_elev = self.ax.elev
        cur_azim = self.ax.azim

        cur_elev -= self.rotation_speed
        if cur_elev < -180:
            cur_elev += 360

        self.ax.view_init(elev=cur_elev, azim=cur_azim)
        self.canvas.draw()

    def rotate_down(self, event):
        cur_elev = self.ax.elev
        cur_azim = self.ax.azim

        cur_elev += self.rotation_speed
        if cur_elev >= 180:
            cur_elev -= 360

        self.ax.view_init(elev=cur_elev, azim=cur_azim)
        self.canvas.draw()

    def change_view(self):
        view_selected = self.var_view_option.get()
        if view_selected == "Diag View 1":
            self.ax.view_init(elev=30, azim=-60)
        elif view_selected == "Diag View 2":
            self.ax.view_init(elev=30, azim=120)
        elif view_selected == "Hue View 1":
            self.ax.view_init(elev=0, azim=-90)
        elif view_selected == "Hue View 2":
            self.ax.view_init(elev=0, azim=90)
        elif view_selected == "Light View 1":
            self.ax.view_init(elev=0, azim=0)
        elif view_selected == "Light View 2":
            self.ax.view_init(elev=0, azim=180)
        elif view_selected == "Top View":
            self.ax.view_init(elev=93, azim=-90)

        self.ax.set_xlim3d(self.original_xlim[0], self.original_xlim[1])
        self.ax.set_ylim3d(self.original_ylim[0], self.original_ylim[1])
        self.ax.set_zlim3d(self.original_zlim[0], self.original_zlim[1])
        self.canvas.draw()

    def generate_new_plot_with_changes(self):
        if self.config_grid.get() == 0:
            grid_on = False
        else:
            grid_on = True

        if self.config_shade.get() == 0:
            shade_on = False
        else:
            shade_on = True

        self.cur_hue_res = int(self.var_hue_resolution.get())
        self.cur_bri_res = float(self.var_bri_resolution.get())

        self.ax.cla()
        show_colors_in_hue_light_3d_bar_plot(self.barcode.colors,
                                             hue_resolution=self.cur_hue_res,
                                             bri_resolution=self.cur_bri_res,
                                             figure_size=self.figure_size,
                                             return_figure=True,
                                             shaded=shade_on,
                                             grid_off=not grid_on,
                                             axes=self.ax,
                                             invert_light_axis=True)

        if self.config_axis.get() == 0:
            self.ax.axis("off")

        # Stores the original x, y, z limits
        self.original_xlim = self.ax.get_xlim3d()
        self.original_ylim = self.ax.get_ylim3d()
        self.original_zlim = self.ax.get_zlim3d()

        self.canvas.draw()

    def turn_on_plot_axis(self):
        if self.config_axis.get() == 0:
            self.checkbox_grid.config(state="disabled")
            self.ax.axis("off")
            self.canvas.draw()
        elif self.config_axis.get() == 1:
            self.checkbox_grid.config(state="normal")
            self.ax.axis("on")
            self.canvas.draw()

    def turn_on_plot_grid(self):
        if self.config_grid.get() == 0:
            self.ax.grid(False)
            self.canvas.draw()
        elif self.config_grid.get() == 1:
            self.ax.grid(True)
            self.canvas.draw()

    def shade_bar_plot(self):
        if self.config_grid.get() == 0:
            grid_on = False
        else:
            grid_on = True

        if self.config_shade.get() == 0:
            shade_on = False
        else:
            shade_on = True

        if self.config_shade.get() == 0:
            self.ax.cla()
            show_colors_in_hue_light_3d_bar_plot(self.barcode.colors,
                                                 figure_size=self.figure_size,
                                                 hue_resolution=self.cur_hue_res,
                                                 bri_resolution=self.cur_bri_res,
                                                 return_figure=True,
                                                 shaded=shade_on,
                                                 grid_off=not grid_on,
                                                 axes=self.ax,
                                                 invert_light_axis=True)
        elif self.config_shade.get() == 1:
            self.ax.cla()
            show_colors_in_hue_light_3d_bar_plot(self.barcode.colors,
                                                 figure_size=self.figure_size,
                                                 hue_resolution=self.cur_hue_res,
                                                 bri_resolution=self.cur_bri_res,
                                                 return_figure=True,
                                                 shaded=shade_on,
                                                 grid_off=not grid_on,
                                                 axes=self.ax,
                                                 invert_light_axis=True)

        if self.config_axis.get() == 0:
            self.ax.axis("off")

        self.canvas.draw()


class HueLightScatterPlotWindow():
    """
    HueLightScatterPlotWindow Class
    GUI window that shows the distribution of the barcode's color in a Hue (x-axis) vs. Light (y-axis)
    scatter plot. The color of the barcode will be converted from RGB to HSV/HSL color space.
    Hue ranges from 0 to 360 degree and light range from 0 to 1 (darkest to the brightest)
    """

    def __init__(self, barcode):
        """
        Initialize

        :param barcode: The input barcode
        """
        self.barcode = barcode

        # Set up the window
        self.window = tkinter.Tk()
        self.window.wm_title("Colors in Hue Light Scatter Plot")
        self.window.iconbitmap(resource_path("kalmus_icon.ico"))

        saturation_threshold = 0.15
        # Set up the plotted figure
        fig, ax = show_colors_in_hue_light_scatter_plot(self.barcode.colors, figure_size=(9, 4.5),
                                                        return_figure=True, remove_border=True,
                                                        saturation_threshold=saturation_threshold)
        frame_type = barcode.frame_type
        frame_type = frame_type.replace("_", " ")
        frame_type = frame_type.title()

        ax.set_title("{:s} Color of {:s} (Only colors with saturation > {:.2f} are included)"
                     .format(barcode.color_metric, frame_type, saturation_threshold))
        plt.tight_layout()

        # Set up the canvas
        canvas = FigureCanvasTkAgg(fig, master=self.window)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Set up the tool bar of the figure
        toolbar = NavigationToolbar2Tk(canvas, self.window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


class OutputCSVWindow():
    """
    OutputCSVWindow class
    GUI window that outputs the per frame level color/brightness data of the inspected barcode
    The data output are stored in the csv file, and the data frame depends on the type of the barcode
    """

    def __init__(self, barcode):
        """
        Initialize

        :param barcode: The barcode to output the per frame level data
        """
        self.barcode = barcode

        # Set up the window
        self.window = tkinter.Tk()
        self.window.wm_title("Output the barcode to CSV")
        self.window.iconbitmap(resource_path("kalmus_icon.ico"))

        # Label prompt for the file name/path to the csv file
        filename_label = tkinter.Label(self.window, text="CSV file path: ")
        filename_label.grid(row=0, column=0, sticky=tkinter.W)

        # Text entry for user to type the file name/path to the csv file
        self.filename_entry = tkinter.Entry(self.window, textvariable="", width=40)
        self.filename_entry.grid(row=0, column=1, columnspan=1, sticky=tkinter.W)

        # Button to browse the folder
        self.button_browse_folder = tkinter.Button(self.window, text="Browse", command=self.browse_folder)
        self.button_browse_folder.grid(row=0, column=2)

        # Button to build/load the barcode using the given json file
        self.button_build_barcode = tkinter.Button(self.window, text="Output", command=self.output_csv_file)
        self.button_build_barcode.grid(row=1, column=1, columnspan=1)

    def output_csv_file(self):
        """
        Output the per frame level data to a csv file
        """
        # Get the file name of the output csv file
        csv_filename = self.filename_entry.get()

        if len(csv_filename) == 0:
            showerror("Invalid Path or Filename", "Please specify the path/filename of the generated csv file.\n")
            return

        # Get the sampled frame rate of the barcode
        sample_rate = self.barcode.sampled_frame_rate

        # Get the starting/skipped over frame of the barcode
        starting_frame = self.barcode.skip_over

        # Generate the corresponding csv file for the type of the barcode
        if self.barcode.barcode_type == 'Color':
            # Data frame of the csv file for the color barcode
            colors = self.barcode.colors
            hsvs = rgb2hsv(colors.reshape(-1, 1, 3).astype("float64") / 255)
            hsvs[..., 0] = 360 * hsvs[..., 0]
            colors = colors.astype("float64")
            brightness = 0.299 * colors[..., 0] + 0.587 * colors[..., 1] + 0.114 * colors[..., 1]

            colors = colors.astype("uint8")
            hsvs = hsvs.reshape(-1, 3)
            brightness = brightness.astype("int64")

            frame_indexes = np.arange(starting_frame, len(colors) * sample_rate + starting_frame, sample_rate)

            dataframe = pd.DataFrame(data={'Frame index': frame_indexes,
                                           'Red (0-255)': colors[..., 0],
                                           'Green (0-255)': colors[..., 1],
                                           'Blue (0-255)': colors[..., 2],
                                           'Hue (0 -360)': (hsvs[..., 0]).astype("int64"),
                                           'Saturation (0 - 1)': hsvs[..., 1],
                                           'Value (lightness) (0 - 1)': hsvs[..., 2],
                                           'Brightness': brightness})

        elif self.barcode.barcode_type == 'Brightness':
            # Data frame of the csv file for the brightness barcode
            brightness = self.barcode.brightness

            frame_indexes = np.arange(starting_frame, len(brightness) * sample_rate + starting_frame, sample_rate)
            # Get the per frame level brightness data
            dataframe = pd.DataFrame(data={'Frame index': frame_indexes,
                                           'Brightness': brightness.astype("uint8").reshape(-1)})

        dataframe = dataframe.set_index('Frame index')

        if not csv_filename.endswith(".csv"):
            csv_filename += ".csv"

        dataframe.to_csv(csv_filename)

        # Quit the window after outputting csv file
        self.window.destroy()

        showinfo("CSV File Generated Successfully", "CSV file has been generated successfully.\n"
                                                    "Path to the File: {:20s}".format(os.path.abspath(csv_filename)))

    def browse_folder(self):
        """
        Browse the folder to locate the json file
        """
        # Get the file name from the user selection
        filename = tkinter.filedialog.asksaveasfilename(initialdir=".", title="Select CSV file",
                                                        filetypes=(("csv files", "*.csv"), ("txt files", "*.txt"),
                                                                   ("All files", "*.*")))

        # Update the file name to the file name text entry
        self.filename_entry.delete(0, tkinter.END)
        self.filename_entry.insert(0, filename)

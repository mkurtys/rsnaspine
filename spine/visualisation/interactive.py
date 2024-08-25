from IPython.display import display
import ipywidgets as widgets
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.patches import Circle, Rectangle, Wedge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spine.spine_exam import InstanceMeta, SpineStudy, SpineSeries
from spine.transforms import patient_coords_to_image_2d
from typing import Optional, Tuple, Union



# draw circle,rectangle, wedge depending on the type of condition
def _choose_patch( xy: tuple, condition: str, color: str, radius: float):
    x,y = xy
    if condition == 'subarticular_stenosis':
        return Circle((x, y), color=color, radius=radius)
    elif condition == 'neural_foraminal_narrowing':
        # rectangle centered at x,y
        return Rectangle((x-radius/2, y-radius/2), width=radius, height=radius, color=color)
    elif condition == 'spinal_canal_stenosis':
        return Wedge((x, y), color=color, r=radius, theta1=45, theta2=45+60)
    else:
        raise ValueError(f'Invalid condition: {condition}')
        

def simple_interactive( axial_series: SpineSeries,
                        sagittal_series: SpineSeries, 
                       descriptions: Optional[pd.DataFrame]=None):
    axial = axial_series.volume
    sagittal = sagittal_series.volume

    axial_meta = axial_series.meta
    sagittal_meta = sagittal_series.meta

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.tight_layout()
    ax_img = ax[0].imshow(axial[0,:,:], cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Axial')
    ax[0].set_ylim(axial.shape[1], 0)
    ax[0].set_xlim(0, axial.shape[2])

    # Show the specified slice on the sagittal plane with 'gray' color-map and sagittal aspect ratio
    sag_img = ax[1].imshow(sagittal[0,:,:,], cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Sagittal')
    ax[1].set_ylim(sagittal.shape[1], 0)
    ax[1].set_xlim(0, sagittal.shape[2])
    lines = []
    points = []

    def simple_interactive_inner(axial_slice, sagittal_slice, layout, draw_points, 
                                 draw_points_opts, draw_all_reference_lines):
        ax_img.set_array(axial[axial_slice,:,:])
        sag_img.set_array(sagittal[sagittal_slice,:,:])
        for line in lines:
            line.remove()
        lines.clear()
        for point in points:
            point.remove()
        points.clear()

        # print(f" slice thickness: {axial_meta[axial_slice].slice_thickness} {sagittal_meta[sagittal_slice].slice_thickness}")
        # print(f" spacing between slices: {axial_meta[axial_slice].spacing_between_slices} {sagittal_meta[sagittal_slice].spacing_between_slices}")

        if draw_points:
            value_to_color = {"Normal/Mild": "green", "Moderate": "yellow", "Severe": "red"}
            if draw_points_opts == "Listed":
                for i, (series_id, instance_number) in enumerate(zip([axial_series.series_id, sagittal_series.series_id], [axial_slice, sagittal_slice])):
                    valid_descriptions = descriptions[(descriptions["series_id"] == series_id) & (descriptions["instance_number"] == instance_number) ]
                    for idx, row in valid_descriptions.iterrows():
                        points.append(ax[i].add_patch(plt.Circle((row["x"], row["y"]), color=value_to_color[row["value"]], radius=3)))
            else:
                for i, (series, instance_number) in enumerate(zip([axial_series, sagittal_series], [axial_slice, sagittal_slice])):
                    series_id = series.series_id
                    for idx, row in descriptions.iterrows():
                        point_coords, is_inside = patient_coords_to_image_2d(row['patient_coords'],
                                                                             series.get_instance(instance_number),
                                                                             return_if_contains=True)
                        if is_inside:
                            points.append(ax[i].add_patch(_choose_patch((point_coords[0], point_coords[1]), condition=row["condition_unsided"],
                                                                         color=value_to_color[row["value"]], radius=5))
                        )
                        

                
            
        if draw_all_reference_lines:
            for i, meta in enumerate(axial_meta):
                line_p1 = patient_coords_to_image_2d(meta.position, sagittal_meta[sagittal_slice])
                line_p2 = patient_coords_to_image_2d(meta.position + meta.orientation[3:]*100, sagittal_meta[sagittal_slice])
                # ax[1].add_patch(plt.Circle((line_p1[0], line_p1[1])))
                if i == axial_slice:
                    color = 'red'
                else:
                    color = 'gray'
                line = ax[1].axline(xy1=(line_p1[0], line_p1[1]),
                             xy2=(line_p2[0], line_p2[1]),
                             ls='--',
                             color=color,
                             lw=0.7,
                             clip_on=True,
                             clip_box=TransformedBbox(Bbox([[0, 0], [1, 1]]), ax[1].transAxes)
                             )
                lines.append(line)
            for i,meta in enumerate(sagittal_meta):
                line_p1 = patient_coords_to_image_2d(meta.position, axial_meta[axial_slice])
                line_p2 = patient_coords_to_image_2d(meta.position + meta.orientation[:3]*100, axial_meta[axial_slice])
                if i == sagittal_slice:
                    color = 'red'
                else:
                    color = 'gray'
                
                line = ax[0].axline(xy1=(line_p1[0], line_p1[1]),
                             xy2=(line_p2[0], line_p2[1]),
                             ls='--',
                             color=color,
                             lw=0.7,
                             clip_on=True,
                             clip_box=TransformedBbox(Bbox([[0, 0], [1, 1]]), ax[0].transAxes))
                lines.append(line)



        fig.canvas.draw_idle()

        # Show the specfied slice on the axial plane with 'gray' color-map and axial aspect ratio

    #interactive = widgets.interactive(simple_interactive_inner,
    #                                  axial_slice=widgets.IntSlider(0, axial.shape[0]-1, 1, 0),
    #                                  sagittal_slice=widgets.IntSlider(0, sagittal.shape[0]-1, 1, 0))
    interactive = widgets.interactive(simple_interactive_inner,
                                      axial_slice=widgets.IntSlider(min=0, max=axial.shape[0]-1, value=0),
                                      sagittal_slice=widgets.IntSlider(min=0, max=sagittal.shape[0]-1, value=0),
                                      layout=widgets.ToggleButtons(options=['1x2', '2x1']),
                                      draw_points=widgets.Checkbox(value=True, description="Draw points"),
                                      draw_points_opts=widgets.Dropdown(options=["Listed", "Projection"], value="Projection"),
                                      draw_all_reference_lines=widgets.Checkbox(value=False, description='Draw all reference lines')
                                      )

    return interactive



def interactive_visualization(
    np_axial: np.ndarray,
    np_sagittal: np.ndarray
    # axial_df: 'pd.DataFrame',
    # sagittal_df: 'pd.DataFrame'
) -> None:
    """Create an interactive visualization of spine axial and sagittal views.

    This function generates a matplotlib figure with ipywidgets for interactivity
    to explore axial and sagittal views of spine imaging data. It allows
    the user to fix one view and animate the other, with options for slider
    control or automatic animation.

    Args:
        np_axial (np.ndarray): 3D numpy array of axial view images.
        np_sagittal (np.ndarray): 3D numpy array of sagittal view images.
        axial_df (pd.DataFrame): DataFrame containing axial metadata.
        sagittal_df (pd.DataFrame): DataFrame containing sagittal metadata.
    """
    fig, (ax_sagittal, ax_axial) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Enhanced Interactive Spine Visualization', fontsize=16)

    # Sagittal View
    ax_sagittal.set_title('Sagittal View')
    sagittal_img = ax_sagittal.imshow(np_sagittal[len(np_sagittal)//2], cmap='gray', aspect='auto')
    ax_sagittal.set_xlabel('Anterior - Posterior')
    ax_sagittal.set_ylabel('Superior - Inferior')
    # sagittal_line = ax_sagittal.axhline(y=np_axial.shape[1]//2, color='r', linestyle='--')

    # Axial View
    ax_axial.set_title('Axial View')
    axial_img = ax_axial.imshow(np_axial[len(np_axial)//2], cmap='gray', aspect='auto')
    ax_axial.set_xlabel('Left - Right')
    ax_axial.set_ylabel('Anterior - Posterior')
    # axial_line = ax_axial.axvline(x=np_sagittal.shape[2]//2, color='r', linestyle='--')

    # Add text annotations for patient information
    # fig.text(0.01, 0.99, f"Patient ID: {axial_df['study_id'].iloc[0]}", ha='left', va='top')
    # fig.text(0.01, 0.97, f"Axial Series: {axial_df['series_description'].iloc[0]}", ha='left', va='top')
    # fig.text(0.01, 0.95, f"Sagittal Series: {sagittal_df['series_description'].iloc[0]}", ha='left', va='top')

    plt.tight_layout()

    # Create ipywidgets
    slice_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max(len(np_axial), len(np_sagittal)) - 1,
        step=1,
        description='Slice:',
        continuous_update=False
    )

    radio_buttons = widgets.RadioButtons(
        options=['Animate Axial', 'Animate Sagittal'],
        description='View to Animate:',
        disabled=False
    )

    animation_checkbox = widgets.Checkbox(
        value=False,
        description='Animation Mode',
        disabled=False
    )

    play_button = widgets.Play(
        value=0,
        min=0,
        max=max(len(np_axial), len(np_sagittal)) - 1,
        step=1,
        interval=200,
        description="Press play",
        disabled=False
    )

    widgets.jslink((play_button, 'value'), (slice_slider, 'value'))

    def update(change):
        slice_num = slice_slider.value
        
        if radio_buttons.value == 'Animate Axial':
            axial_img.set_array(np_axial[slice_num])
            #axial_line.set_xdata(np_sagittal.shape[2]//2)
            # sagittal_line.set_ydata(slice_num % np_axial.shape[0])
            ax_axial.set_title(f'Axial View (Slice {slice_num + 1}/{len(np_axial)})')
            ax_sagittal.set_title('Sagittal View (Fixed)')
        else:
            sagittal_img.set_array(np_sagittal[slice_num % len(np_sagittal)])
            # sagittal_line.set_ydata(np_axial.shape[1]//2)
            # axial_line.set_xdata(slice_num % np_sagittal.shape[0])
            ax_sagittal.set_title(f'Sagittal View (Slice {slice_num + 1}/{len(np_sagittal)})')
            ax_axial.set_title('Axial View (Fixed)')  
        fig.canvas.draw_idle()
        

    def on_radio_change(change):
        if change['new'] == 'Animate Axial':
            sagittal_img.set_array(np_sagittal[len(np_sagittal)//2])
            slice_slider.max = len(np_axial) - 1
            play_button.max = len(np_axial) - 1
        else:
            axial_img.set_array(np_axial[len(np_axial)//2])
            slice_slider.max = len(np_sagittal) - 1
            play_button.max = len(np_sagittal) - 1
        slice_slider.value = 0
        play_button.value = 0
        update({'new': slice_slider.value})

    def on_animation_mode_change(change):
        slice_slider.disabled = change['new']
        play_button.disabled = not change['new']

    slice_slider.observe(update, names='value')
    radio_buttons.observe(on_radio_change, names='value')
    animation_checkbox.observe(on_animation_mode_change, names='value')

    # Display the widgets and the plot
    return fig, widgets.VBox([radio_buttons, animation_checkbox, 
                widgets.HBox([slice_slider, play_button])])
    # plt.show()
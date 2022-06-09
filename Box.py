import matplotlib.pyplot as plt
import pandas as pd
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

def save_plot_box_svg(csv_path='figure4.csv'):

    # Read CSV of metrics into Pandas dataframe
    df = pd.read_csv(csv_path, index_col=0, header=0).T

    # Open plot
    plt.figure()

    # Plot Pandas dataframe
    df.boxplot(grid=False, rot=30)

    # Plot configuration
    plt.ylabel('F1-Measure')
    # plt.xlabel('time duration (s)')# fig 13-14

    if 'window' in csv_path:
        plt.xlabel('Block Size (Frames)')# fig 15-16

    # Save plot
    plt.savefig(csv_path.replace('.csv', '_box.svg'), format='svg', bbox_inches='tight')

    # plt.show()
    return 0

def trans_svg_pdf(svg_dir='svg_plots'):
    for item in os.listdir(svg_dir):
        if '.svg' in item:
            svg_path = os.path.join(svg_dir, item)
            drawing = svg2rlg(svg_path)
            renderPDF.drawToFile(drawing, svg_path.replace('.svg','.pdf'))
            renderPM.drawToFile(drawing, svg_path.replace('.svg','.png'), fmt="PNG")
    return 0

if __name__ == '__main__':

    # Plot directory
    csv_dir = 'results'

    # Plot each model trained
    for item in os.listdir(csv_dir):

        # If file is CSV
        if '.csv' in item:

            # Get CSV path
            csv_path = os.path.join(csv_dir, item)

             # Plot bar
            save_plot_box_svg(csv_path)

    trans_svg_pdf(csv_dir)

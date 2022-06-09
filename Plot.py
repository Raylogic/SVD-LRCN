import matplotlib.pyplot as plt
import pandas as pd
import os

# Plot bar of metrics
def plot_bar(csv_path='figure4.csv', y_min=0.75, y_max=1):

    # Read CSV of metrics into Pandas dataframe
    df = pd.read_csv(csv_path, index_col=0, on_bad_lines='skip').T

    # Open plot
    plt.figure()

    print(df.dtypes)

    # Plot Pandas dataframe
    df.plot.bar(rot=0)

    # Plot configuration
    plt.ylim(y_min, y_max)
    plt.legend(loc='left', bbox_to_anchor=(1, 1))

    # Save plot
    plt.savefig(csv_path.replace('.csv', '_bar.svg'), format='svg', bbox_inches='tight')

    # plt.show()
    return 0

# Plot line of metrics
def plot_line(csv_path='figure5.csv', y_min=0.75, y_max=1):

    # Read CSV of metrics into Pandas dataframe
    df = pd.read_csv(csv_path, index_col=0, on_bad_lines='skip')

    # Open plot
    plt.figure()

    # Plot Pandas dataframe
    df.plot(rot=0)

    # Plot configuration
    plt.ylim(y_min, y_max)

    # Save plot
    plt.savefig(csv_path.replace('.csv', '_line.svg'), format='svg', bbox_inches='tight')
    
    # plt.show()
    return 0

# Insert plots into PDF 
def plot_to_pdf(svg_dir='plots'):

    # Traverse each plot file
    for item in os.listdir(svg_dir):

        # Detect plot file format
        if '.svg' in item:

            # Get plot path
            svg_path = os.path.join(svg_dir, item)

            # Transform the plot into PDF
            cmd_trans = 'inkscape -D -z --file=%s --export-pdf=%s' % (svg_path, svg_path.replace('.svg', '.pdf'))
            print(cmd_trans)

            # Change the file format to -pdf
            if not os.path.isfile(svg_path.replace('.svg', '.pdf')):
                os.system(cmd_trans)
    return 0


if __name__ == '__main__':
    
    # Plot directory
    csv_dir='plots'

    # Plot each model trained
    for item in os.listdir(csv_dir):

        # Get CSV path
        csv_path = os.path.join(csv_dir, item)

        # Read CSV into Pandas Dataframe
        df_data = pd.read_csv(csv_path, index_col=0, on_bad_lines='skip')

        print(df_data)

        # Get min value from dataframe
        # min_value_as_low=df_data.min()-0.03
        
        # Plor bar/line
        plot_bar(csv_path, 0, 1)
        # plot_line(csv_path, min_value_as_low, 1)

    # Move plot to PDF
    plot_to_pdf(csv_dir)

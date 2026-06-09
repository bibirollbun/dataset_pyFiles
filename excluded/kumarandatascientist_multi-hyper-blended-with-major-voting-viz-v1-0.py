import pandas as pd
no_model = pd.read_csv('/kaggle/input/eda-and-using-previous-ratio-as-prediction/submission.csv') 
linear_reg=pd.read_csv('/kaggle/input/linearregression-subtract-bias-makes-lb-0-054/submission.csv') 
lb_best = pd.read_csv('/kaggle/input/123213432432432/submission (30).csv') # 0.05318


blended1 = no_model.copy()

blended1['num_sold'] = (
    (0.11) * no_model['num_sold'] +
    (0.09) * linear_reg['num_sold'] +
    (0.80) * lb_best['num_sold'] 
)*1.003187
# Save the blended results
blended1.to_csv('submission1.csv', index=False)

blended1.head(10)


new_blended_best1 =  pd.read_csv('/kaggle/working/submission1.csv')


blended2 = new_blended_best1.copy()

blended2['num_sold'] = (
    (0.11) * no_model['num_sold'] +
    (0.09) * linear_reg['num_sold'] +
    (0.80) * new_blended_best1['num_sold'] 
)*1.003187
# Save the blended results
blended2.to_csv('submission2.csv', index=False)

blended2.head(10)



new_blended_best2 =  pd.read_csv('/kaggle/working/submission2.csv')


blended3 = new_blended_best2.copy()

blended3['num_sold'] = (
    (0.11) * no_model['num_sold'] +
    (0.09) * linear_reg['num_sold'] +
    (0.80) * new_blended_best2['num_sold'] 
)*1.003187
# Save the blended results
blended3.to_csv('submission3.csv', index=False)

blended3.head(10)



new_blended_best3 =  pd.read_csv('/kaggle/working/submission3.csv')


blended4 = new_blended_best3.copy()

blended4['num_sold'] = (
    (0.11) * no_model['num_sold'] +
    (0.09) * linear_reg['num_sold'] +
    (0.80) * new_blended_best3['num_sold'] 
)*1.003187
# Save the blended results
blended4.to_csv('submission4.csv', index=False)

blended4.head(10)


from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook

output_notebook()

# Convert DataFrames to ColumnDataSource for Bokeh
source1 = ColumnDataSource(blended1.head(50))
source2 = ColumnDataSource(blended2.head(50))
source3 = ColumnDataSource(blended3.head(50))
source4 = ColumnDataSource(blended4.head(50))

# Create a Bokeh figure
p = figure(
    title="Interactive Plot Comparison",
    x_axis_label='id',
    y_axis_label='num_sold',
    width=900,
    height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="pan",
    active_scroll="wheel_zoom"
)

# Add lines for each DataFrame
p.line('id', 'num_sold', source=source1, legend_label='DF1', line_width=2, color='blue')
p.line('id', 'num_sold', source=source2, legend_label='DF2', line_width=2, color='green')
p.line('id', 'num_sold', source=source3, legend_label='DF3', line_width=2, color='red')
p.line('id', 'num_sold', source=source4, legend_label='DF4', line_width=2, color='purple')

# Update legend and show the plot
p.legend.title = "Source"
show(p)



import pandas as pd

class MajorityVoting:
    def __init__(self, sub1, sub2, sub3, sub4):
        # Initialize submissions and sort by 'id'
        self.sub1 = sub1.sort_values(by='id').reset_index(drop=True)
        self.sub2 = sub2.sort_values(by='id').reset_index(drop=True)
        self.sub3 = sub3.sort_values(by='id').reset_index(drop=True)
        self.sub3 = sub4.sort_values(by='id').reset_index(drop=True)

    def majority_vote(self, row):
        # Determine the most frequent value (mode)
        return row.mode()[0]

    def process_voting(self):
        # Combine the submissions into one dataframe
        combined = pd.DataFrame({
            'id': self.sub1['id'],
            'num_sold_1': self.sub1['num_sold'],
            'num_sold_2': self.sub2['num_sold'],
            'num_sold_3': self.sub3['num_sold'],
            'num_sold_4': self.sub3['num_sold']
        })

        # Apply majority voting
        combined['final_num_sold'] = combined[['num_sold_1', 'num_sold_2', 'num_sold_3','num_sold_4']].apply(self.majority_vote, axis=1)

        # Create final submission dataframe
        final_submission = combined[['id', 'final_num_sold']].rename(columns={'final_num_sold': 'num_sold'})
        return final_submission

    def save_to_csv(self, final_submission, filename='submission.csv'):
        # Save final submission to a CSV file
        final_submission.to_csv(filename, index=False)
        print(f"Majority voting completed and saved to '{filename}'")





# Usage
voting = MajorityVoting(blended1, blended2, blended3, blended4)
final_submission = voting.process_voting()
voting.save_to_csv(final_submission)
final_submission.head(10)


from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.io import output_notebook

output_notebook()

# Add adjusted x-offset columns for bar plotting
blended1['x'] = blended1['id'] - 0.3
blended2['x'] = blended2['id'] - 0.1
blended3['x'] = blended3['id'] + 0.1
blended4['x'] = blended4['id'] + 0.3
final_submission['x'] = final_submission['id'] + 0.5

# Convert DataFrames to ColumnDataSource
source1 = ColumnDataSource(blended1.head(50))
source2 = ColumnDataSource(blended2.head(50))
source3 = ColumnDataSource(blended3.head(50))
source4 = ColumnDataSource(blended4.head(50))
source5 = ColumnDataSource(final_submission.head(50))

# Create Line Plot
line_plot = figure(
    title="Line Plot Comparison",
    x_axis_label='id',
    y_axis_label='num_sold',
    width=800,
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_drag="pan",
    active_scroll="wheel_zoom",
)

# Add lines for each DataFrame
line_plot.line('id', 'num_sold', source=source1, legend_label='DF1', line_width=2, color='blue')
line_plot.line('id', 'num_sold', source=source2, legend_label='DF2', line_width=2, color='green')
line_plot.line('id', 'num_sold', source=source3, legend_label='DF3', line_width=2, color='red')
line_plot.line('id', 'num_sold', source=source4, legend_label='DF4', line_width=2, color='purple')
line_plot.line('id', 'num_sold', source=source5, legend_label='DF5', line_width=2, color='black')

# Configure Line Plot Legend
line_plot.legend.title = "Source"

# Create Bar Plot
bar_plot = figure(
    title="Bar Plot Comparison",
    x_axis_label='id',
    y_axis_label='num_sold',
    width=800,
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_drag="pan",
    active_scroll="wheel_zoom",
)

# Add bars for each DataFrame
bar_plot.vbar(x='x', top='num_sold', source=source1, width=0.2, color='blue', legend_label='DF1')
bar_plot.vbar(x='x', top='num_sold', source=source2, width=0.2, color='green', legend_label='DF2')
bar_plot.vbar(x='x', top='num_sold', source=source3, width=0.2, color='red', legend_label='DF3')
bar_plot.vbar(x='x', top='num_sold', source=source4, width=0.2, color='purple', legend_label='DF4')
bar_plot.vbar(x='x', top='num_sold', source=source5, width=0.2, color='black', legend_label='DF5')

# Configure Bar Plot Legend
bar_plot.legend.title = "Source"

# Arrange the plots in a grid layout
layout = gridplot([[line_plot], [bar_plot]])

# Show the plots
show(layout)






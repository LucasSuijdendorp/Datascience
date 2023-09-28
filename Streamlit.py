import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import geopandas as gpd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
from kaggle.api.kaggle_api_extended import KaggleApi
from scipy.stats import linregress
import plotly.express as px
import os
kaggle_data={"username":"lucassuijdendorp","key":"fbbe2173764ce79893ee99215d580f15"}
os.environ['KAGGLE_USERNAME']=kaggle_data["username"]
os.environ['KAGGLE_KEY']=kaggle_data["key"]
#Haal data op van Kaggle
api = KaggleApi()
api.authenticate()

# Download all files of a dataset
# Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
api.dataset_download_files('joebeachcapital/school-shootings', unzip=True)
# Create your state_df DataFrame here

state_df = pd.read_csv('school-shootings-data.csv')


st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Select a page:", ["Home", "Base statistics", "Casualties", "Deaths vs unemployment", "Age distribution", "Weapons", "Shooter"])

# Definieer gegevens voor elke pagina
if selected_page == "Home":
	st.title('School shootings')
	st.text('Dashboard about statistics on American school shootings')
	foto = Image.open('SchoolShooting.jpg')
	st.image(foto, caption='Robb elementary incident 2022')

elif selected_page == "Base statistics":
	dataBS = pd.DataFrame({'year': list(range(1999, 2023)),  # Years from 1999 to 2022
		'cc': [7, 12, 13, 5, 12, 9, 13, 15, 10, 9, 9, 9, 7, 11, 13, 16, 7, 13, 15, 30, 27, 9, 42, 46]})

	# Create a Streamlit app
	st.title('Number of Shootings per Year')

	# Checkbox to toggle trendline visibility, initially set to False
	show_trendline = st.checkbox('Trendline', value=False)

	# Calculate the trend line only once
	slope, intercept, r_value, p_value, std_err = linregress(dataBS['year'], dataBS['cc'])
	trend_line = intercept + slope * dataBS['year']

	# Create a line plot with points
	fig, ax = plt.subplots(figsize=(12, 6))
	line, = ax.plot(dataBS['year'], dataBS['cc'], marker='o', linestyle='-', color='blue', label='Number of Shootings')

	# Add labels and title
	ax.set_title('Number of Shootings per Year')
	ax.set_xlabel('Year')
	ax.set_ylabel('Number of Shootings')

	# Add data labels with a different color and larger font size, initially hidden
	labels = []
	for i, cc in enumerate(dataBS['cc']):
		label = ax.text(dataBS['year'][i], cc, str(cc), ha='center', va='bottom', fontweight='bold', fontsize=12, color='red', visible=False)
		labels.append(label)

	# Show legend
	ax.legend()

	# Plot the trend line based on the checkbox value
	if show_trendline:
		ax.plot(dataBS['year'], trend_line, linestyle='--', color='green', label=f'Trend Line (R-squared = {r_value**2:.2f})')

		# Extend the x-axis to include future years (till 2040)
		future_years = np.array(list(range(1999, 2041)))  # Convert to a NumPy array
		extended_trend_line = intercept + slope * future_years
		ax.plot(future_years, extended_trend_line, linestyle='--', color='red', label='Trend Line (Extended)')
		
	# Function to toggle visibility of data labels
	def toggle_labels(event):
		for label in labels:
			label.set_visible(not label.get_visible())
		fig.canvas.draw()

	# Connect the toggle_labels function to a key press event (e.g., 'T' key)
	fig.canvas.mpl_connect('key_press_event', toggle_labels)

	# Show the plot
	ax.grid()
	st.pyplot(fig)
	
	# Define the order of days of the week (excluding Saturday and Sunday)
	day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

	# Create a custom slider to select the day of the week, including an "All Days" option
	day_options = ["All Days"] + day_order
	selected_day_index = st.selectbox("Select a Day of the Week", day_options)

	# Display the selected day
	selected_day = selected_day_index
	
	# Check if "All Days" is selected or a specific day
	if selected_day == "All Days":
		filtered_data = state_df  # Use all data for "All Days"
	else:
		filtered_data = state_df[state_df['day_of_week'] == selected_day]
	
	# Create the scatter_geo plot using Plotly Express
	fig2 = px.scatter_geo(
		filtered_data,
		lat='lat',
		lon='long',
		projection='albers usa',
		size='casualties',
		color='day_of_week',
		hover_name='school_name',
		title=f'School Shootings on {selected_day}',
		category_orders={"day_of_week": day_order}  # Specify the desired order
	)
	
	# Use Streamlit to display the plot
	st.plotly_chart(fig2)
	
	df = pd.read_csv('school-shootings-data.csv')

	# Create a Streamlit app
	st.title('Number of Shootings by School Type')

	# Group the data by school_type and count the number of shootings
	school_type_counts = state_df['school_type'].value_counts().sort_values(ascending=False)

	# Create a bar plot using Matplotlib
	fig, ax = plt.subplots(figsize=(10, 6))
	school_type_counts.plot(kind='bar', ax=ax, color='royalblue')

	# Add labels and title
	ax.set_title('Number of Shootings by School Type')
	ax.set_xlabel('School Type')
	ax.set_ylabel('Number of Shootings')

	# Rotate x-axis labels for better readability
	plt.xticks(rotation=45, ha='right')
	
	# Show the plot
	st.pyplot(fig)
	

	state_df['gender_shooter1'] = state_df['gender_shooter1'].replace({'m': 'Male', 'f': 'Female'})
	filtered_df = state_df[state_df['gender_shooter1'].isin(['Male', 'Female'])]

	# Create a Streamlit app
	st.title('Number of Shootings by Gender of Shooter')

	# Group the data by gender_shooter1 and count the number of shootings
	gender_counts = filtered_df['gender_shooter1'].value_counts().sort_values(ascending=False)

	# Create a bar plot using Matplotlib
	fig, ax = plt.subplots(figsize=(10, 6))
	gender_counts.plot(kind='bar', ax=ax, color='royalblue')

	# Add labels and title
	ax.set_title('Number of Shootings by Gender of Shooter')
	ax.set_xlabel('Gender of Shooter')
	ax.set_ylabel('Number of Shootings')

	# Rotate x-axis labels for better readability
	plt.xticks(rotation=0)

	# Show the plot
	st.pyplot(fig)

elif selected_page == "Casualties":

	# Set the title of the app
	st.title('Number of deaths, injured and casualties caused by school shootings per year')

	# Create a slider for selecting a year
	selected_year = st.slider('Select a Year', min_value=state_df['year'].min(), max_value=state_df['year'].max())

	# Filter the DataFrame based on the selected year
	filtered_df = state_df[state_df['year'] == selected_year]

	# Filter out states with no deaths, casualties, or injured
	filtered_df = filtered_df[(filtered_df['killed'] > 0) | (filtered_df['injured'] > 0) | (filtered_df['casualties'] > 0)]

	# Create a Plotly bar plot based on the filtered DataFrame
	fig = px.bar(filtered_df, x='state', y='killed', title=f'Deaths in {selected_year}')
	fig.update_xaxes(categoryorder='total descending')  # Sort x-axis labels

	fig2 = px.bar(filtered_df, x='state', y='injured', title=f'Injured in {selected_year}')
	fig2.update_xaxes(categoryorder='total descending')  # Sort x-axis labels

	fig3 = px.bar(filtered_df, x='state', y='casualties', title=f'Casualties in {selected_year}')
	fig3.update_xaxes(categoryorder='total descending')  # Sort x-axis labels

	# Set the y-axis range with a minimum of 0 and maximum of 35
	fig.update_yaxes(range=[0, 35])
	fig2.update_yaxes(range=[0, 35])
	fig3.update_yaxes(range=[0, 35])

	# Display the Plotly plots in Streamlit
	st.plotly_chart(fig)
	st.plotly_chart(fig2)
	st.plotly_chart(fig3)

elif selected_page == "Deaths vs unemployment":
	
	state_killed_totals = state_df.groupby('state')['killed'].sum().reset_index()

	# Rename the columns for clarity
	state_killed_totals.columns = ['State', 'Total_Killed']

	state_name_to_abbreviation = {
		'Alabama': 'AL',
		'Alaska': 'AK',
		'Arizona': 'AZ',
		'Arkansas': 'AR',
		'California': 'CA',
		'Colorado': 'CO',
		'Connecticut': 'CT',
		'Delaware': 'DE',
		'Florida': 'FL',
		'Georgia': 'GA',
		'Hawaii': 'HI',
		'Idaho': 'ID',
		'Illinois': 'IL',
		'Indiana': 'IN',
		'Iowa': 'IA',
		'Kansas': 'KS',
		'Kentucky': 'KY',
		'Louisiana': 'LA',
		'Maine': 'ME',
		'Maryland': 'MD',
		'Massachusetts': 'MA',
		'Michigan': 'MI',
		'Minnesota': 'MN',
		'Mississippi': 'MS',
		'Missouri': 'MO',
		'Montana': 'MT',
		'Nebraska': 'NE',
		'Nevada': 'NV',
		'New Hampshire': 'NH',
		'New Jersey': 'NJ',
		'New Mexico': 'NM',
		'New York': 'NY',
		'North Carolina': 'NC',
		'North Dakota': 'ND',
		'Ohio': 'OH',
		'Oklahoma': 'OK',
		'Oregon': 'OR',
		'Pennsylvania': 'PA',
		'Rhode Island': 'RI',
		'South Carolina': 'SC',
		'South Dakota': 'SD',
		'Tennessee': 'TN',
		'Texas': 'TX',
		'Utah': 'UT',
		'Vermont': 'VT',
		'Virginia': 'VA',
		'Washington': 'WA',
		'West Virginia': 'WV',
		'Wisconsin': 'WI',
		'Wyoming': 'WY'
	}

	# Function to convert state names to abbreviations
	def convert_state_name_to_abbreviation(state_name):
		return state_name_to_abbreviation.get(state_name, state_name)

	# Apply the function to the DataFrame
	state_killed_totals['State'] = state_killed_totals['State'].apply(convert_state_name_to_abbreviation)

    
	# Create the choropleth map
	fig1 = px.choropleth(state_killed_totals,
						locations="State",
						color="Total_Killed",
						color_continuous_scale="viridis",
						locationmode="USA-states",
						labels={'Total_Killed':'Total amount of people killed'})
	fig1.update_layout(
		title_text="State Rankings",
		geo_scope="usa"
	)

	# Display the choropleth map in Streamlit
	st.plotly_chart(fig1)

	# Load your dataset (assuming it's in a CSV file named 'your_dataset.csv')
	unemploymentData = pd.read_csv('Unemployment.csv')

	# Filter the dataset for the years between 1999 and 2022
	filtered_data = unemploymentData[(unemploymentData['Year'] >= 1999) & (unemploymentData['Year'] <= 2022)]

	# Group the data by state and calculate the average unemployment rate
	average_unemployment = filtered_data.groupby('State/Area')['Percent (%) of Labor Force Unemployed in State/Area'].mean().reset_index()

	average_unemployment['State/Area'] = average_unemployment['State/Area'].apply(convert_state_name_to_abbreviation)

	# Create the choropleth map
	fig2 = px.choropleth(average_unemployment,
						locations="State/Area",
						color="Percent (%) of Labor Force Unemployed in State/Area",
						color_continuous_scale="viridis",
						locationmode="USA-states",
						labels={'Total_Unemployed':'Total amount of people unemployd'})
	fig2.update_layout(
		title_text="State Rankings",
		geo_scope="usa"
	)

	# Display the choropleth map in Streamlit
	st.plotly_chart(fig2)
	
	st.text('We could not find a connection between unemployment and casualties in school shootings')



elif selected_page == "Age distribution":
	# Create a Streamlit app
	st.title("Age Distribution of Shooters by State")
	st.sidebar.header("Filter")
	
	# Create a dropdown to select the state, including an "All states" option, sorted alphabetically
	state_options = sorted(list(state_df['state'].unique()))
	state_options.insert(0, "All states")  # Insert "All states" as the first option
	selected_state = st.sidebar.selectbox("Select State", state_options)
	
	# Filter the data based on the selected state
	if selected_state == "All states":
		filtered_data = state_df
	else:
		filtered_data = state_df[state_df['state'] == selected_state]
	
	# Check if there is shooter age data for the selected state or all states
	if not filtered_data.empty and not filtered_data['age_shooter1'].isna().all():
		if selected_state == "All states":
			title = "Age Distribution of Shooters in All States"
		else:
			title = f"Age Distribution of Shooters in {selected_state}"
        
		st.write(title)
		fig, ax = plt.subplots(figsize=(10, 6))
		ax.hist(filtered_data['age_shooter1'].dropna(), bins=20, edgecolor='k')
		ax.set_xlabel("Age")
		ax.set_ylabel("Number of Shooters")
		ax.set_title(title)
		st.pyplot(fig)
	else:
		if selected_state == "All states":
			st.write("No age data available for any state.")
		else:
			st.write(f"No age data available for {selected_state}")
	
	
	df_unemployment = pd.read_csv('Unemployment.csv')
	
	


elif selected_page == "Weapons":
	# Split the "gun_type" column to count individual guns
	guns = state_df['weapon'].str.split('|', expand=True).stack().str.strip()

	# Get the top 10 most used guns in descending order
	top_10_guns = guns.value_counts().head(10)[::-1]

	# Create a Streamlit app
	st.title('Top 10 Most Used Guns in School Shootings')

	# Create a horizontal bar plot using Matplotlib
	fig, ax = plt.subplots(figsize=(10, 6))
	top_10_guns.plot(kind='barh', ax=ax, color='royalblue')

	# Add labels and title
	ax.set_title('Top 10 Most Used Guns in School Shootings')
	ax.set_xlabel('Number of Occurrences')
	ax.set_ylabel('Gun Type')

	# Show the plot
	st.pyplot(fig)

elif selected_page == "Shooter":
    st.header('shooting type')
    
# Bereken de percentages
    total_count = len(state_df)
    shooting_type_counts = state_df['shooting_type'].value_counts()
    percentages = shooting_type_counts / total_count * 100

# Maak een staafdiagram
    fig, ax = plt.subplots()
    ax.bar(percentages.index, percentages.values)
    ax.set_xlabel('Shooting Type')
    ax.set_ylabel('Percentage')
    ax.set_title('Shooting type overview in percentages')
    ax.set_xticklabels(percentages.index, rotation=45, ha='right')
    
    # Voeg de percentages boven de balken toe
    for i, percentage in enumerate(percentages):
        ax.annotate(f'{percentage:.2f}%', 
                xy=(i, percentage), 
                xytext=(0, 5), 
                textcoords='offset points', 
                ha='center', 
                va='bottom')
        ax.set_ylim(0, 100)
        
    # Toon de plot in Streamlit
    st.pyplot(fig)
    
    
# Toevoegen van een selectbox voor 'state'
    selected_state = st.selectbox("Selecteer een staat (state):", state_df['state'].unique())

# Filter de dataset op basis van de geselecteerde staat
    filtered_df = state_df[state_df['state'] == selected_state]

    st.subheader(f"Data for the state: {selected_state}")

# Toon de eerste paar rijen van de gefilterde dataset
    st.write(filtered_df.head())

# Aantal incidenten per 'shooting_type' voor de geselecteerde staat
    incidenten_per_type = filtered_df['shooting_type'].value_counts()

# Maak een barplot met matplotlib
    fig, ax = plt.subplots()
    incidenten_per_type.plot(kind='bar', ax=ax)

# Optioneel: labels en titel voor de plot
    plt.xlabel('Shooting Type')
    plt.ylabel('Number of incidents')
    plt.title(f'Number of incidents per "shooting_type" in {selected_state}')
    plt.xticks(rotation=45) 
# Toon de plot in Streamlit
    st.pyplot(fig)

# Voeg een selectbox toe om 'shooter_relationship1' te kiezen
    selected_relationship = st.selectbox("Select a shooter_relationship1:", state_df['shooter_relationship1'].unique())

# Filter de dataset op basis van de geselecteerde shooter_relationship1
    filtered_df = state_df[state_df['shooter_relationship1'] == selected_relationship]

    st.subheader(f"Data for selected shooter relationship: {selected_relationship}")

# Toon de eerste paar rijen van de gefilterde dataset
    st.write(filtered_df.head())

# Groepeer de gegevens per combinatie van 'shooting_type' en 'shooter_relationship1'
    incidenten_per_combinatie = filtered_df.groupby(['shooting_type', 'shooter_relationship1']).size().unstack(fill_value=0)

# Maak een staafdiagram met matplotlib
    fig, ax = plt.subplots()
    incidenten_per_combinatie.plot(kind='bar', ax=ax, stacked=True, legend=True)
    
    # Voeg waarden toe aan de balken
    for p in ax.patches:
             ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, color='black', xytext=(0, 5), textcoords='offset points')

# Optioneel: labels en titel voor de plot
    plt.xlabel('Combination of shooting_type and shooter_relationship1')
    plt.ylabel('Number of incidents')
    plt.title(f'Numbers of incidents per shooting type & the shooter relation" for {selected_relationship}')
    plt.xticks(rotation=45)
    plt.gcf().set_size_inches(12, 6)
    plt.yticks(range(0, int(max_y) + 1))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    st.pyplot(fig)